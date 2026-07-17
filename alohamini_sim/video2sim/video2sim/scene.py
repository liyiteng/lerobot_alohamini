"""video2sim stage — Isaac scene assembly: splat + collider + robot -> USD.

What: builds the final Isaac Sim stage from the pipeline outputs:
  - the NuRec splat USDZ referenced under one Xform that carries the
    align-npz transform (raw LingBot frame -> upright, metric, floor z=0);
  - the decimated TSDF mesh (V/F npz from ``video2sim.scene_prep``) as an
    INVISIBLE static UsdPhysics collider, placed with an identity transform
    (the npz mesh is already metric/aligned — only the splat needs the
    align transform);
  - the robot USD referenced at the origin for scale;
  - a splat-safe dome light.

Transform convention: Gf.Matrix4d is ROW-VECTOR (p' = p @ M), so the align
matrix is built as ``M[:3, :3] = (R * scale).T`` with the translation in the
BOTTOM row ``M[3, :3] = [0, 0, -floorz * scale]``.

Lighting post-mortem: dome 1200+2000 (the mesh-room recipe) overexposes the
NuRec splat to white blur; the approved GUI view used dome ~350-450. Default
here is 450 (splat-safe).

Pipeline position: final stage — after ``video2sim.export`` (NuRec USDZ),
``video2sim.refine`` (align npz) and ``video2sim.scene_prep`` (collider npz).

Interpreter contract (do NOT auto-switch): runs under the Isaac venv
/home/perelman/isaac5-venv/bin/python (isaacsim + pxr). GPU must be exclusive
enough for RTX rendering — the Isaac GUI itself eats 3-4 GB. The collider npz
is produced beforehand under the MAIN venv via ``python -m
video2sim.scene_prep`` (open3d does not exist in the Isaac venv).

Usage:
  /home/perelman/isaac5-venv/bin/python -m video2sim.scene [--workdir W]
      [--usdz W/export/scene.usdz] [--align-npz W/final_align.npz]
      [--collider-npz W/collider.npz] [--robot R.usd]
      [--dome-intensity 450] [--save-usd W/scene/scene.usd] [--gui]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

TAG = "[scene]"

# NOTE: the old aloha_mini_pro/robot.usd is DEPRECATED — use the parallel-
# gripper AM2 Pro model shipped with this package.
DEFAULT_ROBOT = (
    Path(__file__).resolve().parent.parent
    / "assets" / "am2pro_parallel" / "alohamini2pro_parallel.usd"
)

# Splat-safe dome intensity: 1200+ overexposes the NuRec splat to white blur
# (approved GUI views used ~350-450).
DEFAULT_DOME_INTENSITY = 450.0


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


def align_matrix(align_npz: Path) -> np.ndarray:
    """Row-vector 4x4 (p' = p @ M) from a refine align npz (R/floorz/scale).

    Maps the raw LingBot frame to upright + metric with the floor at z=0.
    """
    al = np.load(align_npz)
    R, floorz, scale = al["R"], float(al["floorz"]), float(al["scale"])
    M = np.eye(4)
    M[:3, :3] = (R * scale).T                    # Gf row-vector: p' = p @ M
    M[3, :3] = np.array([0.0, 0.0, -floorz * scale])
    return M


def build_scene(
    usdz: Path,
    align_npz: Path,
    collider_npz: Path,
    robot: Path,
    save_usd: Path,
    dome_intensity: float = DEFAULT_DOME_INTENSITY,
    gui: bool = False,
) -> Path:
    """Assemble splat + collider + robot into a USD stage; save to ``save_usd``.

    Starts a SimulationApp (headless unless ``gui``); with ``gui`` the app
    stays up until the window is closed. Returns the saved USD path.
    """
    for p, what in ((usdz, "NuRec usdz"), (align_npz, "align npz"),
                    (collider_npz, "collider npz"), (robot, "robot usd")):
        if not p.exists():
            raise SystemExit(f"{TAG} ERROR: {what} not found: {p}")

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp  # noqa: PLC0415 — must precede pxr/omni

    app = SimulationApp({"headless": not gui, "renderer": "RayTracedLighting",
                         "width": 1600, "height": 1000})

    import omni.usd  # noqa: E402,PLC0415
    from omni.isaac.core.utils.extensions import enable_extension  # noqa: E402,PLC0415
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, Vt  # noqa: E402,PLC0415

    # NuRec prim schemas (omni_nurec_types) are required to render the splat.
    enable_extension("omni.usd.schema.omni_nurec_types")

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # splat lives in the raw LingBot frame -> apply the refine alignment
    M = align_matrix(align_npz)
    env = stage.DefinePrim("/World/Env", "Xform")
    env.GetReferences().AddReference(str(usdz.resolve()))
    xf = UsdGeom.Xformable(env)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(*[float(v) for v in M.flatten()]))
    _log(f"splat referenced: {usdz.name} (aligned, floor z=0)")

    # refined TSDF mesh (already metric/aligned) = invisible static collider;
    # CollisionAPI without RigidBodyAPI -> static.
    cd = np.load(collider_npz)
    V = np.asarray(cd["V"], dtype=np.float32)
    F = np.asarray(cd["F"], dtype=np.int32)
    col = UsdGeom.Mesh.Define(stage, "/World/EnvCollider")
    col.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V))
    col.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(F)))
    col.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.reshape(-1)))
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    col.MakeInvisible()
    _log(f"collider: {len(V)} verts / {len(F)} tris (invisible, static)")

    robot_prim = stage.DefinePrim("/World/Robot", "Xform")
    robot_prim.GetReferences().AddReference(str(robot.resolve()))
    _log(f"robot referenced: {robot}")

    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(
        float(dome_intensity))

    save_usd.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(save_usd))
    _log(f"scene saved: {save_usd}")

    if gui:
        _log("splat + collider + robot up — close the window to exit")
        while app.is_running():
            app.update()
    app.close()
    return save_usd


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m video2sim.scene",
        description="video2sim: assemble splat + collider + robot Isaac scene "
                    "(run under /home/perelman/isaac5-venv/bin/python).",
    )
    ap.add_argument("--workdir", type=Path, default=Path("."),
                    help="pipeline working dir; default root for other paths")
    ap.add_argument("--usdz", type=Path, default=None,
                    help="NuRec splat usdz (default: WORKDIR/export/scene.usdz)")
    ap.add_argument("--align-npz", type=Path, default=None,
                    help="refine align npz with R/floorz/scale "
                         "(default: WORKDIR/final_align.npz)")
    ap.add_argument("--collider-npz", type=Path, default=None,
                    help="collider V/F npz from video2sim.scene_prep "
                         "(default: WORKDIR/collider.npz)")
    ap.add_argument("--robot", type=Path, default=DEFAULT_ROBOT,
                    help="robot USD to reference (default: AM2 Pro parallel-"
                         "gripper model; old aloha_mini_pro/robot.usd is "
                         "deprecated)")
    ap.add_argument("--dome-intensity", type=float,
                    default=DEFAULT_DOME_INTENSITY,
                    help="dome light intensity; 1200+ overexposes the NuRec "
                         f"splat to white blur (default: "
                         f"{DEFAULT_DOME_INTENSITY:g})")
    ap.add_argument("--save-usd", type=Path, default=None,
                    help="output USD path (default: WORKDIR/scene/scene.usd)")
    ap.add_argument("--gui", action="store_true",
                    help="open the Isaac GUI and keep it running "
                         "(default: headless scene-save-only)")
    args = ap.parse_args(argv)

    usdz = (args.usdz if args.usdz is not None
            else args.workdir / "export" / "scene.usdz")
    align_npz = (args.align_npz if args.align_npz is not None
                 else args.workdir / "final_align.npz")
    collider_npz = (args.collider_npz if args.collider_npz is not None
                    else args.workdir / "collider.npz")
    save_usd = (args.save_usd if args.save_usd is not None
                else args.workdir / "scene" / "scene.usd")

    build_scene(usdz, align_npz, collider_npz, args.robot, save_usd,
                dome_intensity=args.dome_intensity, gui=args.gui)
    return 0


if __name__ == "__main__":
    sys.exit(main())
