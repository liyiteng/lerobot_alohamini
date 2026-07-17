"""video2sim stage — mesh refine: island filter + gravity/metric alignment.

What: "Layers 2+3" of collider honesty, run on the fused MV-TSDF mesh/cloud:
  layer 2 (mesh hygiene)   — drop small disconnected triangle islands (floaters);
  layer 3 (physics honesty) — gravity-align from a RANSAC floor plane, put the
    floor at z=0, and recover metric scale from the floor->ceiling anchor
    (default 2.4 m), falling back to phone hand-height (1.35 m) when no
    ceiling plane is visible.

Why the camera-track gates on the floor RANSAC: the biggest plane is often a
WALL, not the floor. Walls also have all cameras on one side, so the sign vote
alone is not enough — the spread gate (IQR/median of camera-to-plane distances
< 0.45) is what rejects them: a hand-held phone stays at near-constant height
above the FLOOR, while its distance to a wall varies widely along the walk.

The 4x4 world transform (R / floorz / scale / T) is saved alongside the
geometry so the trained splat — which lives in the SAME LingBot frame — can be
dropped into the identical Isaac coordinates later.

Pipeline position: after LingBot inference + MV-TSDF fusion, before NuRec
export / Isaac scene assembly.

Interpreter contract (do NOT auto-switch): runs under the main venv
/home/perelman/Basic_RL/.venv/bin/python (torch + open3d + numpy).

Usage:
  python -m video2sim.refine [--workdir W] [--mesh W/fuse/mvtsdf_mesh.ply]
      [--cloud W/fuse/mvtsdf.ply] [--pred pred-*.pt] [--out-prefix PREFIX]
      [--ceiling-height 2.4] [--min-island-tris 2000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch

TAG = "[refine]"

# Fallback anchor when no ceiling plane is found: hand-held phone height (m).
HAND_HEIGHT_M = 1.35

# Floor-plane camera-track gates (see module docstring: spread rejects walls).
GATE_ABOVE = 0.9    # fraction of cameras strictly above the plane
GATE_MED = 0.2      # median camera height above the plane (scene units)
GATE_SPREAD = 0.45  # IQR/median of camera heights


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


def load_camera_centers(pred: Path) -> np.ndarray:
    """World-space camera centers (N,3) from a LingBot pred-*.pt track.

    Extrinsics are world->cam; center = -R^T t, written as (-t) @ R.
    """
    d = torch.load(pred, map_location="cpu", weights_only=False, mmap=True)
    E = d["predictions"]["extrinsic"].numpy()
    return np.stack([(-E[i, :3, 3]) @ E[i, :3, :3] for i in range(len(E))])


def filter_islands(
    mesh: o3d.geometry.TriangleMesh, min_tris: int = 2000
) -> o3d.geometry.TriangleMesh:
    """Layer 2: drop disconnected triangle islands smaller than ``min_tris``.

    Mutates ``mesh`` in place and returns it.
    """
    tc, counts, _ = mesh.cluster_connected_triangles()
    tc = np.asarray(tc)
    counts = np.asarray(counts)
    keep = counts[tc] >= min_tris  # ~2k-triangle islands and up survive
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    _log(
        f"after island filter: {len(mesh.vertices)} verts, "
        f"{len(mesh.triangles)} tris ({counts.size} clusters -> "
        f"{(counts >= min_tris).sum()} kept)"
    )
    return mesh


def find_floor(P: np.ndarray, cams: np.ndarray) -> Tuple[np.ndarray, float]:
    """RANSAC the floor plane out of vertex positions ``P`` (N,3).

    Peels up to 8 planes; each candidate must pass the camera-track gates
    (above > 0.9, med > 0.2, spread = IQR/med < 0.45 — the spread gate rejects
    WALLS, which also have all cameras on one side). Returns (up, dd): the unit
    up-normal and plane offset with cameras satisfying ``c @ up + dd > 0``.
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    work = pc.uniform_down_sample(max(1, len(P) // 150000))
    rest, best = work, None
    for k in range(8):
        if len(rest.points) < 4000:
            break
        model, inl = rest.segment_plane(0.02, 3, 1500)
        nrm = np.array(model[:3]) / np.linalg.norm(model[:3])
        dd = model[3]
        sd = cams @ nrm + dd
        # Flip so the majority of cameras sit on the positive side.
        if np.mean(sd < 0) > 0.5:
            nrm, dd, sd = -nrm, -dd, -sd
        med = np.median(sd)
        iqr = np.subtract(*np.percentile(sd, [75, 25]))
        spread = iqr / max(med, 1e-6)
        _log(
            f"  plane {k}: inl={len(inl)} n={nrm.round(2)} med={med:.2f} "
            f"above={np.mean(sd > 0):.2f} spread={spread:.2f}"
        )
        if np.mean(sd > 0) > GATE_ABOVE and med > GATE_MED and spread < GATE_SPREAD:
            if best is None or len(inl) > best[0]:
                best = (len(inl), nrm, dd)
        rest = rest.select_by_index(inl, invert=True)
    if best is None:
        raise RuntimeError(f"{TAG} no floor plane passed the camera-track gates")
    _, up, dd = best
    return up, dd


def gravity_rotation(up: np.ndarray) -> np.ndarray:
    """Rotation that maps the floor normal ``up`` to +z."""
    x = np.cross([0, 1.0, 0], up)
    if np.linalg.norm(x) < 0.1:
        x = np.cross([1.0, 0, 0], up)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(up, x), up])


def compute_scale(
    P: np.ndarray,
    R: np.ndarray,
    floorz: float,
    cams: np.ndarray,
    up: np.ndarray,
    dd: float,
    ceiling_height: float = 2.4,
) -> float:
    """Metric scale from the floor->ceiling anchor.

    RANSACs a plane in the upper 40% of the gravity-aligned points; if it is
    horizontal (|n_z| > 0.8) its median height anchors ``ceiling_height``
    metres. Otherwise fall back to the phone hand-height anchor (1.35 m over
    the median camera height above the floor).
    """
    PA = P @ R.T
    PA[:, 2] -= floorz
    hi = PA[PA[:, 2] > np.percentile(PA[:, 2], 60)]
    hp = o3d.geometry.PointCloud()
    hp.points = o3d.utility.Vector3dVector(hi)
    hp = hp.uniform_down_sample(max(1, len(hi) // 120000))
    cm, cin = hp.segment_plane(0.02, 3, 1500)
    cn = np.array(cm[:3]) / np.linalg.norm(cm[:3])
    if abs(cn[2]) > 0.8:
        h_units = float(np.median(np.asarray(hp.points)[cin, 2]))
        scale = ceiling_height / h_units
        _log(f"ceiling at {h_units:.3f}u -> scale {scale:.3f} "
             f"({ceiling_height} m anchor)")
    else:
        camz = float(np.median(cams @ up + dd))
        scale = HAND_HEIGHT_M / camz
        _log(f"no ceiling plane (n={cn.round(2)}); hand-height fallback "
             f"scale {scale:.3f}")
    return scale


def refine(
    mesh_path: Path,
    cloud_path: Path,
    pred_path: Path,
    out_prefix: Path,
    ceiling_height: float = 2.4,
    min_island_tris: int = 2000,
) -> Dict[str, Path]:
    """Run island filter + gravity/metric alignment; write outputs.

    Writes ``<out_prefix>_mesh.ply``, ``<out_prefix>_cloud.ply`` and
    ``<out_prefix>_align.npz`` (keys: R, floorz, scale, T). Returns the
    written paths.
    """
    cams = load_camera_centers(pred_path)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    _log(f"mesh in: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # --- layer 2: drop small disconnected islands (floaters) ---
    filter_islands(mesh, min_island_tris)

    # --- layer 3: gravity + metric ---
    P = np.asarray(mesh.vertices)
    up, dd = find_floor(P, cams)
    R = gravity_rotation(up)
    floorz = -dd
    scale = compute_scale(P, R, floorz, cams, up, dd, ceiling_height)

    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = np.array([0, 0, -floorz * scale])

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    align_path = Path(f"{out_prefix}_align.npz")
    mesh_out = Path(f"{out_prefix}_mesh.ply")
    cloud_out = Path(f"{out_prefix}_cloud.ply")
    # Save the transform so the trained splat (same LingBot frame) can be
    # dropped into the identical Isaac coordinates later.
    np.savez(align_path, R=R, floorz=floorz, scale=scale, T=T)

    mesh.transform(T)
    o3d.io.write_triangle_mesh(str(mesh_out), mesh)
    cl = o3d.io.read_point_cloud(str(cloud_path))
    cl.transform(T)
    o3d.io.write_point_cloud(str(cloud_out), cl)
    ext = np.asarray(mesh.vertices).max(0) - np.asarray(mesh.vertices).min(0)
    _log(f"final: extent {ext.round(2)} m, floor z=0 | wrote "
         f"{mesh_out.name} / {cloud_out.name} / {align_path.name}")
    return {"mesh": mesh_out, "cloud": cloud_out, "align": align_path}


def _default_fused(workdir: Path, name: str) -> Path:
    """Fused geometry where fuse writes it (WORKDIR/fuse/<name>, or WORKDIR root).

    Same two-location probe as export.py. Fail loudly if neither exists:
    open3d only WARNS on a missing file and returns an empty mesh, which then
    dies much later with a confusing segment_plane RANSAC error.
    """
    for cand in (workdir / "fuse" / name, workdir / name):
        if cand.exists():
            return cand
    raise SystemExit(
        f"{TAG} ERROR: no {name} at {workdir / 'fuse' / name} or "
        f"{workdir / name}; run video2sim.fuse first, or pass --mesh/--cloud"
    )


def _default_pred(workdir: Path) -> Path:
    """Newest pred-*.pt under <workdir>/lingbot (matches lingbot_infer output)."""
    preds = sorted((workdir / "lingbot").glob("pred-*.pt"),
                   key=lambda p: p.stat().st_mtime)
    if not preds:
        raise SystemExit(
            f"{TAG} ERROR: no pred-*.pt in {workdir / 'lingbot'}; pass --pred"
        )
    return preds[-1]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m video2sim.refine",
        description="video2sim: mesh island filter + gravity/metric alignment.",
    )
    ap.add_argument("--workdir", type=Path, default=Path("."),
                    help="pipeline working dir; default root for other paths")
    ap.add_argument("--mesh", type=Path, default=None,
                    help="fused mesh (default: WORKDIR/fuse/mvtsdf_mesh.ply, "
                         "else WORKDIR/mvtsdf_mesh.ply)")
    ap.add_argument("--cloud", type=Path, default=None,
                    help="fused point cloud (default: WORKDIR/fuse/mvtsdf.ply, "
                         "else WORKDIR/mvtsdf.ply)")
    ap.add_argument("--pred", type=Path, default=None,
                    help="LingBot pred-*.pt for the camera track "
                         "(default: newest in WORKDIR/lingbot)")
    ap.add_argument("--out-prefix", type=Path, default=None,
                    help="output prefix; writes <prefix>_mesh.ply, "
                         "<prefix>_cloud.ply, <prefix>_align.npz "
                         "(default: WORKDIR/final)")
    ap.add_argument("--ceiling-height", type=float, default=2.4,
                    help="floor->ceiling metric anchor in metres (default: 2.4)")
    ap.add_argument("--min-island-tris", type=int, default=2000,
                    help="drop mesh islands smaller than this (default: 2000)")
    args = ap.parse_args(argv)

    mesh = (args.mesh if args.mesh is not None
            else _default_fused(args.workdir, "mvtsdf_mesh.ply"))
    cloud = (args.cloud if args.cloud is not None
             else _default_fused(args.workdir, "mvtsdf.ply"))
    # Guard explicit paths too: open3d returns an EMPTY mesh (warning only)
    # for a missing file, so fail here instead of deep in floor RANSAC.
    for p, flag in ((mesh, "--mesh"), (cloud, "--cloud")):
        if not p.exists():
            raise SystemExit(f"{TAG} ERROR: {flag} file not found: {p}")
    pred = args.pred if args.pred is not None else _default_pred(args.workdir)
    out_prefix = (args.out_prefix if args.out_prefix is not None
                  else args.workdir / "final")

    refine(mesh, cloud, pred, out_prefix,
           ceiling_height=args.ceiling_height,
           min_island_tris=args.min_island_tris)
    return 0


if __name__ == "__main__":
    sys.exit(main())
