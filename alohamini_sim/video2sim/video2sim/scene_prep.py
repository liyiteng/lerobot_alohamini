"""video2sim stage — collider prep: refined mesh PLY -> decimated V/F npz.

What: decimates the REFINED (metric/aligned, floor at z=0) TSDF mesh to a
physics-friendly triangle budget and saves compact ``V`` (float32, N x 3) /
``F`` (int32, M x 3) arrays to an npz. The npz is the hand-off format to the
Isaac venv: :mod:`video2sim.scene` re-creates the mesh from raw arrays so the
Isaac side never needs open3d.

Why the split: open3d lives in the MAIN venv while USD assembly runs under the
Isaac venv — the npz is the only thing that crosses that boundary. The input
mesh MUST already be metric/aligned (output of ``video2sim.refine``): the
collider is placed with an IDENTITY transform in the scene, only the splat
gets the align-npz transform.

Pipeline position: after ``video2sim.refine`` (final_mesh.ply), before
``video2sim.scene`` (USD assembly).

Interpreter contract (do NOT auto-switch): runs under the main venv
/home/perelman/Basic_RL/.venv/bin/python (open3d + numpy).

Usage:
  python -m video2sim.scene_prep [--workdir W] [--mesh W/final_mesh.ply]
      [--out W/collider.npz] [--tris 300000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d

TAG = "[scene-prep]"

# 300k tris is the validated collider budget: full detail for contact honesty
# without blowing up PhysX cooking (same budget as the CuRobo env-mesh
# obstacle recipe).
DEFAULT_TRIS = 300_000


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


def build_collider_npz(
    mesh_path: Path,
    out_npz: Path,
    target_tris: int = DEFAULT_TRIS,
) -> Path:
    """Decimate ``mesh_path`` to ``target_tris`` and save V/F arrays.

    ``mesh_path`` must be the refined, already metric/aligned mesh (floor at
    z=0) — the scene assembler places the collider without any transform.
    Writes ``out_npz`` with keys ``V`` (float32, N x 3) and ``F`` (int32,
    M x 3); returns the written path.
    """
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    n_in = len(mesh.triangles)
    if n_in == 0:
        raise SystemExit(f"{TAG} ERROR: no triangles in {mesh_path}")
    _log(f"mesh in: {len(mesh.vertices)} verts, {n_in} tris ({mesh_path})")

    if n_in > target_tris:
        mesh = mesh.simplify_quadric_decimation(target_tris)
        _log(f"decimated: {len(mesh.vertices)} verts, "
             f"{len(mesh.triangles)} tris (target {target_tris})")
    else:
        _log(f"already <= {target_tris} tris; no decimation")

    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.triangles, dtype=np.int32)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, V=V, F=F)
    _log(f"wrote {out_npz} ({len(V)} verts / {len(F)} tris)")
    return out_npz


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m video2sim.scene_prep",
        description="video2sim: refined mesh PLY -> decimated collider npz.",
    )
    ap.add_argument("--workdir", type=Path, default=Path("."),
                    help="pipeline working dir; default root for other paths")
    ap.add_argument("--mesh", type=Path, default=None,
                    help="refined metric/aligned mesh "
                         "(default: WORKDIR/final_mesh.ply)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output npz with V/F arrays "
                         "(default: WORKDIR/collider.npz)")
    ap.add_argument("--tris", type=int, default=DEFAULT_TRIS,
                    help=f"decimation triangle budget (default: {DEFAULT_TRIS})")
    args = ap.parse_args(argv)

    mesh = args.mesh if args.mesh is not None else args.workdir / "final_mesh.ply"
    out = args.out if args.out is not None else args.workdir / "collider.npz"
    build_collider_npz(mesh, out, target_tris=args.tris)
    return 0


if __name__ == "__main__":
    sys.exit(main())
