"""LingBot streaming predictions -> COLMAP text sparse model (SfM bypass).

What: converts the LingBot ``pred-*.pt`` poses/intrinsics plus the
MV-consistency TSDF cloud into a COLMAP text model (cameras.txt /
images.txt / points3D.txt), so the gsplat training stage runs WITHOUT any
SfM. LingBot extrinsics are WORLD->CAM in the same convention as COLMAP,
so qvec/tvec are taken directly from them.

Pipeline stage: LingBot predictions + fused TSDF cloud -> THIS
(COLMAP text export) -> gsplat training.

Conventions (verified in this project — do not "improve"):
  - One shared PINHOLE camera: median of per-frame intrinsics at prediction
    resolution, scaled up to the full frame resolution.
  - extrinsic E (3,4) is WORLD->CAM: qvec/tvec written directly, no inversion.
  - Init points are subsampled from the TSDF cloud with --max-points
    (default 350000): the dense init IS the InstantSplat fast-path —
    growth-free training converged in 10k iters vs 80k, with 38x fewer
    floaters.
  - Prediction index i pairs with sorted jpg i (stride 1 always).

Interpreter contract (documented, not auto-switched): runs under the main
venv /home/perelman/Basic_RL/.venv/bin/python (torch/open3d/pycolmap/numpy/PIL).

Usage:
  python -m video2sim.to_colmap [--workdir W] [--pred PRED.pt]
                                [--cloud CLOUD.ply] [--frames_dir DIR]
                                [--max-points 350000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
from PIL import Image

TAG = "[to_colmap]"


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


def rot_to_qvec(R: np.ndarray) -> np.ndarray:
    """COLMAP qvec = (qw, qx, qy, qz) of the world->cam rotation."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)


def export_colmap(
    pred: Path,
    cloud: Path,
    frames_dir: Path,
    workdir: Path,
    max_points: int = 350000,
) -> Path:
    """Write WORKDIR/sparse/0 COLMAP text model + WORKDIR/frames symlink.

    Returns the sparse model directory.
    """
    jpgs = sorted(frames_dir.glob("*.jpg"))
    d = torch.load(str(pred), map_location="cpu", weights_only=False, mmap=True)
    p = d["predictions"]
    E = p["extrinsic"].numpy()          # (N,3,4) world->cam
    K = p["intrinsic"].numpy()          # (N,3,3) at pred resolution
    N, H, W = p["depth"].shape[0], p["depth"].shape[1], p["depth"].shape[2]
    assert len(jpgs) >= N, f"{len(jpgs)} jpgs < {N} preds"

    # Shared PINHOLE camera: median intrinsics at pred res, scaled to full res.
    FW, FH = Image.open(jpgs[0]).size
    sx, sy = FW / W, FH / H
    fx = float(np.median(K[:, 0, 0])) * sx
    fy = float(np.median(K[:, 1, 1])) * sy
    cx = float(np.median(K[:, 0, 2])) * sx
    cy = float(np.median(K[:, 1, 2])) * sy
    _log(f"{N} frames, pred {W}x{H} -> full {FW}x{FH}, "
         f"shared PINHOLE fx={fx:.1f} fy={fy:.1f}")

    sp = workdir / "sparse" / "0"
    sp.mkdir(parents=True, exist_ok=True)
    fr = workdir / "frames"
    if not fr.is_symlink() and not fr.is_dir():
        fr.symlink_to(frames_dir)

    with open(sp / "cameras.txt", "w") as f:
        f.write("# Camera list: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        f.write(f"1 PINHOLE {FW} {FH} {fx} {fy} {cx} {cy}\n")

    with open(sp / "images.txt", "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n#   POINTS2D[]\n")
        for i in range(N):
            # world->cam qvec/tvec taken directly from LingBot extrinsics.
            R, t = E[i][:3, :3], E[i][:3, 3]
            q = rot_to_qvec(R)
            name = jpgs[i].name
            f.write(f"{i + 1} {q[0]} {q[1]} {q[2]} {q[3]} "
                    f"{t[0]} {t[1]} {t[2]} 1 {name}\n\n")

    # Init points from the TSDF cloud. Dense init (350k) is the InstantSplat
    # fast-path: growth-free training converged in 10k iters vs 80k, with
    # 38x fewer floaters.
    pc = o3d.io.read_point_cloud(str(cloud))
    P = np.asarray(pc.points)
    C = (np.asarray(pc.colors) * 255).astype(np.uint8) if pc.has_colors() else \
        np.full((len(P), 3), 128, np.uint8)
    if len(P) > max_points:
        sel = np.random.default_rng(0).choice(len(P), max_points, replace=False)
        P, C = P[sel], C[sel]
    with open(sp / "points3D.txt", "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for j in range(len(P)):
            f.write(f"{j + 1} {P[j, 0]} {P[j, 1]} {P[j, 2]} "
                    f"{C[j, 0]} {C[j, 1]} {C[j, 2]} 1.0\n")
    _log(f"wrote {sp}: {N} images, {len(P)} init points")
    return sp


def validate_pycolmap(sparse_dir: Path) -> Tuple[int, int, int]:
    """Read the text model back with pycolmap; return (cameras, images, points)."""
    import pycolmap
    rec = pycolmap.Reconstruction(str(sparse_dir))
    nc, ni, np3 = len(rec.cameras), len(rec.images), len(rec.points3D)
    _log(f"pycolmap read-back: {nc} cameras, {ni} images, {np3} points3D")
    return nc, ni, np3


def _default_pred(workdir: Path) -> Path:
    """Newest pred-*.pt in WORKDIR/lingbot (where the inference stage writes)."""
    preds = sorted((workdir / "lingbot").glob("pred-*.pt"),
                   key=lambda p: p.stat().st_mtime)
    if not preds:
        raise SystemExit(
            f"{TAG} ERROR: no pred-*.pt in {workdir / 'lingbot'}; pass --pred")
    return preds[-1]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m video2sim.to_colmap",
        description="LingBot predictions -> COLMAP text sparse model (no SfM)")
    ap.add_argument("--workdir", type=Path, default=Path("."),
                    help="pipeline working dir; sparse model goes to WORKDIR/sparse/0")
    ap.add_argument("--pred", type=Path, default=None,
                    help="predictions .pt (default: newest WORKDIR/lingbot/pred-*.pt)")
    ap.add_argument("--cloud", type=Path, default=None,
                    help="TSDF PLY for init points (default: WORKDIR/fuse/mvtsdf.ply)")
    ap.add_argument("--frames_dir", type=Path, default=None,
                    help="original-res jpgs, sorted order == pred order "
                         "(default: WORKDIR/frames)")
    ap.add_argument("--max-points", "--max_points", dest="max_points",
                    type=int, default=350000,
                    help="init-point subsample cap; dense init (350k) is the "
                         "InstantSplat fast-path (10k iters vs 80k, 38x fewer floaters)")
    args = ap.parse_args(argv)

    pred = args.pred if args.pred is not None else _default_pred(args.workdir)
    cloud = args.cloud if args.cloud is not None else args.workdir / "fuse" / "mvtsdf.ply"
    frames_dir = args.frames_dir if args.frames_dir is not None else args.workdir / "frames"

    sp = export_colmap(pred=pred, cloud=cloud, frames_dir=frames_dir,
                       workdir=args.workdir, max_points=args.max_points)
    validate_pycolmap(sp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
