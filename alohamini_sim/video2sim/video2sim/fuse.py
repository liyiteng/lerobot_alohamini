"""Multi-view-consistency filtered TSDF fusion of LingBot-Map predictions.

What: fuses the per-frame depth/pose predictions (``pred-*.pt`` from the
LingBot inference stage) into one clean point cloud + triangle mesh — the
geometry that becomes the Isaac Sim physics collider.

Pipeline stage: frames + LingBot predictions -> THIS (MV-filtered TSDF
fusion) -> collider mesh / point cloud for scene assembly.

How:
  1. Per-frame MV-consistency mask: unproject depth_i to world, reproject into
     temporal neighbors (offsets +/-4, 8, 12), keep pixels whose reprojected
     depth agrees with the neighbor's depth map within 3% relative depth in
     >= --min_consistent neighbors (and conf > --conf).
  2. Integrate all frames into a ScalableTSDFVolume (voxel 0.004, trunc 0.02)
     with (optionally) pose-graph-corrected extrinsics
     (world_new = T_i @ world_old  =>  new world->cam = E4_i @ inv(T_i)).
  3. Export point cloud + triangle mesh PLY.

Conventions (verified in this project — do not "improve"):
  extrinsic E (3,4) is WORLD->CAM: Pc = R @ Pw + t.
  Unprojection: Pc = K^-1 [u,v,1] * z ; Pw = (Pc - t) @ R.

The kept-pixel % report doubles as a pose-consistency health metric:
same-track poses hold ~88%, cross-track poses collapse to ~58%.

Interpreter contract (documented, not auto-switched): runs under the main
venv /home/perelman/Basic_RL/.venv/bin/python (torch/open3d/numpy/PIL).

Usage:
  python -m video2sim.fuse [--workdir W] [--pred PRED.pt] [--poses P.npz]
                           [--frames_glob 'W/frames/f_*.jpg'] [--out OUT.ply]
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
import torch
from PIL import Image

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

TAG = "[fuse]"

RAM_GUARD_GB = 45.0
REL_DEPTH_TOL = 0.03  # |z_proj - z_nb| / z_nb consistency threshold


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


def check_ram() -> None:
    if not _PSUTIL:
        return
    rss_gb = psutil.Process().memory_info().rss / 2**30
    if rss_gb > RAM_GUARD_GB:
        _log(f"WARNING: RSS {rss_gb:.1f} GB exceeds {RAM_GUARD_GB} GB guard — continuing")


def load_predictions(path: Path, limit: Optional[int]) -> Tuple[dict, int]:
    """Load the LingBot ``pred-*.pt`` (mmapped) and return (predictions, n_used)."""
    d = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    pr = d["predictions"]
    n_total = pr["depth"].shape[0]
    n = min(n_total, limit) if limit else n_total
    _log(f"pred: {n_total} frames total, using {n}")
    return pr, n


def corrected_extrinsics(pr: dict, n: int, poses_path: Optional[Path]) -> np.ndarray:
    """Return (n,4,4) corrected WORLD->CAM matrices."""
    E4 = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    E4[:, :3, :] = pr["extrinsic"][:n].numpy().astype(np.float64)
    if poses_path is None:
        _log("no --poses: using raw extrinsics")
        return E4
    frame_T = np.load(str(poses_path))["frame_T"]
    if frame_T.shape[0] < n:
        raise ValueError(f"--poses frame_T has {frame_T.shape[0]} frames < {n} needed")
    # world_new = T @ world_old  =>  new world->cam = E4 @ inv(T)
    E4c = np.einsum("nij,njk->nik", E4, np.linalg.inv(frame_T[:n]))
    _log(f"applied pose corrections from {poses_path}")
    return E4c


class DepthCache:
    """Lazy per-frame float32 depth maps read from the mmapped tensor."""

    def __init__(self, depth_tensor: torch.Tensor) -> None:
        self._t = depth_tensor
        self._cache: Dict[int, np.ndarray] = {}

    def get(self, i: int) -> np.ndarray:
        if i not in self._cache:
            self._cache[i] = self._t[i, :, :, 0].numpy().astype(np.float32)
        return self._cache[i]


def mv_consistency_mask(
    i: int,
    depth_i: np.ndarray,
    K: np.ndarray,
    E4: np.ndarray,
    neighbors: Sequence[int],
    depth_cache: DepthCache,
    uv1: np.ndarray,
) -> np.ndarray:
    """Count, per pixel of frame i, in how many neighbors its depth is consistent.

    uv1: (H*W, 3) homogeneous pixel coords [u, v, 1].
    Returns int16 (H,W) consistency counts (only for pixels with depth > 0).
    """
    h, w = depth_i.shape
    z = depth_i.reshape(-1)
    valid_i = z > 0

    Ki_inv = np.linalg.inv(K[i])
    pc = (uv1 @ Ki_inv.T) * z[:, None]                     # cam_i points
    Ri, ti = E4[i, :3, :3], E4[i, :3, 3]
    pw = (pc - ti) @ Ri                                    # world points (= R^T (Pc - t))

    count = np.zeros(h * w, dtype=np.int16)
    for j in neighbors:
        Rj, tj = E4[j, :3, :3], E4[j, :3, 3]
        pcj = pw @ Rj.T + tj
        zj = pcj[:, 2]
        front = valid_i & (zj > 1e-6)
        fx, fy, cx, cy = K[j][0, 0], K[j][1, 1], K[j][0, 2], K[j][1, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.rint(fx * pcj[:, 0] / zj + cx).astype(np.int32)
            v = np.rint(fy * pcj[:, 1] / zj + cy).astype(np.int32)
        inb = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = np.flatnonzero(inb)
        dj = depth_cache.get(j)[v[idx], u[idx]]
        ok = (dj > 0) & (np.abs(zj[idx] - dj) / np.maximum(dj, 1e-9) < REL_DEPTH_TOL)
        count[idx[ok]] += 1
    return count.reshape(h, w)


def fuse(
    pred: Path,
    frames_glob: str,
    out: Path,
    poses: Optional[Path] = None,
    voxel: float = 0.004,
    sdf_trunc: float = 0.02,
    conf_thresh: float = 2.3,
    frame_stride: int = 1,
    limit: Optional[int] = None,
    nb_offsets: Sequence[int] = (4, 8, 12),
    min_consistent: int = 2,
) -> Tuple[Path, Path]:
    """MV-filter + TSDF-integrate all prediction frames; write cloud + mesh PLY.

    Returns (cloud_ply_path, mesh_ply_path).
    """
    jpgs = sorted(glob.glob(frames_glob))
    _log(f"{len(jpgs)} color frames, frame_stride={frame_stride}")

    pr, n = load_predictions(pred, limit)
    h, w = pr["depth"].shape[1:3]
    K = pr["intrinsic"][:n].numpy().astype(np.float64)
    conf = pr["depth_conf"]
    E4 = corrected_extrinsics(pr, n, poses)
    depth_cache = DepthCache(pr["depth"])

    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    uv1 = np.stack([uu.ravel(), vv.ravel(), np.ones(h * w)], axis=1)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    px_total = px_kept = 0
    skipped_color = 0
    t_mv = t_int = 0.0
    t0 = time.time()

    for i in range(n):
        ci = i * frame_stride
        if ci >= len(jpgs):
            skipped_color += 1
            continue

        ts = time.time()
        depth_i = depth_cache.get(i)
        neighbors = [i + s * o for o in nb_offsets for s in (-1, 1) if 0 <= i + s * o < n]
        counts = mv_consistency_mask(i, depth_i, K, E4, neighbors, depth_cache, uv1)
        need = min(min_consistent, len(neighbors))
        conf_i = conf[i].numpy()
        keep = (counts >= need) & (conf_i > conf_thresh) & (depth_i > 0)
        px_total += int((depth_i > 0).sum())
        px_kept += int(keep.sum())
        depth_f = np.where(keep, depth_i, 0.0).astype(np.float32)
        t_mv += time.time() - ts

        ts = time.time()
        color = np.asarray(
            Image.open(jpgs[ci]).convert("RGB").resize((w, h), Image.BILINEAR))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(depth_f),
            depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(
            w, h, K[i][0, 0], K[i][1, 1], K[i][0, 2], K[i][1, 2])
        volume.integrate(rgbd, intr, E4[i])
        t_int += time.time() - ts

        if i % 20 == 0:
            check_ram()
        if i % 50 == 0 or i == n - 1:
            _log(f"frame {i + 1}/{n}  kept so far "
                 f"{100.0 * px_kept / max(px_total, 1):.1f}%")

    if skipped_color:
        _log(f"WARNING: {skipped_color} frames skipped (color index out of range)")
    # Kept % is a pose-consistency health metric: same-track ~88%, cross-track ~58%.
    kept_pct = 100.0 * px_kept / max(px_total, 1)
    _log(f"MV filter kept {px_kept}/{px_total} pixels ({kept_pct:.1f}%)")
    _log(f"timing: mv-filter {t_mv:.1f}s  integrate {t_int:.1f}s")

    ts = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    pcd = volume.extract_point_cloud()
    o3d.io.write_point_cloud(str(out), pcd)
    _log(f"point cloud: {len(pcd.points)} points -> {out}")
    mesh = volume.extract_triangle_mesh()
    mesh_out = out.with_name(out.stem + "_mesh.ply")
    o3d.io.write_triangle_mesh(str(mesh_out), mesh)
    _log(f"mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles -> {mesh_out}")
    _log(f"extract {time.time() - ts:.1f}s  total {time.time() - t0:.1f}s")
    return out, mesh_out


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
        prog="python -m video2sim.fuse",
        description="MV-consistency filtered TSDF fusion of LingBot-Map predictions")
    ap.add_argument("--workdir", type=Path, default=Path("."),
                    help="pipeline working dir; default root for --pred/--frames_glob/--out")
    ap.add_argument("--pred", type=Path, default=None,
                    help="predictions .pt (default: newest WORKDIR/lingbot/pred-*.pt)")
    ap.add_argument("--poses", type=Path, default=None,
                    help="npz with frame_T (N,4,4) pose-graph corrections, "
                         "world_new = T @ world_old")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--sdf_trunc", type=float, default=0.02)
    ap.add_argument("--conf", type=float, default=2.3)
    ap.add_argument("--frame_stride", type=int, default=1,
                    help="color/frame index = pred_index * frame_stride into sorted jpg list")
    ap.add_argument("--limit", type=int, default=None, help="use only first N pred frames")
    ap.add_argument("--nb", default="4,8,12", help="comma-separated neighbor offsets (+/-)")
    ap.add_argument("--min_consistent", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None,
                    help="cloud PLY output; mesh goes next to it as *_mesh.ply "
                         "(default: WORKDIR/fuse/mvtsdf.ply)")
    ap.add_argument("--frames_glob", default=None,
                    help="glob for the RGB jpgs matching the prediction "
                         "(default: WORKDIR/frames/f_*.jpg)")
    args = ap.parse_args(argv)

    pred = args.pred if args.pred is not None else _default_pred(args.workdir)
    out = args.out if args.out is not None else args.workdir / "fuse" / "mvtsdf.ply"
    frames_glob = (args.frames_glob if args.frames_glob is not None
                   else str(args.workdir / "frames" / "f_*.jpg"))
    offsets = [int(x) for x in args.nb.split(",") if x.strip()]

    fuse(pred=pred, frames_glob=frames_glob, out=out, poses=args.poses,
         voxel=args.voxel, sdf_trunc=args.sdf_trunc, conf_thresh=args.conf,
         frame_stride=args.frame_stride, limit=args.limit,
         nb_offsets=offsets, min_consistent=args.min_consistent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
