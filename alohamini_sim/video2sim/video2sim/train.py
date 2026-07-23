"""video2sim stage — 3DGS training (gsplat MCMC strategy, 8GB-bounded) + flythrough.

What: trains a Gaussian splat from posed frames (COLMAP sparse model at
``WORKDIR/sparse/0`` + ``WORKDIR/frames``, or a VGGT npz dir via ``--vggt``)
and writes ``splat.pt`` (plus ``refined_poses.npz`` when ``--pose-opt`` is on,
and ``flythrough.mp4`` when ``--video`` is passed).

Pipeline stage: frames + poses (sfm / to_colmap / vggt) -> THIS -> splat.pt
-> NuRec export (needs SH3 zero-padded PLY) -> Isaac scene.

Hard-won constraints baked into this trainer (do NOT "improve"):
  - Learning rates are keyed BY NAME: ``torch.nn.ParameterDict`` sorts its keys
    alphabetically, so a positional zip silently shuffled every lr (opacities
    got 1e-3 instead of 5e-2 — the single biggest cause of the fog splats).
  - scene_scale (camera spread) scales the means lr and the MCMC noise, and is
    CAPPED at 6.0: forward-motion / corridor scenes give a huge spread that
    over-inflates both and collapses opacity to fog.
  - MCMC strategy with relocation stopping at 80% of iters (settle phase).
    The output is an MCMC ensemble — downstream must NOT opacity-prune it
    (min_opacity pruning deletes ~79% of an MCMC ensemble).
  - rasterization ``packed=True`` ALWAYS: sparse tile-intersection buffers are
    the difference between OOM and fitting at full-res on 8GB.
  - D-SSIM 0.2 loss term: plain L1 is minimized just as well by big
    translucent blobs — the fog optimum.
  - PPISP-lite per-view exposure/white-balance (identity at novel-view time).
  - Pose optimization anchors view 0 (gauge); per-view se3 via differentiable
    ``se3_exp`` through the rasterizer's viewmat gradients, lr 1e-4.
  - Images cached on CPU as uint8 (float32 full-res would be ~12GB on a 31GB
    box); /255 happens on GPU after the H2D copy.
  - Official step ordering: optimizer step first, THEN MCMC relocation/noise.

WARNING ``--depth-reg``: depth regularization may ONLY pair with SAME-track
poses. The LingBot prediction file must come from the very track that produced
the training poses (identical world frame); pairing e.g. COLMAP poses with a
depth file from a different run silently corrupts geometry instead of killing
floaters.

GPU must be exclusive during training (an open Isaac GUI eats 3-4GB of the 8GB).

Interpreter contract (do NOT auto-switch): runs under the main venv
/home/perelman/Basic_RL/.venv/bin/python (torch + gsplat + pycolmap + scipy + PIL).

Usage:
  python -m video2sim.train [--workdir W] [--iters 10000] [--downscale 1]
      [--cap 350000] [--vggt DIR] [--pose-opt | --no-pose-opt]
      [--depth-reg PRED.pt] [--depth-w 0.1] [--scene-scale-cap 6.0]
      [--out SPLAT.pt] [--video] [--video-out MP4]
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# gsplat JIT-compiles its CUDA kernels on first import and needs ninja on
# PATH; the venv ships ninja in its bin dir but a bare interpreter call does
# not put it there. Must run BEFORE the gsplat import.
import os as _os
import sys as _sys
_os.environ["PATH"] = (str(Path(_sys.executable).parent) + _os.pathsep
                       + _os.environ.get("PATH", ""))

import pycolmap
from gsplat import rasterization, MCMCStrategy
from scipy.spatial import cKDTree

TAG = "[train]"


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


class View(NamedTuple):
    """One training view: world->cam extrinsic, native intrinsics, image path."""

    T: np.ndarray  # (4,4) world->cam
    fx: float
    fy: float
    cx: float
    cy: float
    path: Path
    w: int  # native image width
    h: int  # native image height


def load_colmap(sparse_dir: Path, frames_dir: Path) -> Tuple[List[View], np.ndarray, np.ndarray]:
    """Load views + sparse points from a COLMAP model (binary or text)."""
    rec = pycolmap.Reconstruction(str(sparse_dir))
    _log(f"model: {rec.num_reg_images()} imgs, {rec.num_points3D()} pts")
    vs: List[View] = []
    for img in rec.images.values():
        cam = rec.cameras[img.camera_id]
        T = np.eye(4)
        T[:3, :3] = img.cam_from_world().rotation.matrix()
        T[:3, 3] = img.cam_from_world().translation
        p = frames_dir / img.name
        if p.exists():
            vs.append(View(T, cam.focal_length_x, cam.focal_length_y,
                           cam.principal_point_x, cam.principal_point_y,
                           p, cam.width, cam.height))
    pts = np.array([q.xyz for q in rec.points3D.values()], dtype=np.float32)
    cols = np.array([q.color for q in rec.points3D.values()], dtype=np.float32) / 255.0
    return vs, pts, cols


def load_vggt(vggt_dir: Path) -> Tuple[List[View], np.ndarray, np.ndarray]:
    """Load views + points from a VGGT export dir (vggt.npz + frames_vggt/)."""
    vd = Path(vggt_dir)
    z = np.load(vd / "vggt.npz")
    vs: List[View] = []
    for i, n in enumerate(z["names"]):
        K = z["K"][i]
        p = vd / "frames_vggt" / str(n)
        w0, h0 = Image.open(p).size
        vs.append(View(z["extrinsics"][i], K[0, 0], K[1, 1], K[0, 2], K[1, 2], p, w0, h0))
    _log(f"VGGT: {len(vs)} views, {len(z['points'])} pts")
    return vs, z["points"].astype(np.float32), z["colors"].astype(np.float32)


def se3_exp(d: torch.Tensor) -> torch.Tensor:
    """(6,) axis-angle+trans -> (4,4) SE3, differentiable."""
    w, u = d[:3], d[3:]
    th = w.norm()
    I3 = torch.eye(3, device=d.device)
    K = torch.zeros(3, 3, device=d.device)
    K = K + torch.stack([
        torch.stack([K[0, 0], -w[2], w[1]]),
        torch.stack([w[2], K[1, 1], -w[0]]),
        torch.stack([-w[1], w[0], K[2, 2]])])
    if float(th) < 1e-8:
        R, V = I3 + K, I3
    else:
        s, c = torch.sin(th), torch.cos(th)
        R = I3 + (s / th) * K + ((1 - c) / th ** 2) * (K @ K)
        V = I3 + ((1 - c) / th ** 2) * K + ((th - s) / th ** 3) * (K @ K)
    T = torch.cat([torch.cat([R, (V @ u)[:, None]], 1),
                   torch.tensor([[0, 0, 0, 1.0]], device=d.device)], 0)
    return T


def make_ssim_window(dev: str) -> torch.Tensor:
    """11x11 gaussian window (sigma 1.5) for the D-SSIM loss term."""
    g = torch.exp(-(torch.arange(11, device=dev, dtype=torch.float32) - 5) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim(a: torch.Tensor, b: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """SSIM of a, b: HxWx3 in [0,1] (win from :func:`make_ssim_window`)."""
    x = a.permute(2, 0, 1)[:, None]
    y = b.permute(2, 0, 1)[:, None]
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx = F.conv2d(x, win, padding=5)
    my = F.conv2d(y, win, padding=5)
    sxx = F.conv2d(x * x, win, padding=5) - mx * mx
    syy = F.conv2d(y * y, win, padding=5) - my * my
    sxy = F.conv2d(x * y, win, padding=5) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / \
        ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return s.mean()


def render_splat(params: torch.nn.ParameterDict, T: torch.Tensor, K: torch.Tensor,
                 W: int, H: int, mode: str = "RGB"):
    """Rasterize the splat for one view. Returns (img, alpha, info).

    packed=True: sparse tile-intersection buffers — the difference between
    OOM and fitting at full-res on 8GB (2x died at ~260k gaussians @1080p).
    """
    return rasterization(params["means"],
        params["quats"] / params["quats"].norm(dim=-1, keepdim=True).clamp_min(1e-8),
        torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
        torch.sigmoid(params["colors"]), T[None], K[None], W, H,
        render_mode=mode, packed=True)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.empty(4); t = np.trace(R)
    if t > 0:
        s = math.sqrt(t+1)*2; q[0]=0.25*s; q[1]=(R[2,1]-R[1,2])/s; q[2]=(R[0,2]-R[2,0])/s; q[3]=(R[1,0]-R[0,1])/s
    else:
        i = np.argmax(np.diag(R)); j, k = (i+1)%3, (i+2)%3
        s = math.sqrt(R[i,i]-R[j,j]-R[k,k]+1)*2
        q[0]=(R[k,j]-R[j,k])/s; q[i+1]=0.25*s; q[j+1]=(R[j,i]+R[i,j])/s; q[k+1]=(R[k,i]+R[i,k])/s
    return q/np.linalg.norm(q)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def render_flythrough(params: torch.nn.ParameterDict, views: List[View],
                      downscale: int, fly_dir: Path, video_out: Path,
                      dev: str = "cuda") -> None:
    """Flythrough: smooth path through training poses (slerp between every 3rd view)."""
    d = downscale
    W, H = views[0].w // d, views[0].h // d
    keys = views[::3]
    fly_dir.mkdir(exist_ok=True, parents=True)
    fidx = 0
    with torch.no_grad():
        for a in range(len(keys) - 1):
            T0, T1 = keys[a].T, keys[a + 1].T
            q0, q1 = mat_to_quat(T0[:3, :3]), mat_to_quat(T1[:3, :3])
            if np.dot(q0, q1) < 0:
                q1 = -q1
            for s in np.linspace(0, 1, 8, endpoint=False):
                q = q0 * (1 - s) + q1 * s; q /= np.linalg.norm(q)
                T = np.eye(4); T[:3, :3] = quat_to_mat(q); T[:3, 3] = T0[:3, 3] * (1 - s) + T1[:3, 3] * s
                Tt = torch.tensor(T, device=dev, dtype=torch.float32)
                fx, fy, cx, cy = views[0].fx, views[0].fy, views[0].cx, views[0].cy
                K = torch.tensor([[fx/d, 0, cx/d], [0, fy/d, cy/d], [0, 0, 1]],
                                 device=dev, dtype=torch.float32)
                img, _, _ = render_splat(params, Tt, K, W, H)
                Image.fromarray((img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                                ).save(fly_dir / f"{fidx:05d}.png")
                fidx += 1
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "24",
                    "-i", str(fly_dir / "%05d.png"),
                    "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p",
                    str(video_out)], check=True)
    _log(f"{video_out.name}: {fidx} frames")


def train(workdir: Path,
          *,
          frames_dir: Optional[Path] = None,
          sparse_dir: Optional[Path] = None,
          iters: int = 10000,
          downscale: int = 1,
          cap: int = 350000,
          vggt: Optional[Path] = None,
          pose_opt: bool = True,
          depth_reg: Optional[Path] = None,
          depth_w: float = 0.1,
          scene_scale_cap: float = 6.0,
          out: Optional[Path] = None,
          video: bool = False,
          video_out: Optional[Path] = None,
          dev: str = "cuda") -> Path:
    """Train the splat; returns the path of the saved splat.pt.

    ``depth_reg`` may ONLY pair with SAME-track poses (see module docstring).
    """
    wd = Path(workdir)
    frames_dir = frames_dir if frames_dir is not None else wd / "frames"
    sparse_dir = sparse_dir if sparse_dir is not None else wd / "sparse" / "0"
    out = out if out is not None else wd / "splat.pt"
    video_out = video_out if video_out is not None else wd / "flythrough.mp4"
    d = downscale

    views, pts, cols = load_vggt(vggt) if vggt else load_colmap(sparse_dir, frames_dir)
    views.sort(key=lambda v: str(v.path.name))
    _log(f"{len(views)} views")
    W, H = views[0].w // d, views[0].h // d

    # scene scale (camera spread) — official gsplat trainers scale the means lr and
    # the MCMC noise by this; without it geometry can't move and only scales grow,
    # which is exactly the fog failure mode we hit
    cam_pos = np.array([-(v.T[:3, :3].T @ v.T[:3, 3]) for v in views])
    scene_scale = float(np.linalg.norm(cam_pos - cam_pos.mean(0), axis=1).max()) * 1.1
    # forward-motion / elongated (corridor) scenes give a huge camera-spread that
    # over-inflates the means lr + MCMC noise and collapses opacity to fog; cap it
    scene_scale = min(scene_scale, scene_scale_cap)
    _log(f"scene_scale {scene_scale:.2f}")

    N = len(pts)
    # init scales from mean NN distance heuristic
    nn = cKDTree(pts).query(pts, k=4)[0][:, 1:].mean(axis=1)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(pts, device=dev)),
        "scales": torch.nn.Parameter(torch.log(torch.tensor(
            np.tile(nn[:, None], (1, 3)), device=dev, dtype=torch.float32).clamp_min(1e-4))),
        "quats": torch.nn.Parameter(torch.cat([torch.ones(N, 1, device=dev),
                                               torch.zeros(N, 3, device=dev)], 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.5, device=dev))),
        "colors": torch.nn.Parameter(torch.logit(torch.tensor(cols, device=dev).clamp(0.01, 0.99))),
    }).to(dev)
    # lrs keyed BY NAME: ParameterDict sorts its keys alphabetically, so a
    # positional zip silently shuffled every lr (opacities got 1e-3 instead of
    # 5e-2 — the single biggest cause of the fog splats)
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "colors": 2.5e-3}
    opts = {k: torch.optim.Adam([{"params": v, "lr": lrs[k], "name": k}], eps=1e-15)
            for k, v in params.items()}
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        opts["means"], gamma=0.01 ** (1.0 / iters))
    # leave the last 20% as a settle phase with no relocation (official recipe)
    strategy = MCMCStrategy(cap_max=cap, verbose=False,
                            refine_stop_iter=int(iters * 0.8))
    state = strategy.initialize_state()
    # PPISP-lite (idea from NVIDIA PPISP, CVPR'26): phone auto-exposure/AWB varies
    # per frame and otherwise gets baked into the splat as geometric mud. Learn a
    # per-view exposure (log-scalar) + white-balance (RGB gains); identity at
    # novel-view time.
    n_views_total = len(views)
    ppisp = torch.nn.Parameter(torch.zeros(n_views_total, 4, device=dev))  # [logE, logR, logG, logB]
    ppisp_opt = torch.optim.Adam([ppisp], lr=5e-3)

    cache: Dict[int, torch.Tensor] = {}

    def get_view(i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        T, fx, fy, cx, cy, p, w0, h0 = views[i]
        # per-view size: mixed portrait/landscape sets must not be squashed into
        # views[0]'s shape, and K must scale by the ACTUAL resize ratio
        Wi, Hi = w0 // d, h0 // d
        if i not in cache:
            im = Image.open(p).convert("RGB").resize((Wi, Hi))
            # CPU cache as uint8: full-res float32 would be ~12GB on this 31GB box;
            # uint8 is 3GB and the /255 happens on GPU after the ~3MB H2D copy
            cache[i] = torch.tensor(np.asarray(im, np.uint8))
        sx, sy = Wi / w0, Hi / h0
        K = torch.tensor([[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]],
                         device=dev, dtype=torch.float32)
        return (torch.tensor(T, device=dev, dtype=torch.float32), K,
                cache[i].to(dev).float() / 255.0, Wi, Hi)

    # D-SSIM term (0.2 weight, official 3DGS/gsplat recipe): without it plain L1
    # is minimized just as well by big translucent blobs — the fog optimum
    win = make_ssim_window(dev)

    # LongSplat/InstantSplat-style JOINT POSE OPTIMIZATION (--pose-opt, default on):
    # per-view se3 delta optimized through the rasterizer's viewmat gradients.
    # Absorbs feedforward-pose drift (LingBot: 54cm rms vs BA) without SfM.
    # View 0 stays frozen (gauge anchor).
    if pose_opt:
        pose_delta = torch.nn.Parameter(torch.zeros(len(views), 6, device=dev))
        pose_delta_opt = torch.optim.Adam([pose_delta], lr=1e-4)
        _log(f"[POSEOPT] on: {len(views)} views x se3, lr 1e-4, view0 anchored")

    # DN-Splatter-style DEPTH regularization (--depth-reg; normals skipped by design):
    # LingBot per-frame dense depth shares the training world frame, so rendered
    # expected-depth can be supervised directly. Kills floaters/fog at novel views —
    # and it only moves gaussians (classic parameter semantics), so NuRec-safe.
    # SAME-TRACK ONLY: the depth file must come from the run that made the poses.
    _dcache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    if depth_reg:
        _dp = torch.load(depth_reg, map_location="cpu", weights_only=False, mmap=True)["predictions"]
        _dd, _dcf = _dp["depth"], _dp["depth_conf"]
        _name2pred = {p.name: k for k, p in enumerate(sorted(frames_dir.glob("*.jpg")))}
        _log(f"[DEPTHREG] on: {depth_reg} w={depth_w} ({len(_name2pred)} frames)")

    def get_depth(i: int, Wi: int, Hi: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # cache the SMALL native-res depth (294x518, ~0.6MB/view) and upsample on
        # GPU per use — a full-res CPU cache would be ~5GB on this 31GB box
        if i not in _dcache:
            k = _name2pred[Path(views[i].path).name]
            _dcache[i] = (_dd[k, :, :, 0].clone(), _dcf[k].clone())
        dz0, cf0 = _dcache[i]
        dz = F.interpolate(dz0[None, None].to(dev), (Hi, Wi),
                           mode="bilinear", align_corners=False)[0, 0]
        cf = F.interpolate(cf0[None, None].to(dev), (Hi, Wi),
                           mode="bilinear", align_corners=False)[0, 0]
        return dz, (cf > 2.3) & (dz > 0.05)

    for it in range(iters):
        i = np.random.randint(len(views))
        T, K, gt, Wi, Hi = get_view(i)
        if pose_opt and i != 0:
            T = se3_exp(pose_delta[i]) @ T
        img, alpha, info = render_splat(params, T, K, Wi, Hi,
                                        mode="RGB+ED" if depth_reg else "RGB")
        corr = torch.exp(ppisp[i, 0]) * torch.exp(ppisp[i, 1:4])[None, None, :]
        pred = img[0, ..., :3] * corr
        loss = 0.8 * F.l1_loss(pred, gt) \
            + 0.2 * (1.0 - ssim(pred.clamp(0, 1), gt, win)) \
            + 0.01 * torch.sigmoid(params["opacities"]).mean() \
            + 0.01 * torch.exp(params["scales"]).mean() \
            + 0.01 * ppisp.mean(dim=0).pow(2).sum()  # pin mean exposure to identity
        if depth_reg:
            dz, dm = get_depth(i, Wi, Hi)
            if dm.any():
                loss = loss + depth_w * (img[0, ..., 3] - dz).abs()[dm].mean()
        loss.backward()
        # official ordering: optimizer step first, THEN MCMC relocation/noise —
        # relocating before the step applied stale grads to teleported gaussians
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        ppisp_opt.step(); ppisp_opt.zero_grad(set_to_none=True)
        if pose_opt:
            pose_delta_opt.step(); pose_delta_opt.zero_grad(set_to_none=True)
        means_sched.step()
        strategy.step_post_backward(params, opts, state, it, info,
                                    lr=means_sched.get_last_lr()[0])
        if it % 1000 == 0:
            op_now = torch.sigmoid(params["opacities"].detach())
            extra = ""
            if pose_opt:
                dm_ = pose_delta.detach()
                extra = (f" pose|t|med={dm_[:, 3:].norm(dim=1).median()*100:.1f}cm"
                         f" max={dm_[:, 3:].norm(dim=1).max()*100:.1f}cm")
            _log(f"it {it} loss {loss.item():.4f} n={len(params['means'])} "
                 f"op>0.5={int((op_now > 0.5).sum())}{extra}")
        if it > 0 and it % 10000 == 0:
            # long-run insurance: a crash at hour 5 must not cost the whole run
            torch.save({k: v.detach().cpu() for k, v in params.items()},
                       out.parent / "splat_ckpt.pt")

    save_extra: Dict[str, torch.Tensor] = {}
    if pose_opt:
        save_extra["pose_delta"] = pose_delta.detach().cpu()
        # refined world->cam per view, for TSDF re-fusion / downstream consumers
        with torch.no_grad():
            refined = [(se3_exp(pose_delta[i]) @ torch.tensor(
                views[i].T, dtype=torch.float32, device=dev)).cpu().numpy()
                if i != 0 else views[i].T for i in range(len(views))]
        np.savez(str(wd / "refined_poses.npz"),
                 E=np.stack(refined), names=[Path(v.path).name for v in views])
        _log("[POSEOPT] refined_poses.npz saved")
    torch.save({**{k: v.detach().cpu() for k, v in params.items()}, **save_extra}, out)
    op_fin = torch.sigmoid(params["opacities"].detach())
    _log(f"final opacity: >0.2={int((op_fin > 0.2).sum())} "
         f">0.5={int((op_fin > 0.5).sum())} >0.8={int((op_fin > 0.8).sum())}")

    if video:
        render_flythrough(params, views, d, wd / "fly", video_out, dev=dev)
    else:
        _log("(no --video) done")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="gsplat MCMC 3DGS trainer (8GB-bounded) + flythrough video")
    ap.add_argument("--workdir", type=Path, default=Path("."),
                    help="pipeline working dir (sparse/0, frames; outputs land here)")
    ap.add_argument("--frames-dir", type=Path, default=None,
                    help="training frames dir (default: WORKDIR/frames)")
    ap.add_argument("--sparse-dir", type=Path, default=None,
                    help="COLMAP sparse model dir (default: WORKDIR/sparse/0)")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--cap", type=int, default=350000,
                    help="MCMC cap_max: max number of gaussians")
    ap.add_argument("--vggt", type=Path, default=None,
                    help="vggt.npz dir: use VGGT poses/points instead of COLMAP")
    ap.add_argument("--pose-opt", action=argparse.BooleanOptionalAction, default=True,
                    help="per-view se3 pose refinement, view 0 anchored, lr 1e-4; "
                         "saves refined_poses.npz (default: on)")
    ap.add_argument("--depth-reg", type=Path, default=None,
                    help="LingBot pred .pt for depth regularization. WARNING: may "
                         "ONLY pair with SAME-track poses (identical world frame)")
    ap.add_argument("--depth-w", type=float, default=0.1,
                    help="depth regularization weight")
    ap.add_argument("--scene-scale-cap", type=float, default=6.0,
                    help="cap on camera-spread scene scale (corridor fog guard)")
    ap.add_argument("--out", type=Path, default=None,
                    help="splat output path (default: WORKDIR/splat.pt)")
    ap.add_argument("--video", action="store_true",
                    help="render the slerp flythrough mp4 after training (default: off)")
    ap.add_argument("--video-out", type=Path, default=None,
                    help="flythrough mp4 path (default: WORKDIR/flythrough.mp4)")
    args = ap.parse_args(argv)

    train(args.workdir,
          frames_dir=args.frames_dir,
          sparse_dir=args.sparse_dir,
          iters=args.iters,
          downscale=args.downscale,
          cap=args.cap,
          vggt=args.vggt,
          pose_opt=args.pose_opt,
          depth_reg=args.depth_reg,
          depth_w=args.depth_w,
          scene_scale_cap=args.scene_scale_cap,
          out=args.out,
          video=args.video,
          video_out=args.video_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
