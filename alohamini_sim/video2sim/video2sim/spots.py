"""video2sim stage — open-spot finder: robot-taskable floor spots + capture cam.

What: from the gravity-aligned metric point cloud (refine output), find open
floor spots with maximum clearance where the robot task can be staged, and
pick a real capture-trajectory camera pose that films the task area.

Method (validated on room3):
  - obstacle occupancy on a 5 cm grid from points 0.15 < z < 1.6 m
    (= stuff between ankle and head height);
  - floor mask |z| < 0.06 m, dilated 2 cells;
  - euclidean distance transform of the free grid = clearance to the nearest
    obstacle, only where real floor exists;
  - CRITICAL: candidates are restricted to within 1.2 m of the WALKED CAMERA
    TRACK (track cells dilated by track_radius/res). The first pick at
    (5.17, 2.62) was the largest clearance GLOBALLY, which turned out to be
    recon garbage OUTSIDE the walls — reconstruction noise past the room
    boundary looks like wide-open floor to a naive distance transform.
  - the top spot defines the task origin: the Isaac scene stage shifts the
    room by ROOM_SHIFT = -spot so the open spot lands at (0, 0).

pick_camera_pose(): choose an ACTUAL capture-trajectory pose facing a target.
Splat gaussians occlude arbitrary viewpoints, but a spot the videographer
stood at with (approximately) this exact view direction is guaranteed
fog-free (v4 postmortem: a synthetic camera 2.4 m out sat inside splat
furniture and filmed only blur). Candidates must sit 1.3-3.0 m from the
target at 1.2-1.8 m height; scored by view-angle offset plus a distance
penalty pulling toward the ideal 2.0 m.

Pipeline position: after refine (aligned cloud + align.npz), before the Isaac
scene assembly (which consumes the spot as ROOM_SHIFT and the camera pose for
rep.create.camera).

Interpreter contract (do NOT auto-switch): runs under the main venv
/home/perelman/Basic_RL/.venv/bin/python (torch + open3d + scipy + numpy).

Usage:
  python -m video2sim.spots [--workdir W] [--cloud C.ply] [--pred pred-*.pt]
      [--align align.npz] [--out-spot open_spot.npy] [--grid-res 0.05]
      [--track-radius 1.2] [--camera-target X Y Z] [--out-camera cam.npz]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt

TAG = "[spots]"

# Obstacle band: stuff between ankle and head height (m).
OBSTACLE_Z = (0.15, 1.6)
# Floor points: |z| below this (m); the floor sits at z=0 after refine.
FLOOR_Z = 0.06
# Occupancy grid resolution (m/cell).
GRID_RES = 0.05
# Candidates must lie within this many metres of the walked camera track.
TRACK_RADIUS = 1.2
# Minimum separation between reported spots (m).
MIN_SEPARATION = 1.0

# Capture-camera windows (task frame, metres).
CAM_DIST_RANGE = (1.3, 3.0)
CAM_HEIGHT_RANGE = (1.2, 1.8)
CAM_IDEAL_DIST = 2.0
CAM_DIST_WEIGHT = 8.0  # deg-per-metre penalty for straying from ideal dist


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


def load_aligned_cams(pred_path: Path, align_path: Path) -> np.ndarray:
    """Camera-track centers (N,3) in the gravity-aligned metric frame.

    Extrinsics are world->cam; center = -R^T t, written as (-t) @ R. The
    align.npz transform (R / floorz / scale from refine) maps the LingBot
    frame into the floor-at-z=0 metric frame the cloud lives in.
    """
    d = torch.load(pred_path, map_location="cpu", weights_only=False, mmap=True)
    E = d["predictions"]["extrinsic"].numpy()
    cams = np.stack([(-E[i, :3, 3]) @ E[i, :3, :3] for i in range(len(E))])
    al = np.load(align_path)
    R, fz, sc = al["R"], float(al["floorz"]), float(al["scale"])
    camsA = (cams @ R.T) * sc
    camsA[:, 2] -= fz * sc
    return camsA


def find_open_spots(
    cloud_path: Path,
    pred_path: Path,
    align_path: Path,
    grid_res: float = GRID_RES,
    obstacle_z: Tuple[float, float] = OBSTACLE_Z,
    floor_z: float = FLOOR_Z,
    floor_dilate: int = 2,
    track_radius: float = TRACK_RADIUS,
    min_separation: float = MIN_SEPARATION,
) -> List[Tuple[np.ndarray, float]]:
    """Top-2 open floor spots [(xy, clearance_m), ...] in the aligned frame.

    Obstacle occupancy from the ankle-to-head band, distance transform for
    clearance, valid only on (dilated) real floor AND within ``track_radius``
    of the walked camera track — the global max-clearance cell was recon
    garbage OUTSIDE the walls (room3 postmortem).
    """
    pc = o3d.io.read_point_cloud(str(cloud_path))
    P = np.asarray(pc.points)
    camsA = load_aligned_cams(pred_path, align_path)
    _log(f"cam track z range: {camsA[:, 2].min().round(2)} "
         f"{camsA[:, 2].max().round(2)}")

    # obstacles = stuff between ankle and head height
    obs = P[(P[:, 2] > obstacle_z[0]) & (P[:, 2] < obstacle_z[1])]
    floor = P[np.abs(P[:, 2]) < floor_z]
    res = grid_res
    mn = P[:, :2].min(0)
    span = P[:, :2].max(0) - mn
    gw, gh = int(span[0] / res) + 1, int(span[1] / res) + 1
    occ = np.zeros((gw, gh), bool)
    flr = np.zeros((gw, gh), bool)
    near = np.zeros((gw, gh), bool)
    oi = ((obs[:, :2] - mn) / res).astype(int)
    occ[oi[:, 0], oi[:, 1]] = True
    fi = ((floor[:, :2] - mn) / res).astype(int)
    flr[fi[:, 0], fi[:, 1]] = True
    # camera track cells (cams may leave the cloud bbox -> bounds check)
    ci = ((camsA[:, :2] - mn) / res).astype(int)
    ci = ci[(ci[:, 0] >= 0) & (ci[:, 0] < gw) & (ci[:, 1] >= 0) & (ci[:, 1] < gh)]
    near[ci[:, 0], ci[:, 1]] = True
    # within track_radius of the walked path
    near = binary_dilation(near, iterations=int(track_radius / res))

    # clearance to nearest obstacle, only where real floor exists (dilated a
    # bit) AND near the walked track
    dist = distance_transform_edt(~occ) * res
    valid = binary_dilation(flr, iterations=floor_dilate) & near
    dist[~valid] = 0

    ix = np.unravel_index(np.argmax(dist), dist.shape)
    spot = mn + np.array(ix) * res
    _log(f"open spot: x={spot[0]:.2f} y={spot[1]:.2f} "
         f"clearance={dist[ix]:.2f} m")
    _log(f"dist from cam-track centroid: "
         f"{np.linalg.norm(spot - camsA[:, :2].mean(0)):.2f} m")

    # second candidate away from first
    d2 = dist.copy()
    xx, yy = np.meshgrid(np.arange(gw), np.arange(gh), indexing="ij")
    d2[np.hypot(xx - ix[0], yy - ix[1]) * res < min_separation] = 0
    ix2 = np.unravel_index(np.argmax(d2), d2.shape)
    spot2 = mn + np.array(ix2) * res
    _log(f"2nd spot:  x={spot2[0]:.2f} y={spot2[1]:.2f} "
         f"clearance={d2[ix2]:.2f} m")

    return [(spot, float(dist[ix])), (spot2, float(d2[ix2]))]


def pick_camera_pose(
    pred_path: Path,
    align_path: Path,
    target: np.ndarray,
    room_shift: np.ndarray,
    dist_range: Tuple[float, float] = CAM_DIST_RANGE,
    height_range: Tuple[float, float] = CAM_HEIGHT_RANGE,
    ideal_dist: float = CAM_IDEAL_DIST,
    dist_weight: float = CAM_DIST_WEIGHT,
) -> Dict[str, object]:
    """Best capture-trajectory camera pose facing ``target`` (task frame).

    Splat gaussians occlude arbitrary viewpoints — a synthetic camera 2.4 m
    out sat INSIDE splat furniture and filmed only blur (v4 postmortem) — so
    only real capture poses are candidates. ``room_shift`` is the xy shift
    that maps the aligned frame into the task frame (= -open_spot).

    Returns {"frame", "eye", "view_dir", "angle_off_deg", "dist"}; ``eye`` is
    the camera position in the task frame, ready for rep.create.camera
    (position=eye, look_at=target).
    """
    d = torch.load(pred_path, map_location="cpu", weights_only=False, mmap=True)
    E = d["predictions"]["extrinsic"].numpy()
    al = np.load(align_path)
    R, fz, sc = al["R"], float(al["floorz"]), float(al["scale"])
    target = np.asarray(target, float)
    shift = np.asarray(room_shift, float).reshape(-1)

    best: Optional[Tuple[float, int, np.ndarray, float, float, np.ndarray]] = None
    for i in range(len(E)):
        Rc, t = E[i, :3, :3], E[i, :3, 3]
        C = (-t) @ Rc                      # cam center, lingbot frame
        Ca = (C @ R.T) * sc
        Ca[2] -= fz * sc
        Ca[:2] += shift[:2]
        fwd = Rc[2, :]                     # cam +z (view dir), row-vector form
        fwda = fwd @ R.T
        fwda = fwda / np.linalg.norm(fwda)
        to_t = target - Ca
        dist = float(np.linalg.norm(to_t))
        if not (dist_range[0] < dist < dist_range[1]
                and height_range[0] < Ca[2] < height_range[1]):
            continue
        ang = float(np.degrees(np.arccos(np.clip(np.dot(fwda, to_t / dist),
                                                 -1, 1))))
        score = ang + abs(dist - ideal_dist) * dist_weight
        if best is None or score < best[0]:
            best = (score, i, Ca.copy(), ang, dist, fwda.copy())
    if best is None:
        raise RuntimeError(
            f"{TAG} no capture pose within dist {dist_range} / "
            f"height {height_range} of target {target.round(2).tolist()}"
        )
    _, i, Ca, ang, dist, fwda = best
    _log(f"best frame {i}: cam=({Ca[0]:.2f},{Ca[1]:.2f},{Ca[2]:.2f}) "
         f"view-angle-off={ang:.1f}deg dist={dist:.2f}m")
    return {"frame": i, "eye": Ca, "view_dir": fwda,
            "angle_off_deg": ang, "dist": dist}


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
        prog="python -m video2sim.spots",
        description="video2sim: open floor spots + capture camera pose.",
    )
    ap.add_argument("--workdir", type=Path, default=Path("."),
                    help="pipeline working dir; default root for other paths")
    ap.add_argument("--cloud", type=Path, default=None,
                    help="aligned cloud (default: WORKDIR/final_cloud.ply)")
    ap.add_argument("--pred", type=Path, default=None,
                    help="LingBot pred-*.pt for the camera track "
                         "(default: newest in WORKDIR/lingbot)")
    ap.add_argument("--align", type=Path, default=None,
                    help="refine align npz (default: WORKDIR/final_align.npz)")
    ap.add_argument("--out-spot", type=Path, default=None,
                    help="write top spot xy here (default: "
                         "WORKDIR/open_spot.npy)")
    ap.add_argument("--grid-res", type=float, default=GRID_RES,
                    help="occupancy grid resolution in m (default: 0.05)")
    ap.add_argument("--obstacle-z", type=float, nargs=2, default=OBSTACLE_Z,
                    metavar=("MIN", "MAX"),
                    help="obstacle band, ankle-to-head (default: 0.15 1.6)")
    ap.add_argument("--floor-z", type=float, default=FLOOR_Z,
                    help="|z| threshold for floor points (default: 0.06)")
    ap.add_argument("--floor-dilate", type=int, default=2,
                    help="floor mask dilation iterations (default: 2)")
    ap.add_argument("--track-radius", type=float, default=TRACK_RADIUS,
                    help="max distance from the walked camera track in m "
                         "(default: 1.2; the UNRESTRICTED max was recon "
                         "garbage outside the walls)")
    ap.add_argument("--min-separation", type=float, default=MIN_SEPARATION,
                    help="min distance between reported spots (default: 1.0)")
    ap.add_argument("--camera-target", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="task-frame point the capture camera must face "
                         "(e.g. 0.0 -0.3 0.75 = desk/cube region); when "
                         "given, also picks a capture camera pose")
    ap.add_argument("--room-shift", type=float, nargs=2, default=None,
                    metavar=("X", "Y"),
                    help="aligned->task frame xy shift for the camera pick "
                         "(default: -top_spot)")
    ap.add_argument("--out-camera", type=Path, default=None,
                    help="optional npz to save the picked camera pose "
                         "(keys: frame, eye, view_dir, angle_off_deg, dist)")
    args = ap.parse_args(argv)

    cloud = args.cloud if args.cloud is not None else args.workdir / "final_cloud.ply"
    align = args.align if args.align is not None else args.workdir / "final_align.npz"
    pred = args.pred if args.pred is not None else _default_pred(args.workdir)
    out_spot = (args.out_spot if args.out_spot is not None
                else args.workdir / "open_spot.npy")

    spots = find_open_spots(
        cloud, pred, align,
        grid_res=args.grid_res,
        obstacle_z=(args.obstacle_z[0], args.obstacle_z[1]),
        floor_z=args.floor_z,
        floor_dilate=args.floor_dilate,
        track_radius=args.track_radius,
        min_separation=args.min_separation,
    )
    out_spot.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_spot, spots[0][0])
    _log(f"wrote top spot -> {out_spot}")

    if args.camera_target is not None:
        # task origin = the open spot: the Isaac stage shifts the room by
        # ROOM_SHIFT = -spot, so cameras get the same shift here
        shift = (np.asarray(args.room_shift, float)
                 if args.room_shift is not None else -spots[0][0])
        pose = pick_camera_pose(pred, align,
                                np.asarray(args.camera_target, float), shift)
        if args.out_camera is not None:
            args.out_camera.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.out_camera, **{k: np.asarray(v)
                                         for k, v in pose.items()})
            _log(f"wrote camera pose -> {args.out_camera}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
