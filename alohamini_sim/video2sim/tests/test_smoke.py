"""video2sim CPU-only smoke tests.

Covers the pure-math / file-format seams of the pipeline (no GPU, no network):
  * ``train.se3_exp``  vs an independent scipy reference (Rotation + expm);
  * ``export`` 62-prop SH3 PLY writer/reader roundtrip;
  * ``to_colmap`` COLMAP text model write -> pycolmap read-back (+ pose integrity);
  * ``to_colmap.rot_to_qvec`` / ``train.mat_to_quat`` / ``train.quat_to_mat``
    roundtrips vs scipy, including trace<=0 branches;
  * ``export.prune_floaters`` free-space pruning on a synthetic cloud —
    including the MCMC contract that low-opacity gaussians SURVIVE at the
    default ``min_opacity=0.0``.

Interpreter contract (do not auto-switch): run under the MAIN venv
``/home/perelman/Basic_RL/.venv/bin/python`` (torch/gsplat/pycolmap/open3d/
scipy/PIL). Heavy third-party imports are guarded with ``pytest.importorskip``
so collection stays import-safe on machines without them.

Usage:
  cd /home/perelman/AlohaMini/video2sim
  /home/perelman/Basic_RL/.venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import pytest
except ImportError:  # pragma: no cover — pytest missing entirely
    print(
        "pytest is required to run the video2sim smoke tests. Install it into "
        "the main venv with:\n"
        "  uv pip install --python /home/perelman/Basic_RL/.venv/bin/python pytest",
        flush=True,
    )
    raise

# Make the package importable when the file is run without conftest.py
# (e.g. `python tests/test_smoke.py`).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

torch = pytest.importorskip("torch")
scipy_spatial = pytest.importorskip("scipy.spatial")
from scipy.linalg import expm  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _skew(w: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix of a 3-vector."""
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])


def _write_xyz_ply(path: Path, pts: np.ndarray) -> None:
    """Minimal binary_little_endian PLY with float x/y/z props (TSDF stand-in)."""
    pts = np.ascontiguousarray(np.asarray(pts, dtype="<f4"))
    hdr = [
        "ply", "format binary_little_endian 1.0",
        f"element vertex {len(pts)}",
        "property float x", "property float y", "property float z",
        "end_header",
    ]
    with open(path, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode())
        f.write(pts.tobytes())


def _fake_splat_pt(
    path: Path,
    means: np.ndarray,
    opacities: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Save a synthetic gsplat checkpoint dict; return the tensors used."""
    rng = rng if rng is not None else np.random.default_rng(0)
    n = len(means)
    sp = {
        "means": torch.tensor(means, dtype=torch.float32),
        # deliberately unnormalized: export must normalize
        "quats": torch.tensor(rng.normal(size=(n, 4)), dtype=torch.float32),
        "scales": torch.tensor(rng.normal(size=(n, 3)) - 4.0, dtype=torch.float32),
        "opacities": torch.tensor(
            opacities if opacities is not None else rng.normal(size=n),
            dtype=torch.float32),
        # values in [0,1] -> export takes them as raw rgb (no sigmoid branch)
        "colors": torch.tensor(rng.uniform(0.05, 0.95, size=(n, 3)),
                               dtype=torch.float32),
    }
    torch.save(sp, path)
    return sp


# ---------------------------------------------------------------------------
# se3_exp vs scipy
# ---------------------------------------------------------------------------

def test_se3_exp_matches_scipy() -> None:
    """se3_exp == matrix exponential of the 4x4 twist (independent reference)."""
    pytest.importorskip("gsplat")
    pytest.importorskip("pycolmap")
    pytest.importorskip("PIL")
    from video2sim.train import se3_exp

    rng = np.random.default_rng(3)
    for _ in range(10):
        w = rng.normal(size=3)
        u = rng.normal(size=3)
        d = torch.tensor(np.concatenate([w, u]), dtype=torch.float32)
        T = se3_exp(d).numpy()

        # rotation block vs scipy Rotation
        R_ref = Rotation.from_rotvec(w).as_matrix()
        assert np.allclose(T[:3, :3], R_ref, atol=1e-5)

        # full SE3 vs scipy expm of the twist matrix
        twist = np.zeros((4, 4))
        twist[:3, :3] = _skew(w)
        twist[:3, 3] = u
        assert np.allclose(T, expm(twist), atol=1e-5)
        assert np.allclose(T[3], [0.0, 0.0, 0.0, 1.0])


def test_se3_exp_zero_twist_identity_branch() -> None:
    """theta < 1e-8 branch: R = I, t = u (differentiable small-angle path)."""
    pytest.importorskip("gsplat")
    pytest.importorskip("pycolmap")
    pytest.importorskip("PIL")
    from video2sim.train import se3_exp

    u = np.array([0.1, -0.2, 0.3])
    d = torch.tensor(np.concatenate([np.zeros(3), u]), dtype=torch.float32)
    T = se3_exp(d).numpy()
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-7)
    assert np.allclose(T[:3, 3], u, atol=1e-7)


# ---------------------------------------------------------------------------
# 62-prop SH3 PLY writer/reader roundtrip
# ---------------------------------------------------------------------------

def test_ply_writer_reader_roundtrip_62prop(tmp_path: Path) -> None:
    from video2sim.export import C0, export_sh3_ply, read_ply

    rng = np.random.default_rng(7)
    n = 100
    means = rng.normal(size=(n, 3)).astype(np.float32)
    splat_pt = tmp_path / "splat.pt"
    sp = _fake_splat_pt(splat_pt, means, rng=rng)

    out_ply = tmp_path / "splat_sh3.ply"
    n_written = export_sh3_ply(splat_pt, out_ply)
    assert n_written == n

    arr, props = read_ply(out_ply)
    names = [nm for nm, _ in props]
    expected = (
        ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(45)]
        + ["opacity", "scale_0", "scale_1", "scale_2",
           "rot_0", "rot_1", "rot_2", "rot_3"]
    )
    assert names == expected
    assert len(props) == 62  # the NuRec-required 62-prop layout
    assert len(arr) == n

    # positions / opacity logits / log-scales roundtrip exactly (float32)
    assert np.allclose(np.stack([arr["x"], arr["y"], arr["z"]], 1), means)
    assert np.allclose(arr["opacity"], sp["opacities"].numpy())
    assert np.allclose(
        np.stack([arr[f"scale_{i}"] for i in range(3)], 1), sp["scales"].numpy())

    # SH3 rest bands are ZERO-padded (SH0-only payloads hit the NuRec
    # ~47 s/frame slow path)
    rest = np.stack([arr[f"f_rest_{i}"] for i in range(45)], 1)
    assert np.all(rest == 0.0)

    # normals are zero placeholders
    assert np.all(np.stack([arr["nx"], arr["ny"], arr["nz"]], 1) == 0.0)

    # quats normalized on export
    rot = np.stack([arr[f"rot_{i}"] for i in range(4)], 1)
    q_ref = sp["quats"].numpy()
    q_ref = q_ref / np.linalg.norm(q_ref, axis=1, keepdims=True)
    assert np.allclose(rot, q_ref, atol=1e-6)
    assert np.allclose(np.linalg.norm(rot, axis=1), 1.0, atol=1e-6)

    # standard 3DGS DC encoding: rgb = 0.5 + C0 * f_dc
    f_dc = np.stack([arr[f"f_dc_{i}"] for i in range(3)], 1)
    assert np.allclose(0.5 + C0 * f_dc, sp["colors"].numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# COLMAP text model write -> pycolmap read
# ---------------------------------------------------------------------------

def test_colmap_text_write_pycolmap_read(tmp_path: Path) -> None:
    o3d = pytest.importorskip("open3d")
    pytest.importorskip("pycolmap")
    PIL_Image = pytest.importorskip("PIL.Image")
    from video2sim.to_colmap import export_colmap, validate_pycolmap

    rng = np.random.default_rng(11)
    n_frames, pred_w, pred_h = 6, 16, 12
    full_w, full_h = 32, 24  # full res = 2x pred res

    # LingBot-shaped predictions file: world->cam extrinsics, pred-res intrinsics
    R_all = Rotation.random(n_frames, random_state=0).as_matrix()
    t_all = rng.normal(scale=0.5, size=(n_frames, 3))
    E = np.concatenate([R_all, t_all[:, :, None]], axis=2).astype(np.float32)
    K = np.tile(np.array([[8.0, 0.0, 8.0],
                          [0.0, 8.0, 6.0],
                          [0.0, 0.0, 1.0]], dtype=np.float32),
                (n_frames, 1, 1))
    pred_pt = tmp_path / "pred-test.pt"
    torch.save({"predictions": {
        "extrinsic": torch.tensor(E),
        "intrinsic": torch.tensor(K),
        "depth": torch.zeros(n_frames, pred_h, pred_w, 1),
    }}, pred_pt)

    # full-res jpgs, sorted order == prediction order
    frames_dir = tmp_path / "src_frames"
    frames_dir.mkdir()
    for i in range(n_frames):
        PIL_Image.new("RGB", (full_w, full_h), (i * 20, 64, 128)).save(
            frames_dir / f"f_{i + 1:04d}.jpg")

    # TSDF init cloud
    n_pts, max_points = 200, 120
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(rng.normal(size=(n_pts, 3)))
    pc.colors = o3d.utility.Vector3dVector(rng.uniform(size=(n_pts, 3)))
    cloud_ply = tmp_path / "cloud.ply"
    assert o3d.io.write_point_cloud(str(cloud_ply), pc)

    workdir = tmp_path / "work"
    sparse = export_colmap(pred=pred_pt, cloud=cloud_ply, frames_dir=frames_dir,
                           workdir=workdir, max_points=max_points)
    assert (sparse / "cameras.txt").is_file()
    assert (sparse / "images.txt").is_file()
    assert (sparse / "points3D.txt").is_file()
    assert (workdir / "frames").exists()  # symlink for the trainer

    nc, ni, np3 = validate_pycolmap(sparse)
    assert (nc, ni, np3) == (1, n_frames, max_points)

    # pose integrity: pycolmap must give back the exact world->cam extrinsics
    import pycolmap
    rec = pycolmap.Reconstruction(str(sparse))
    cam = rec.cameras[1]
    assert (cam.width, cam.height) == (full_w, full_h)
    # shared PINHOLE intrinsics scaled from pred res to full res (x2)
    assert np.allclose(
        [cam.focal_length_x, cam.focal_length_y,
         cam.principal_point_x, cam.principal_point_y],
        [16.0, 16.0, 16.0, 12.0], atol=1e-6)
    for img in rec.images.values():
        i = int(img.name[2:6]) - 1  # f_%04d.jpg
        cfw = img.cam_from_world()
        assert np.allclose(cfw.rotation.matrix(), E[i, :3, :3], atol=1e-5)
        assert np.allclose(cfw.translation, E[i, :3, 3], atol=1e-5)


# ---------------------------------------------------------------------------
# quaternion helpers roundtrip
# ---------------------------------------------------------------------------

def _branch_rotations() -> np.ndarray:
    """Rotations exercising every rot_to_qvec branch (trace > 0 and <= 0)."""
    special = [
        np.eye(3),
        Rotation.from_rotvec([np.pi, 0, 0]).as_matrix(),   # i-branch
        Rotation.from_rotvec([0, np.pi, 0]).as_matrix(),   # j-branch
        Rotation.from_rotvec([0, 0, np.pi]).as_matrix(),   # k-branch
        Rotation.from_rotvec(np.pi * np.array([0.6, 0.64, 0.48])).as_matrix(),
    ]
    return np.stack(special + list(Rotation.random(8, random_state=5).as_matrix()))


def test_rot_to_qvec_quat_to_mat_roundtrip() -> None:
    pytest.importorskip("open3d")
    pytest.importorskip("PIL")
    pytest.importorskip("gsplat")
    pytest.importorskip("pycolmap")
    from video2sim.to_colmap import rot_to_qvec
    from video2sim.train import mat_to_quat, quat_to_mat

    for R in _branch_rotations():
        q = rot_to_qvec(R)
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)

        # qvec -> matrix closes the loop
        assert np.allclose(quat_to_mat(q), R, atol=1e-9)

        # matches scipy (COLMAP wxyz vs scipy xyzw, up to global sign)
        q_scipy = np.roll(Rotation.from_matrix(R).as_quat(), 1)
        assert (np.allclose(q, q_scipy, atol=1e-9)
                or np.allclose(q, -q_scipy, atol=1e-9))

        # the trainer's independent implementation agrees (up to sign)
        q2 = mat_to_quat(R)
        assert (np.allclose(q2, q, atol=1e-9)
                or np.allclose(q2, -q, atol=1e-9))
        assert np.allclose(quat_to_mat(q2), R, atol=1e-9)


# ---------------------------------------------------------------------------
# free-space pruning on a synthetic cloud
# ---------------------------------------------------------------------------

def test_free_space_pruning_synthetic(tmp_path: Path) -> None:
    from video2sim.export import export_sh3_ply, prune_floaters, read_ply

    rng = np.random.default_rng(23)
    radius = 0.20

    # surface oracle: a 1x1 m floor plane sampled at 5 cm
    xs = np.linspace(0.0, 1.0, 21)
    gx, gy = np.meshgrid(xs, xs)
    surface = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
    tsdf_ply = tmp_path / "tsdf.ply"
    _write_xyz_ply(tsdf_ply, surface)

    # 30 near-surface gaussians (5 cm off the plane) + 10 floaters at z=1.5
    n_near, n_far = 30, 10
    near = np.column_stack([rng.uniform(0, 1, size=(n_near, 2)),
                            np.full(n_near, 0.05)])
    far = np.column_stack([rng.uniform(0, 1, size=(n_far, 2)),
                           np.full(n_far, 1.5)])
    means = np.vstack([near, far]).astype(np.float32)

    # MCMC contract: give half the near gaussians ~0.2% opacity — they must
    # SURVIVE at the default min_opacity=0.0 (opacity pruning deletes ~79% of
    # an MCMC ensemble; the criterion is SPATIAL only)
    opacities = rng.normal(size=n_near + n_far)
    opacities[:n_near // 2] = -6.0
    splat_pt = tmp_path / "splat.pt"
    _fake_splat_pt(splat_pt, means, opacities=opacities, rng=rng)
    splat_ply = tmp_path / "splat_sh3.ply"
    export_sh3_ply(splat_pt, splat_ply)

    pruned_ply = tmp_path / "splat_pruned.ply"
    kept = prune_floaters(splat_ply, tsdf_ply, pruned_ply,
                          radius=radius, min_opacity=0.0)
    assert kept == n_near  # every floater dropped, every near gaussian kept

    arr, props = read_ply(pruned_ply)
    assert len(props) == 62  # pruning must not change the prop layout
    assert len(arr) == n_near
    assert np.allclose(arr["z"], 0.05, atol=1e-6)  # only near-surface survive
    # the low-opacity near-surface gaussians are still there
    assert int((arr["opacity"] < -5.0).sum()) == n_near // 2


if __name__ == "__main__":  # manual run: python tests/test_smoke.py
    raise SystemExit(pytest.main([__file__, "-q"]))
