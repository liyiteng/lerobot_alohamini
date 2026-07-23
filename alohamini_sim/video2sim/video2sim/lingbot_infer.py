"""LingBot-Map streaming inference wrapper (video2sim geometry stage).

What: runs the RTX4060-8GB fork's ``scripts/predict_stream.py`` as a
subprocess over an extracted-frames directory and reports the produced
``pred-*.pt`` predictions file (depth / extrinsics / intrinsics per frame).

Why a subprocess: the fork script owns its own CUDA allocator setup
(``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` must be exported
*before* ``import torch``), so it cannot be safely imported into an
already-torch-initialized process.

Pipeline stage: frames (stage 1, frame extraction) -> THIS (monocular
streaming 3D reconstruction) -> point cloud / pose export for splat
training and scene assembly.

Interpreter contract (documented, not auto-switched): the child process
runs under the main venv ``/home/perelman/Basic_RL/.venv/bin/python``
(torch/gsplat/pycolmap/open3d/scipy). LingBot inference additionally needs
``PYTHONPATH=<lingbot fork>`` and ``ninja`` on PATH (this wrapper sets
both). This wrapper module itself is stdlib-only and runs anywhere.

8GB-GPU recipe baked in (do not "improve"):
  * fp8 KV cache + sliding-window 48 (``--kv_cache_fp8``);
  * ``--max_frame_num`` sized to the actual sequence, not the fork's 1024
    default (the FlashInfer paged-KV pool is pre-allocated from it);
  * stride 1 always;
  * GPU must be exclusive — Isaac GUI alone eats 3-4GB.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

TAG = "[lingbot]"

# Durable checkout of the RTX4060-8GB fork (branch rtx4060_8g, incl. the
# validated local mods to flashinfer_cache.py / create_ply.py /
# predict_long.py). Do NOT point this at a session scratchpad path.
DEFAULT_LINGBOT_REPO = Path(
    "/home/perelman/lingbot-map/lingbot-map-rtx4060-8g"
)
DEFAULT_MODEL = Path("/home/perelman/lingbot-map/ckpt/lingbot-map-long.pt")
DEFAULT_PYTHON = Path("/home/perelman/Basic_RL/.venv/bin/python")

# Extensions must match the fork loader's load_images() default
# (image_ext=".jpg,.png,.JPG", case-sensitive glob on Linux) so our frame
# count — and therefore --max_frame_num — equals what the child loads.
_FRAME_EXTS = (".jpg", ".png", ".JPG")

# GPU exclusivity guard: CUDA SfM / LingBot inference need the card to
# themselves; a lingering Isaac GUI eats 3-4GB and OOMs the paged-KV pool.
_GPU_BUSY_MIB = 2048


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


def count_frames(frames_dir: Path) -> int:
    """Count frames exactly as the fork's load_images() will glob them."""
    return sum(1 for p in frames_dir.iterdir()
               if p.is_file() and p.name.endswith(_FRAME_EXTS))


def warn_if_gpu_busy(threshold_mib: int = _GPU_BUSY_MIB) -> None:
    """Warn if nvidia-smi reports >threshold MiB already in use on any GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        _log(f"WARNING: could not query nvidia-smi ({e}); "
             "cannot verify GPU exclusivity")
        return
    for i, line in enumerate(out.strip().splitlines()):
        try:
            used = int(line.strip())
        except ValueError:
            continue
        if used > threshold_mib:
            _log(f"WARNING: GPU {i} already has {used} MiB in use "
                 f"(> {threshold_mib}). LingBot needs the GPU exclusively "
                 "(Isaac GUI alone eats 3-4GB) — expect OOM in the "
                 "FlashInfer paged-KV pool if you proceed.")


def run_lingbot_infer(
    frames: Path,
    out: Path,
    lingbot_repo: Path = DEFAULT_LINGBOT_REPO,
    model: Path = DEFAULT_MODEL,
    sliding_window: int = 48,
    num_scale_frames: int = 2,
    stride: int = 1,
    python: Path = DEFAULT_PYTHON,
) -> Path:
    """Run predict_stream.py over ``frames``; return the produced pred-*.pt.

    ``stride`` should stay 1 (pipeline invariant); it is exposed only for
    smoke tests on long captures.
    """
    frames = frames.resolve()
    out = out.resolve()
    lingbot_repo = lingbot_repo.resolve()
    script = lingbot_repo / "scripts" / "predict_stream.py"
    if not script.is_file():
        raise FileNotFoundError(f"predict_stream.py not found: {script}")
    if not frames.is_dir():
        raise FileNotFoundError(f"frames dir not found: {frames}")

    n_frames = count_frames(frames)
    if n_frames == 0:
        raise FileNotFoundError(
            f"no frames matching {_FRAME_EXTS} in {frames}")
    # Frames actually ingested after stride (stride 1 always in-pipeline).
    n_loaded = -(-n_frames // stride)
    # FlashInfer pre-allocates the paged-KV pool from max_frame_num
    # (special pages ~ max_frames*6/page_size per block x24 blocks); the
    # fork's default of 1024 was 6.7GB on its own -> OOM on 8GB. Size it
    # to the actual sequence, +8 headroom.
    max_frame_num = n_loaded + 8
    _log(f"{n_frames} frames in {frames} "
         f"(stride {stride} -> {n_loaded} loaded, max_frame_num {max_frame_num})")

    warn_if_gpu_busy()

    env = os.environ.copy()
    # PYTHONPATH must point at the fork: lingbot_map.utils.loadimage exists
    # only in the fork and must shadow the editable install's package.
    prev_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(lingbot_repo) + (os.pathsep + prev_pp if prev_pp else ""))
    # The venv bin dir must lead PATH: the fp8 KV cache path JIT-compiles a
    # CUDA extension and needs ninja (installed in the venv) resolvable.
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")

    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python), str(script),
        "--image_folder", str(frames),
        "--model_path", str(model),
        "--output_dir", str(out),
        "--stride", str(stride),
        "--num_scale_frames", str(num_scale_frames),
        # fp8 KV + sliding-window 48 is the LingBot 8GB recipe. NOTE the
        # fork keeps use_sdpa=False by default (issue #79: FlashInfer
        # paged-KV attention backend) and image_size=518 — the ckpt
        # pos_embed (1,1370,1024) is size-bound, so never override it.
        "--kv_cache_sliding_window", str(sliding_window),
        "--kv_cache_fp8",
        "--max_frame_num", str(max_frame_num),
    ]
    _log("exec: " + " ".join(cmd))
    t0 = time.time()
    # cwd = fork repo (paths above are absolute, so this is safe) to keep
    # any repo-relative asset lookups working.
    proc = subprocess.run(cmd, env=env, cwd=str(lingbot_repo))
    if proc.returncode != 0:
        raise RuntimeError(
            f"predict_stream.py failed with exit code {proc.returncode}")
    _log(f"inference subprocess done in {time.time() - t0:.1f}s")

    pred = _locate_pred(out, newer_than=t0)
    _log(f"PRED -> {pred}")
    return pred


def _locate_pred(out: Path, newer_than: float) -> Path:
    """Find the pred-*.pt this run produced (newest; must postdate launch)."""
    preds = sorted(out.glob("pred-*.pt"), key=lambda p: p.stat().st_mtime)
    if not preds:
        raise FileNotFoundError(f"no pred-*.pt produced in {out}")
    newest = preds[-1]
    if newest.stat().st_mtime < newer_than:
        _log(f"WARNING: newest pred file predates this run: {newest}")
    return newest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run LingBot-Map predict_stream.py (8GB fp8-KV recipe) "
                    "over an extracted-frames directory.")
    ap.add_argument("--workdir", type=Path, default=Path("."),
                    help="pipeline working dir; default root for --frames/--out")
    ap.add_argument("--frames", type=Path, default=None,
                    help="frames dir (default: WORKDIR/frames)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir for pred-*.pt (default: WORKDIR/lingbot)")
    ap.add_argument("--lingbot-repo", type=Path, default=DEFAULT_LINGBOT_REPO,
                    help="LingBot-Map RTX4060-8GB fork checkout")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--sliding-window", type=int, default=48,
                    help="KV-cache sliding window; 48 + fp8 KV is the 8GB recipe")
    ap.add_argument("--num-scale-frames", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1,
                    help="frame stride; keep 1 (pipeline invariant)")
    ap.add_argument("--python", type=Path, default=DEFAULT_PYTHON,
                    help="interpreter for the child (main venv; has torch + ninja)")
    args = ap.parse_args(argv)

    frames = args.frames if args.frames is not None else args.workdir / "frames"
    out = args.out if args.out is not None else args.workdir / "lingbot"

    pred = run_lingbot_infer(
        frames=frames, out=out, lingbot_repo=args.lingbot_repo,
        model=args.model, sliding_window=args.sliding_window,
        num_scale_frames=args.num_scale_frames, stride=args.stride,
        python=args.python)
    print(pred, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
