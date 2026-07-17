"""video2sim stage 1 — video -> frames directory via ffmpeg.

What: decodes a phone video into numbered JPEG frames (``f_%04d.jpg``) for the
SfM stage. Downstream (COLMAP/GLOMAP) treats the frames as ONE camera track
(``--single-camera`` is MANDATORY for video SfM), so every frame must come out
of the exact same decode/scale path — a single ffmpeg invocation guarantees that.

Why fps=8: dense enough for sequential matching on a walking-speed phone pass,
sparse enough to keep feature extraction tractable. Downstream uses stride 1
always — thinning happens HERE, at extraction, not later.

Validated behavior source (do not "improve"):
  ffmpeg -i V -vf fps=8 -q:v 2 out/f_%04d.jpg
  - landscape source used no scale (already at/below target long side);
  - portrait upright source used scale=-2:1036 (height = long side) — exposed
    here as --long-side.
  - ffmpeg auto-applies rotation metadata (autorotate) BEFORE the filter chain
    — do NOT add a transpose filter, it would double-rotate. For the same
    reason orientation must be judged on the DISPLAY size (rotation-corrected),
    not the stored stream size.

Interpreter contract: runs under the main venv
/home/perelman/Basic_RL/.venv/bin/python (stdlib-only module; needs the
``ffmpeg``/``ffprobe`` binaries on PATH). Do not auto-switch interpreters.

Usage:
  python -m video2sim.extract VIDEO.mp4 [--workdir W] [--out DIR] [--fps 8] [--long-side 1920]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

TAG = "[extract]"

FRAME_PATTERN = "f_%04d.jpg"
FRAME_GLOB = "f_*.jpg"


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


def _require_binary(name: str) -> str:
    """Resolve an executable on PATH or fail with a clear message."""
    path = shutil.which(name)
    if path is None:
        raise SystemExit(
            f"{TAG} ERROR: '{name}' not found on PATH. "
            f"Install ffmpeg (e.g. 'sudo apt install ffmpeg') and retry."
        )
    return path


def probe_display_size(video: Path) -> Tuple[int, int]:
    """Return (width, height) of the video as DISPLAYED.

    ffmpeg autorotates using the rotation metadata before any -vf filter runs,
    so the scale decision must use rotation-corrected dimensions: a portrait
    phone clip is often stored landscape + rot=90/270.
    """
    ffprobe = _require_binary("ffprobe")
    # Full -show_streams JSON: the 'stream_side_data' show_entries section only
    # exists in ffprobe >= 5; parsing the full stream dict works on 4.4 too.
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_streams",
        "-of", "json",
        str(video),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{TAG} ERROR: ffprobe failed on {video}:\n{proc.stderr.strip()}")
    try:
        stream = json.loads(proc.stdout)["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{TAG} ERROR: could not parse ffprobe output for {video}: {exc}")

    rotation = 0
    # Modern ffprobe: displaymatrix side data; older files: 'rotate' stream tag.
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = int(round(float(side_data["rotation"])))
            break
    else:
        tag_rotate = stream.get("tags", {}).get("rotate")
        if tag_rotate is not None:
            rotation = int(round(float(tag_rotate)))

    if abs(rotation) % 180 == 90:
        width, height = height, width
    return width, height


def build_vf(fps: float, display_size: Tuple[int, int], long_side: int) -> str:
    """Build the -vf chain: fps, plus a scale term when downscaling is needed.

    - long_side <= 0 disables scaling entirely.
    - No upscaling: if the display long side is already <= long_side, skip the
      scale term (this reproduces the validated landscape run, which used no
      scale).
    - Portrait (h >= w): scale=-2:{long_side} (height is the long side, as in
      the validated scale=-2:1036 run). Landscape: scale={long_side}:-2.
      -2 keeps the derived dimension even, which JPEG/SfM tooling expects.
    """
    filters = [f"fps={fps:g}"]
    width, height = display_size
    if long_side > 0 and max(width, height) > long_side:
        if height >= width:
            filters.append(f"scale=-2:{long_side}")
        else:
            filters.append(f"scale={long_side}:-2")
    return ",".join(filters)


def extract_frames(
    video: Path,
    out_dir: Path,
    fps: float = 8.0,
    long_side: int = 1920,
    force: bool = False,
) -> Tuple[int, Tuple[int, int]]:
    """Extract frames from `video` into `out_dir`.

    Returns (frame_count, (frame_width, frame_height)).
    """
    ffmpeg = _require_binary("ffmpeg")
    video = video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"{TAG} ERROR: video not found: {video}")

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(out_dir.glob(FRAME_GLOB))
    if stale:
        if not force:
            # Stale frames silently poison the SfM track (mixed resolutions /
            # non-contiguous numbering with --single-camera). Refuse by default.
            raise SystemExit(
                f"{TAG} ERROR: {out_dir} already holds {len(stale)} '{FRAME_GLOB}' frames. "
                f"Pass --force to delete and re-extract."
            )
        _log(f"--force: removing {len(stale)} stale frames from {out_dir}")
        for frame in stale:
            frame.unlink()

    display_size = probe_display_size(video)
    vf = build_vf(fps, display_size, long_side)
    _log(f"source display size: {display_size[0]}x{display_size[1]}  vf: '{vf}'")

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-stats", "-y",
        "-i", str(video),
        "-vf", vf,
        # -q:v 2 = near-lossless JPEG; feature extractors are quality-sensitive.
        "-q:v", "2",
        str(out_dir / FRAME_PATTERN),
    ]
    _log("running: " + " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"{TAG} ERROR: ffmpeg exited with code {proc.returncode}")

    frames = sorted(out_dir.glob(FRAME_GLOB))
    if not frames:
        raise SystemExit(f"{TAG} ERROR: ffmpeg produced no frames in {out_dir}")
    frame_size = probe_display_size(frames[0])
    _log(f"extracted {len(frames)} frames at {frame_size[0]}x{frame_size[1]} -> {out_dir}")
    return len(frames), frame_size


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m video2sim.extract",
        description="video2sim stage 1: video -> frames directory via ffmpeg.",
    )
    parser.add_argument("video", type=Path, help="input video file (phone capture)")
    parser.add_argument(
        "--workdir", type=Path, default=Path("."),
        help="pipeline working directory; defaults for other paths are relative to it (default: .)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="frames output directory (default: <workdir>/frames)",
    )
    parser.add_argument("--fps", type=float, default=8.0, help="extraction rate (default: 8)")
    parser.add_argument(
        "--long-side", type=int, default=1920,
        help="downscale so the display long side is at most this; <=0 disables scaling; "
             "never upscales (default: 1920)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="delete pre-existing f_*.jpg frames in --out instead of aborting",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out if args.out is not None else args.workdir / "frames"
    extract_frames(args.video, out_dir, fps=args.fps, long_side=args.long_side, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
