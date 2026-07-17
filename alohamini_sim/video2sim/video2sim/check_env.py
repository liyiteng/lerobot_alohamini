"""video2sim stage 0 — preflight environment check (pass/fail table).

What: verifies every runtime the pipeline touches BEFORE burning hours on a
run: main-venv imports (torch+CUDA, gsplat>=1.5, pycolmap, open3d, scipy,
PIL), the ffmpeg/ffprobe binaries, the NuRec venv (pxr importable), the
Isaac venv interpreter, the LingBot fork checkout + checkpoint, ninja on
the main venv bin (the fp8 KV cache path JIT-compiles a CUDA extension and
needs ninja resolvable when the venv bin leads PATH), GPU VRAM (>= 8 GiB
with current free memory — the GPU must be EXCLUSIVE for CUDA SfM / LingBot
inference / training; an open Isaac GUI alone eats 3-4 GB), and total RAM
(the trainer's uint8 image cache is ~3 GB and fuse warns at 45 GB RSS on
the 31 GB box).

Exit status: nonzero when any check FAILs, with the failures listed at the
end. WARN rows (e.g. a busy GPU) do not fail the check but should be read.

Interpreter contract (do NOT auto-switch): this module is stdlib-only and
runs under any Python; the heavy imports are verified in SUBPROCESSES under
their own interpreters —
  - main venv:    /home/perelman/Basic_RL/.venv/bin/python
                  (torch/gsplat/pycolmap/open3d/scipy/PIL)
  - NuRec export: /home/perelman/nurec-venv/bin/python (pxr)
  - Isaac scene:  /home/perelman/isaac5-venv/bin/python (existence check
                  only — importing isaacsim boots a SimulationApp and
                  grabs the GPU)
  - LingBot:      main venv + PYTHONPATH=<lingbot fork> + ninja on PATH

Usage:
  python -m video2sim.check_env [--main-python P] [--nurec-python P]
      [--isaac-python P] [--lingbot-repo DIR] [--model CKPT.pt]
      [--min-vram-gib 8] [--timeout 240]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

from .lingbot_infer import DEFAULT_LINGBOT_REPO, DEFAULT_MODEL

TAG = "[check-env]"

DEFAULT_MAIN_PYTHON = Path("/home/perelman/Basic_RL/.venv/bin/python")
DEFAULT_NUREC_PYTHON = Path("/home/perelman/nurec-venv/bin/python")
DEFAULT_ISAAC_PYTHON = Path("/home/perelman/isaac5-venv/bin/python")

# gsplat >= 1.5: MCMCStrategy + the packed rasterization signature used by
# video2sim.train.
GSPLAT_MIN = (1, 5)

# GPU exclusivity guard (mirrors video2sim.lingbot_infer._GPU_BUSY_MIB):
# CUDA SfM / LingBot inference / training need the card to themselves; a
# lingering Isaac GUI eats 3-4GB and OOMs the FlashInfer paged-KV pool.
GPU_BUSY_MIB = 2048

# One subprocess under the MAIN venv reports every import independently, so a
# missing scipy still lets torch report its CUDA status.
_MAIN_IMPORTS_SNIPPET = r"""
import json
res = {}

def rec(name, fn):
    try:
        res[name] = {"ok": True, "detail": fn()}
    except Exception as e:
        res[name] = {"ok": False, "detail": "%s: %s" % (type(e).__name__, e)}

def _torch():
    import torch
    cuda = torch.cuda.is_available()
    dev = torch.cuda.get_device_name(0) if cuda else "NO CUDA"
    return {"version": torch.__version__, "cuda": cuda, "device": dev}
rec("torch", _torch)

def _gsplat():
    import gsplat
    return {"version": str(getattr(gsplat, "__version__", "0"))}
rec("gsplat", _gsplat)

def _pycolmap():
    import pycolmap
    return {"version": str(getattr(pycolmap, "__version__", "?"))}
rec("pycolmap", _pycolmap)

def _open3d():
    import open3d
    return {"version": open3d.__version__}
rec("open3d", _open3d)

def _scipy():
    import scipy
    return {"version": scipy.__version__}
rec("scipy", _scipy)

def _pil():
    import PIL
    return {"version": PIL.__version__}
rec("PIL", _pil)

print(json.dumps(res))
"""

_PXR_SNIPPET = """\
from pxr import Usd
v = getattr(Usd, "GetVersion", lambda: None)()
suffix = " (USD %s)" % ".".join(str(x) for x in v) if v else ""
print("pxr ok" + suffix)
"""


class CheckResult(NamedTuple):
    """One row of the pass/fail table."""

    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


def _version_tuple(v: str) -> Tuple[int, ...]:
    """Leading numeric parts of a version string ('1.5.3+cu121' -> (1,5,3))."""
    parts: List[int] = []
    for tok in v.split(".")[:3]:
        num = ""
        for ch in tok:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    return tuple(parts)


def check_main_imports(python: Path, timeout: int) -> List[CheckResult]:
    """Verify torch+CUDA / gsplat>=1.5 / pycolmap / open3d / scipy / PIL.

    Runs one subprocess under the main venv so this checker itself stays
    stdlib-only and interpreter-agnostic.
    """
    names = ["torch", "gsplat", "pycolmap", "open3d", "scipy", "PIL"]
    rows: List[CheckResult] = []
    if not python.is_file():
        rows.append(CheckResult("main venv python", "FAIL", f"not found: {python}"))
        rows += [CheckResult(f"main: {n}", "FAIL", "(main venv python missing)")
                 for n in names]
        return rows
    rows.append(CheckResult("main venv python", "PASS", str(python)))

    try:
        proc = subprocess.run(
            [str(python), "-c", _MAIN_IMPORTS_SNIPPET],
            capture_output=True, text=True, timeout=timeout)
        res = json.loads(proc.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as e:
        err = getattr(e, "stderr", None) or str(e)
        rows += [CheckResult(f"main: {n}", "FAIL", f"import probe failed: {err}")
                 for n in names]
        return rows

    for n in names:
        r = res.get(n, {"ok": False, "detail": "no report"})
        if not r["ok"]:
            rows.append(CheckResult(f"main: {n}", "FAIL", str(r["detail"])))
            continue
        d = r["detail"]
        if n == "torch":
            status = "PASS" if d["cuda"] else "FAIL"
            rows.append(CheckResult(
                "main: torch + CUDA", status,
                f"{d['version']}, cuda={d['cuda']} ({d['device']})"))
        elif n == "gsplat":
            ok = _version_tuple(d["version"]) >= GSPLAT_MIN
            rows.append(CheckResult(
                "main: gsplat >= 1.5", "PASS" if ok else "FAIL",
                d["version"] + ("" if ok else f" (< {'.'.join(map(str, GSPLAT_MIN))})")))
        else:
            rows.append(CheckResult(f"main: {n}", "PASS", d["version"]))
    return rows


def check_binary(name: str) -> CheckResult:
    """Check that an executable resolves on PATH."""
    path = shutil.which(name)
    if path is None:
        return CheckResult(name, "FAIL", "not on PATH")
    return CheckResult(name, "PASS", path)


def check_nurec(python: Path, timeout: int) -> List[CheckResult]:
    """NuRec venv interpreter exists AND pxr imports under it."""
    if not python.is_file():
        return [CheckResult("nurec venv python", "FAIL", f"not found: {python}"),
                CheckResult("nurec: pxr import", "FAIL", "(interpreter missing)")]
    rows = [CheckResult("nurec venv python", "PASS", str(python))]
    try:
        proc = subprocess.run([str(python), "-c", _PXR_SNIPPET],
                              capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return rows + [CheckResult("nurec: pxr import", "FAIL", str(e))]
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "?"
        rows.append(CheckResult("nurec: pxr import", "FAIL", tail))
    else:
        rows.append(CheckResult("nurec: pxr import", "PASS", proc.stdout.strip()))
    return rows


def check_isaac(python: Path) -> CheckResult:
    """Isaac venv interpreter exists (no import — isaacsim would grab the GPU)."""
    if python.is_file():
        return CheckResult("isaac venv python", "PASS", str(python))
    return CheckResult("isaac venv python", "FAIL", f"not found: {python}")


def check_lingbot(repo: Path, model: Path) -> List[CheckResult]:
    """LingBot fork checkout (scripts/predict_stream.py) + checkpoint file."""
    rows: List[CheckResult] = []
    script = repo / "scripts" / "predict_stream.py"
    if script.is_file():
        rows.append(CheckResult("lingbot fork", "PASS", str(repo)))
    else:
        rows.append(CheckResult(
            "lingbot fork", "FAIL", f"predict_stream.py not found under {repo}"))
    if model.is_file():
        size_gb = model.stat().st_size / 2**30
        rows.append(CheckResult("lingbot ckpt", "PASS", f"{model} ({size_gb:.1f} GB)"))
    else:
        rows.append(CheckResult("lingbot ckpt", "FAIL", f"not found: {model}"))
    return rows


def check_ninja(main_python: Path) -> CheckResult:
    """ninja must sit in the MAIN venv bin: the fp8 KV cache path JIT-compiles
    a CUDA extension and lingbot_infer puts that bin dir at the head of PATH."""
    ninja = main_python.parent / "ninja"
    if ninja.is_file():
        return CheckResult("ninja (main venv bin)", "PASS", str(ninja))
    return CheckResult("ninja (main venv bin)", "FAIL",
                       f"not found: {ninja} (fp8 KV JIT build needs it)")


def check_gpu(min_vram_gib: float) -> List[CheckResult]:
    """GPU VRAM >= threshold, and current free memory (exclusivity warning)."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return [CheckResult("GPU VRAM", "FAIL", f"nvidia-smi failed: {e}")]

    rows: List[CheckResult] = []
    # nvidia-smi reports total minus driver-reserved memory (the validated
    # RTX 4060 "8 GB" card shows 8188 MiB), so allow 256 MiB of slack.
    min_mib = min_vram_gib * 1024 - 256
    for i, line in enumerate(out.strip().splitlines()):
        try:
            name, total_s, used_s, free_s = line.rsplit(",", 3)
            total, used, free = int(total_s), int(used_s), int(free_s)
        except ValueError:
            continue
        detail = (f"{name.strip()}: {total} MiB total, {free} MiB free, "
                  f"{used} MiB used")
        if total < min_mib:
            rows.append(CheckResult(
                f"GPU{i} VRAM >= {min_vram_gib:g} GiB", "FAIL", detail))
        elif used > GPU_BUSY_MIB:
            # Not a hard failure, but training/inference will OOM if this
            # stays: the GPU must be exclusive (Isaac GUI alone eats 3-4GB).
            rows.append(CheckResult(
                f"GPU{i} VRAM >= {min_vram_gib:g} GiB", "WARN",
                detail + f" — busy (> {GPU_BUSY_MIB} MiB used); "
                         "GPU must be exclusive for SfM/LingBot/training"))
        else:
            rows.append(CheckResult(
                f"GPU{i} VRAM >= {min_vram_gib:g} GiB", "PASS", detail))
    if not rows:
        rows.append(CheckResult("GPU VRAM", "FAIL", "no GPU reported by nvidia-smi"))
    return rows


def check_ram() -> CheckResult:
    """Report MemTotal (fuse RSS guard is 45 GB; trainer cache ~3 GB uint8)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                kb = int(line.split()[1])
                return CheckResult("RAM total", "PASS", f"{kb / 2**20:.1f} GiB")
        return CheckResult("RAM total", "FAIL", "MemTotal not in /proc/meminfo")
    except OSError as e:
        return CheckResult("RAM total", "FAIL", str(e))


def run_checks(
    main_python: Path = DEFAULT_MAIN_PYTHON,
    nurec_python: Path = DEFAULT_NUREC_PYTHON,
    isaac_python: Path = DEFAULT_ISAAC_PYTHON,
    lingbot_repo: Path = DEFAULT_LINGBOT_REPO,
    model: Path = DEFAULT_MODEL,
    min_vram_gib: float = 8.0,
    timeout: int = 240,
) -> List[CheckResult]:
    """Run every check; return the full row list (no printing)."""
    rows: List[CheckResult] = []
    _log("probing main venv imports (torch/open3d are slow importers) ...")
    rows += check_main_imports(main_python, timeout)
    rows.append(check_binary("ffmpeg"))
    rows.append(check_binary("ffprobe"))
    _log("probing NuRec venv (pxr) ...")
    rows += check_nurec(nurec_python, timeout)
    rows.append(check_isaac(isaac_python))
    rows += check_lingbot(lingbot_repo, model)
    rows.append(check_ninja(main_python))
    rows += check_gpu(min_vram_gib)
    rows.append(check_ram())
    return rows


def print_table(rows: List[CheckResult]) -> List[CheckResult]:
    """Print the pass/fail table; return the FAIL rows."""
    width = max(len(r.name) for r in rows)
    _log("-" * (width + 60))
    for r in rows:
        _log(f"{r.status:<4}  {r.name:<{width}}  {r.detail}")
    _log("-" * (width + 60))
    fails = [r for r in rows if r.status == "FAIL"]
    warns = [r for r in rows if r.status == "WARN"]
    if warns:
        _log(f"{len(warns)} warning(s): " + ", ".join(r.name for r in warns))
    if fails:
        _log(f"FAILURES ({len(fails)}):")
        for r in fails:
            _log(f"  FAIL {r.name}: {r.detail}")
    else:
        _log("all checks passed")
    return fails


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m video2sim.check_env",
        description="video2sim stage 0: verify venvs, binaries, LingBot fork, "
                    "GPU and RAM; nonzero exit on any failure.")
    ap.add_argument("--main-python", type=Path, default=DEFAULT_MAIN_PYTHON,
                    help="main venv interpreter (torch/gsplat/pycolmap/open3d)")
    ap.add_argument("--nurec-python", type=Path, default=DEFAULT_NUREC_PYTHON,
                    help="NuRec venv interpreter (pxr) used by video2sim.export")
    ap.add_argument("--isaac-python", type=Path, default=DEFAULT_ISAAC_PYTHON,
                    help="Isaac venv interpreter used by video2sim.scene")
    ap.add_argument("--lingbot-repo", type=Path, default=DEFAULT_LINGBOT_REPO,
                    help="LingBot-Map RTX4060-8GB fork checkout")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                    help="LingBot checkpoint .pt")
    ap.add_argument("--min-vram-gib", type=float, default=8.0,
                    help="minimum GPU VRAM (the whole recipe is 8GB-bounded)")
    ap.add_argument("--timeout", type=int, default=240,
                    help="per-subprocess timeout in seconds")
    args = ap.parse_args(argv)

    rows = run_checks(
        main_python=args.main_python, nurec_python=args.nurec_python,
        isaac_python=args.isaac_python, lingbot_repo=args.lingbot_repo,
        model=args.model, min_vram_gib=args.min_vram_gib, timeout=args.timeout)
    fails = print_table(rows)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
