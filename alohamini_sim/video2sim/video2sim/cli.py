"""video2sim end-to-end runner — thin orchestrator over the stage modules.

What: drives the whole phone-video -> photoreal Isaac scene pipeline as a
sequence of stage SUBPROCESSES, each `python -m video2sim.<stage>` under the
interpreter the stage's contract demands. Contains NO business logic — every
constant that matters lives in the stage modules; this file only sequences,
times, resumes and routes interpreters.

Stage order (each idempotent — skipped when its sentinel output exists):
  extract       video -> frames                      (main venv)
  lingbot_infer frames -> lingbot/pred-*.pt          (main venv; wrapper sets
                                                      PYTHONPATH + ninja PATH)
  fuse          pred -> fuse/mvtsdf.ply + _mesh.ply  (main venv)
  refine        fuse mesh -> final_* metric/aligned  (main venv)
  to_colmap     pred + cloud -> sparse/0             (main venv)
  train         sparse/0 + frames -> splat.pt        (main venv; GPU exclusive)
  export        splat.pt -> export/scene.usdz        (main venv; USDZ substep
                                                      subprocessed to NuRec venv
                                                      by the export module)
  scene_prep    final_mesh.ply -> collider.npz       (main venv)
  scene         usdz+align+collider -> scene/scene.usd (Isaac venv; optional,
                                                      runs with --gui / --until scene)

Interpreter contract (documented, not auto-switched): this module itself is
stdlib(+yaml)-only and runs anywhere; the interpreters for the child stages
come from the config:
  main venv:   /home/perelman/Basic_RL/.venv/bin/python (torch/gsplat/pycolmap/open3d/scipy)
  NuRec venv:  /home/perelman/nurec-venv/bin/python (used internally by export)
  Isaac venv:  /home/perelman/isaac5-venv/bin/python (scene stage)
LingBot inference PYTHONPATH/ninja setup is owned by video2sim.lingbot_infer.

Config: a YAML file (--config) merged over built-in defaults, then CLI flags
override the YAML. Recognized top-level keys: main_python / nurec_python /
isaac_python / lingbot_repo / lingbot_model / robot / video / workdir, and
``stage_args`` — per-stage extra CLI arguments passed verbatim to the stage
module (appended LAST, so they win over the orchestrator's explicit flags):

  stage_args:
    train: {iters: 20000, video: true}     # dict: key -> --key value (verbatim,
    fuse: ["--frame_stride", "1"]          #   no underscore/hyphen rewriting)

Usage:
  python -m video2sim run VIDEO.mp4 --workdir OUT [--config cfg.yaml]
      [--from STAGE] [--until STAGE] [--gui] [--force]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TAG = "[cli]"

STAGES: Tuple[str, ...] = (
    "extract", "lingbot_infer", "fuse", "refine", "to_colmap",
    "train", "export", "scene_prep", "scene",
)

# `scene` is optional by default: it needs the Isaac venv + enough free GPU
# (the Isaac GUI alone eats 3-4GB). It runs when --gui is passed or when
# --until scene is requested explicitly.
DEFAULT_UNTIL = "scene_prep"

DEFAULTS: Dict[str, Any] = {
    "main_python": "/home/perelman/Basic_RL/.venv/bin/python",
    "nurec_python": "/home/perelman/nurec-venv/bin/python",
    "isaac_python": "/home/perelman/isaac5-venv/bin/python",
    "lingbot_repo": None,   # None -> lingbot_infer module default
    "lingbot_model": None,  # None -> lingbot_infer module default
    "robot": None,          # None -> scene module default (AM2 Pro parallel)
    "video": None,
    "workdir": None,
    "stage_args": {},
}

# Package root (parent of the video2sim package dir): prepended to PYTHONPATH
# so every child interpreter resolves `-m video2sim.<stage>` regardless of cwd.
_PKG_ROOT = Path(__file__).resolve().parent.parent


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """DEFAULTS <- YAML file (when given). CLI overrides are applied later."""
    cfg = dict(DEFAULTS)
    if config_path is None:
        return cfg
    try:
        import yaml
    except ImportError:
        raise SystemExit(f"{TAG} ERROR: --config given but PyYAML is not importable "
                         f"under {sys.executable}")
    if not config_path.is_file():
        raise SystemExit(f"{TAG} ERROR: config not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"{TAG} ERROR: config must be a YAML mapping: {config_path}")
    cfg.update(loaded)
    return cfg


def stage_extra_args(cfg: Dict[str, Any], stage: str) -> List[str]:
    """Normalize cfg['stage_args'][stage] to a flat argv list.

    dict -> ``--key value`` pairs (key used verbatim — stage modules differ on
    underscore vs hyphen flags, so NO rewriting); ``true`` -> bare flag,
    ``false``/``null`` -> dropped. list -> passed through as strings.
    """
    spec = (cfg.get("stage_args") or {}).get(stage)
    if spec is None:
        return []
    if isinstance(spec, dict):
        out: List[str] = []
        for k, v in spec.items():
            flag = f"--{k}"
            if v is True:
                out.append(flag)
            elif v is False or v is None:
                continue
            else:
                out.extend([flag, str(v)])
        return out
    if isinstance(spec, (list, tuple)):
        return [str(x) for x in spec]
    raise SystemExit(f"{TAG} ERROR: stage_args.{stage} must be a mapping or list")


def sentinel(stage: str, workdir: Path) -> Tuple[bool, str]:
    """(done, human-readable sentinel) for skip-if-output-exists resume."""
    W = workdir
    if stage == "extract":
        hits = sorted((W / "frames").glob("f_*.jpg"))
        return bool(hits), f"{W / 'frames'}/f_*.jpg ({len(hits)} frames)"
    if stage == "lingbot_infer":
        hits = sorted((W / "lingbot").glob("pred-*.pt"))
        return bool(hits), f"{W / 'lingbot'}/pred-*.pt ({len(hits)} files)"
    if stage == "fuse":
        p = W / "fuse" / "mvtsdf.ply"
        return p.exists() and p.with_name("mvtsdf_mesh.ply").exists(), str(p)
    if stage == "refine":
        p = W / "final_align.npz"
        return p.exists() and (W / "final_mesh.ply").exists(), str(p)
    if stage == "to_colmap":
        p = W / "sparse" / "0" / "points3D.txt"
        return p.exists(), str(p)
    if stage == "train":
        p = W / "splat.pt"
        return p.exists(), str(p)
    if stage == "export":
        p = W / "export" / "scene.usdz"
        return p.exists(), str(p)
    if stage == "scene_prep":
        p = W / "collider.npz"
        return p.exists(), str(p)
    if stage == "scene":
        p = W / "scene" / "scene.usd"
        return p.exists(), str(p)
    raise ValueError(f"unknown stage {stage!r}")


def build_cmd(
    stage: str,
    cfg: Dict[str, Any],
    workdir: Path,
    video: Optional[Path],
    force: bool,
    gui: bool,
) -> List[str]:
    """argv for one stage subprocess (explicit paths bridge stage-default gaps)."""
    main_py = str(cfg["main_python"])
    W = str(workdir)

    def base(mod: str, python: str = main_py) -> List[str]:
        return [python, "-m", f"video2sim.{mod}", "--workdir", W]

    if stage == "extract":
        if video is None:
            raise SystemExit(
                f"{TAG} ERROR: extract must run but no VIDEO was given "
                f"(pass it positionally or as config key 'video')")
        cmd = [main_py, "-m", "video2sim.extract", str(video), "--workdir", W]
        if force:
            cmd.append("--force")  # extract refuses to clobber stale frames otherwise
    elif stage == "lingbot_infer":
        cmd = base("lingbot_infer") + ["--python", main_py]
        if cfg.get("lingbot_repo"):
            cmd += ["--lingbot-repo", str(cfg["lingbot_repo"])]
        if cfg.get("lingbot_model"):
            cmd += ["--model", str(cfg["lingbot_model"])]
    elif stage == "fuse":
        cmd = base("fuse")
    elif stage == "refine":
        # refine's own defaults probe WORKDIR/fuse/ then WORKDIR root; pass
        # the fuse outputs explicitly anyway to keep the stage deterministic.
        cmd = base("refine") + [
            "--mesh", str(workdir / "fuse" / "mvtsdf_mesh.ply"),
            "--cloud", str(workdir / "fuse" / "mvtsdf.ply"),
        ]
    elif stage == "to_colmap":
        cmd = base("to_colmap")
    elif stage == "train":
        cmd = base("train")
    elif stage == "export":
        cmd = base("export") + ["--nurec-python", str(cfg["nurec_python"])]
    elif stage == "scene_prep":
        cmd = base("scene_prep")
    elif stage == "scene":
        cmd = base("scene", python=str(cfg["isaac_python"]))
        if cfg.get("robot"):
            cmd += ["--robot", str(cfg["robot"])]
        if gui:
            cmd.append("--gui")
    else:
        raise ValueError(f"unknown stage {stage!r}")
    # config stage_args go LAST so they override the orchestrator's flags
    # (argparse last-occurrence-wins).
    return cmd + stage_extra_args(cfg, stage)


def run_stage(stage: str, cmd: List[str]) -> float:
    """Run one stage subprocess; return elapsed seconds. Exits on failure."""
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_PKG_ROOT) + (os.pathsep + prev if prev else "")
    _log(f"--- {stage}: " + " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(
            f"{TAG} ERROR: stage '{stage}' failed with exit code "
            f"{proc.returncode} after {dt:.1f}s")
    _log(f"--- {stage}: done in {dt:.1f}s")
    return dt


def run_pipeline(
    video: Optional[Path],
    workdir: Path,
    cfg: Dict[str, Any],
    from_stage: str = STAGES[0],
    until_stage: str = DEFAULT_UNTIL,
    gui: bool = False,
    force: bool = False,
) -> None:
    """Run the selected stage range with skip-if-output-exists resume."""
    i0, i1 = STAGES.index(from_stage), STAGES.index(until_stage)
    if i0 > i1:
        raise SystemExit(f"{TAG} ERROR: --from {from_stage} is after --until {until_stage}")
    selected = STAGES[i0:i1 + 1]

    workdir = workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if video is not None:
        video = video.expanduser().resolve()
    _log(f"workdir {workdir} | stages: {' -> '.join(selected)}"
         + (" (--force: re-run even if outputs exist)" if force else ""))

    timings: List[Tuple[str, float]] = []
    t_all = time.time()
    for stage in selected:
        done, what = sentinel(stage, workdir)
        # scene with --gui is an interactive session — always worth (re)opening.
        if done and not force and not (stage == "scene" and gui):
            _log(f"--- {stage}: skip (exists: {what})")
            timings.append((stage, 0.0))
            continue
        cmd = build_cmd(stage, cfg, workdir, video, force, gui)
        timings.append((stage, run_stage(stage, cmd)))

    _log(f"pipeline done in {time.time() - t_all:.1f}s")
    for stage, dt in timings:
        _log(f"  {stage:<14} {'skipped' if dt == 0.0 else f'{dt:8.1f}s'}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m video2sim",
        description="video2sim end-to-end runner: phone video -> photoreal "
                    "Isaac Sim scene with physics collider.",
        epilog="stages: " + " -> ".join(STAGES)
               + f"  ('scene' runs only with --gui or --until scene; "
                 f"default --until {DEFAULT_UNTIL})",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run", help="run the pipeline (idempotent: existing stage outputs are skipped)")
    run.add_argument("video", type=Path, nargs="?", default=None,
                     help="input phone video (optional when resuming past extract)")
    run.add_argument("--workdir", type=Path, default=None,
                     help="pipeline working directory (all stage outputs land here)")
    run.add_argument("--config", type=Path, default=None,
                     help="YAML config merged over defaults; CLI flags override it")
    run.add_argument("--from", dest="from_stage", choices=STAGES, default=STAGES[0],
                     help="first stage to run (default: extract)")
    run.add_argument("--until", dest="until_stage", choices=STAGES, default=None,
                     help=f"last stage to run (default: {DEFAULT_UNTIL}; "
                          f"--gui bumps it to scene)")
    run.add_argument("--gui", action="store_true",
                     help="run the scene stage with the Isaac GUI open "
                          "(implies --until scene unless --until is given)")
    run.add_argument("--force", action="store_true",
                     help="re-run selected stages even when their outputs exist")
    run.add_argument("--main-python", type=Path, default=None,
                     help="main venv interpreter override")
    run.add_argument("--nurec-python", type=Path, default=None,
                     help="NuRec venv interpreter override (export USDZ substep)")
    run.add_argument("--isaac-python", type=Path, default=None,
                     help="Isaac venv interpreter override (scene stage)")
    run.add_argument("--lingbot-repo", type=Path, default=None,
                     help="LingBot-Map fork checkout override")
    run.add_argument("--lingbot-model", type=Path, default=None,
                     help="LingBot checkpoint override")
    run.add_argument("--robot", type=Path, default=None,
                     help="robot USD override for the scene stage")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    for key, val in (("main_python", args.main_python),
                     ("nurec_python", args.nurec_python),
                     ("isaac_python", args.isaac_python),
                     ("lingbot_repo", args.lingbot_repo),
                     ("lingbot_model", args.lingbot_model),
                     ("robot", args.robot)):
        if val is not None:
            cfg[key] = val

    video = args.video if args.video is not None else (
        Path(cfg["video"]) if cfg.get("video") else None)
    workdir = args.workdir if args.workdir is not None else (
        Path(cfg["workdir"]) if cfg.get("workdir") else None)
    if workdir is None:
        raise SystemExit(f"{TAG} ERROR: --workdir is required "
                         f"(or config key 'workdir')")
    until = args.until_stage or ("scene" if args.gui else DEFAULT_UNTIL)

    run_pipeline(video, workdir, cfg,
                 from_stage=args.from_stage, until_stage=until,
                 gui=args.gui, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
