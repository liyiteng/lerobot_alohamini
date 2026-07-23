"""ASPIRE execution and state-only dataset generation for AlohaMini Pro."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Root of the vendored package tree (upstream: the AlohaMini `maniskill/` checkout;
# here: `alohamini_sim/data_engine/`). Putting it on sys.path makes the original
# top-level imports (`data_gen.*`, `agents.*`) resolve from the vendored copies.
MANISKILL_ROOT = Path(__file__).resolve().parents[2]
if str(MANISKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(MANISKILL_ROOT))

# Explicit registration imports (upstream relied on `import data_gen` importing
# .tasks as a side effect; the vendored data_gen/__init__ is kept import-safe for
# the LeRobot bridge, so register the agent uids and env ids here instead).
import agents.aloha_mini  # noqa: F401  (registers aloha_mini_so100_v2 / _pro_v2 / _pro_v3)
import data_gen.tasks  # noqa: F401  (registers AlohaMiniMultiYCB-v1 and friends)
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
from data_gen.aspire_engine.skills_runtime import PITCH_DEFAULT, SkillRuntime
from data_gen.aspire_engine.writer_adapter import AspireStateWriter
from data_gen.intern_engine.skills.ik import actor_position, resolve_actor

DEFAULT_OBJECT = "077_rubiks_cube"
DEFAULT_TARGET_XY = np.array([0.06, -0.31], np.float32)
PICK_X_RANGE = (-0.20, 0.02)
PICK_Y_RANGE = (-0.34, -0.28)
PICK_CENTER_XY = np.array([-0.13, -0.31], np.float32)
PICK_CENTER_JITTER = np.array([0.006, 0.006], np.float32)


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _default_cfg(out_dir: str | Path | None = None) -> dict[str, Any]:
    # Smoke and v1 dataset generation are state-only. Set render_mode to
    # "rgb_array" in cfg on a host with a render device if RGB hooks are needed.
    return {
        "render_mode": None,
        "render_backend": "none",
        "object_xy_noise": 0.01,
        "base_xy": (-0.40, 0.18),
        "fps": 20,
        "out_dir": str(out_dir) if out_dir is not None else None,
        "overwrite": True,
        "dataset_name": "aloha_mini_pro_aspire_state",
        "robot_type": "aloha_mini_pro_v2",
        "seed_base": 0,
    }


def _object_ids(command: dict[str, Any]) -> list[str]:
    if command.get("object_ids"):
        return [str(item) for item in command["object_ids"]]
    if command.get("object_id"):
        return [str(command["object_id"])]
    if command.get("object"):
        return [str(command["object"])]
    return [DEFAULT_OBJECT]


def _target_xy(command: dict[str, Any]) -> np.ndarray:
    target = command.get("target_xy", DEFAULT_TARGET_XY)
    return np.asarray(target, dtype=np.float32).reshape(2)


def _annotation(command: dict[str, Any]) -> str:
    if command.get("annotation"):
        return str(command["annotation"])
    verb = str(command.get("verb", "move"))
    obj = str(command.get("object_id") or command.get("object") or DEFAULT_OBJECT)
    target = _target_xy(command)
    return f"{verb} {obj} to ({target[0]:.3f}, {target[1]:.3f})"


def _compile_command(command: dict[str, Any]) -> list[dict[str, Any]]:
    verb = str(command.get("verb", "move")).lower()
    if verb in {"move", "place", "pick_place", "pick-and-place"}:
        return [
            {"skill": "goto_station", "purpose": "pick", "target": "object"},
            {"skill": "pick", "object": "target"},
            {"skill": "carry", "to_station": "place"},
            {"skill": "place", "target_xy": _target_xy(command).tolist()},
        ]
    if verb == "push":
        return [{"skill": "push", "object": "target", "target_xy": _target_xy(command).tolist()}]
    raise ValueError(f"unsupported command verb {verb!r}")


try:
    from data_gen.aspire_engine.task_compiler import compile_command as _external_compile_command
except Exception:
    _external_compile_command = None


def _compiler_command(command: dict[str, Any]) -> dict[str, Any]:
    cmd = dict(command)
    if "object" not in cmd:
        cmd["object"] = cmd.get("object_id") or _object_ids(cmd)[0]
    if "goal" not in cmd and "target_xy" in cmd:
        cmd["goal"] = {"type": "zone", "xy": _target_xy(cmd).tolist()}
    return cmd


def _compile(command: dict[str, Any]) -> list[dict[str, Any]]:
    if _external_compile_command is not None:
        try:
            return list(_external_compile_command(_compiler_command(command)))
        except Exception:
            return _compile_command(command)
    return _compile_command(command)


def _make_env(command: dict[str, Any], seed: int, cfg: Any):
    object_ids = _object_ids(command)
    rng = np.random.default_rng(seed)
    if command.get("pick_xy") is not None:
        pick_xy = np.asarray(command["pick_xy"], dtype=np.float32).reshape(2)
    else:
        jitter = rng.uniform(-PICK_CENTER_JITTER, PICK_CENTER_JITTER).astype(np.float32)
        pick_xy = PICK_CENTER_XY + jitter
        pick_xy[0] = np.clip(pick_xy[0], PICK_X_RANGE[0], PICK_X_RANGE[1])
        pick_xy[1] = np.clip(pick_xy[1], PICK_Y_RANGE[0], PICK_Y_RANGE[1])

    render_mode = _cfg_get(cfg, "render_mode", None)
    kwargs: dict[str, Any] = {
        "num_envs": 1,
        "obs_mode": "state",
        "control_mode": "pd_joint_pos_fixed_base",
        "render_mode": render_mode,
        "reward_mode": "none",
        "sim_backend": "physx_cpu",
        "robot_uid": "aloha_mini_pro_v2",
        "base_xy": tuple(_cfg_get(cfg, "base_xy", (-0.40, 0.18))),
        "object_ids": object_ids,
        "slot_override_xy": [tuple(float(v) for v in pick_xy)],
        "object_xy_noise": float(_cfg_get(cfg, "object_xy_noise", 0.01)),
    }
    render_backend = _cfg_get(cfg, "render_backend", None)
    if render_backend is not None:
        kwargs["render_backend"] = render_backend
    if render_mode == "rgb_array":
        kwargs.setdefault("render_eye", [0.60, -1.05, 1.20])
        kwargs.setdefault("render_target", [-0.05, -0.30, 0.80])
    env = gym.make("AlohaMiniMultiYCB-v1", **kwargs)
    return env, pick_xy


def _note_failure(cfg: Any, runtime: Any = None, stage: str = "", extra: dict | None = None) -> None:
    """Fill cfg['failure_sink'] (if provided) so aspire_loop can diagnose the failure.

    ASPIRE-style: on any filtered exit the per-primitive traces + the failing stage
    are exposed to the agentic repair loop instead of being silently dropped.
    """
    sink = _cfg_get(cfg, "failure_sink", None)
    if not isinstance(sink, dict):
        return
    sink["stage"] = stage
    if extra:
        sink["extra"] = extra
    if runtime is not None:
        sink["traces"] = list(getattr(runtime, "traces", []))
        failed = next((t for t in reversed(sink["traces"]) if not t.get("ok", True)), None)
        sink["failed_trace"] = failed


def run_episode(command: dict[str, Any], seed: int, cfg: Any) -> dict[str, Any] | None:
    """Run one command. Return None on any filtered execution failure."""

    env = None
    command = dict(command)
    runtime = None
    try:
        env, pick_xy = _make_env(command, seed, cfg)
        env.reset(seed=seed)
        runtime = SkillRuntime(env, record=True)
        target_xy = _target_xy(command)
        object_ids = _object_ids(command)
        object_name = runtime.object_name(command.get("object_name") or object_ids[0])
        obj_actor = resolve_actor(env, object_name)
        obj0 = actor_position(obj_actor).copy()
        pitch_deg = float(command.get("pitch_deg", PITCH_DEFAULT))
        plan = _compile(command)
        verb = str(command.get("verb", "move")).lower()

        if verb in {"move", "place", "pick_place", "pick-and-place"}:
            # Both stations are feasibility-checked before picking, preserving the
            # validated invariant that place feasibility never teleports a held object.
            pick_trace = runtime.preselect_station(obj0, "pick", pitch_deg)
            if not pick_trace["ok"]:
                _note_failure(cfg, runtime, "preselect_pick")
                return None
            place_pt = np.array([target_xy[0], target_xy[1], obj0[2]], np.float32)
            place_trace = runtime.preselect_station(place_pt, "place", pitch_deg)
            if not place_trace["ok"]:
                _note_failure(cfg, runtime, "preselect_place")
                return None

        for step in plan:
            skill = str(step.get("skill"))
            if skill == "goto_station":
                purpose = str(step.get("purpose", "pick"))
                if purpose == "pick":
                    target = obj0
                else:
                    target = step.get("target_xy", target_xy)
                trace = runtime.goto_station(np.asarray(target, np.float32), purpose)
            elif skill == "pick":
                trace = runtime.pick(object_name, pitch_deg=pitch_deg)
            elif skill == "carry":
                trace = runtime.carry(step.get("to_station", step.get("to_station_purpose", "place")))
            elif skill == "place":
                trace = runtime.place(np.asarray(step.get("target_xy", target_xy), np.float32))
            elif skill == "push":
                trace = runtime.push(object_name, np.asarray(step.get("target_xy", target_xy), np.float32))
            else:
                trace = {"ok": False, "info": {"skill": skill}, "evidence": {"error": "unknown skill"}}
                runtime.traces.append(trace)
            if not trace.get("ok", False):
                _note_failure(cfg, runtime, f"skill:{skill}")
                return None

        objf = actor_position(obj_actor).copy()
        xy_err = float(np.linalg.norm(objf[:2] - target_xy))
        success = bool(xy_err < 0.06)
        if verb == "push":
            success = bool(xy_err <= 0.05)
        if not success:
            _note_failure(
                cfg,
                runtime,
                "final_check",
                {"xy_err": xy_err, "target_xy": target_xy.tolist(), "object_final": objf.tolist()},
            )
            return None

        return {
            "steps": runtime.steps,
            "annotation": _annotation(command),
            "command": {
                **command,
                "object_ids": object_ids,
                "target_xy": target_xy.tolist(),
                "sampled_pick_xy": pick_xy.tolist(),
            },
            "seed": int(seed),
            "success": True,
            "traces": runtime.traces,
            "final": {
                "object_name": object_name,
                "object_start": obj0.tolist(),
                "object_final": objf.tolist(),
                "target_xy": target_xy.tolist(),
                "xy_error": xy_err,
                "num_steps": len(runtime.steps),
            },
        }
    except Exception as exc:
        _note_failure(cfg, runtime, "exception", {"error": repr(exc)})
        return None
    finally:
        if env is not None:
            env.close()


def _worker(args: tuple[int, int, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    command_index, seed, command, cfg = args
    episode = run_episode(command, seed, cfg)
    return {
        "command_index": command_index,
        "seed": seed,
        "ok": episode is not None,
        "episode": episode,
    }


def generate_dataset(
    commands: list[dict[str, Any]],
    episodes_per_command: int,
    out_dir: str | Path,
    workers: int = 2,
) -> dict[str, Any]:
    cfg = _default_cfg(out_dir)
    tasks: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    seed_base = int(cfg["seed_base"])
    for command_index, command in enumerate(commands):
        command_seeds = command.get("seeds")
        for episode_index in range(int(episodes_per_command)):
            if command_seeds:
                seed = int(command_seeds[episode_index % len(command_seeds)])
            else:
                seed = seed_base + command_index * 1000 + episode_index
            tasks.append((command_index, seed, dict(command), dict(cfg)))

    if int(workers) <= 1:
        results = [_worker(task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(workers)) as pool:
            results = list(pool.map(_worker, tasks))

    successes = [result["episode"] for result in results if result["episode"] is not None]
    writer = AspireStateWriter(
        out_dir,
        fps=int(cfg["fps"]),
        dataset_name=str(cfg["dataset_name"]),
        robot_type=str(cfg["robot_type"]),
        overwrite=bool(cfg["overwrite"]),
    )
    writer_summary = writer.write_episodes(successes)
    return {
        "out_dir": str(out_dir),
        "results": results,
        "episodes": successes,
        "writer": writer_summary,
    }


def _listing(root: str | Path) -> list[str]:
    root = Path(root)
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{rel}{suffix}")
    return lines


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "output" / "smoke_state"
    commands = [
        {
            "verb": "move",
            "object_id": DEFAULT_OBJECT,
            "object_ids": [DEFAULT_OBJECT],
            "target_xy": DEFAULT_TARGET_XY.tolist(),
            "seeds": [3, 5],
            "annotation": "move the rubiks cube to the target spot",
        },
        {
            "verb": "move",
            "object_id": DEFAULT_OBJECT,
            "object_ids": [DEFAULT_OBJECT],
            "target_xy": DEFAULT_TARGET_XY.tolist(),
            "seeds": [3, 5],
            "annotation": "place the rubiks cube on the marked location",
        },
    ]
    print("SMOKE out_dir", out_dir, flush=True)
    print("SMOKE no_render render_mode=None render_backend=none", flush=True)
    summary = generate_dataset(commands, episodes_per_command=2, out_dir=out_dir, workers=1)
    ok_count = 0
    for result in summary["results"]:
        ok = bool(result["ok"])
        ok_count += int(ok)
        episode = result["episode"]
        steps = len(episode["steps"]) if episode is not None else 0
        status = "ok" if ok else "filtered"
        print(
            f"EP command={result['command_index']} seed={result['seed']} {status} steps={steps}",
            flush=True,
        )
    total = len(summary["results"])
    print(f"SUCCESS_RATE {ok_count}/{total}", flush=True)
    print("WRITER", json.dumps(summary["writer"], sort_keys=True), flush=True)
    print("OUT_DIR_LISTING_BEGIN", flush=True)
    for line in _listing(out_dir):
        print(line, flush=True)
    print("OUT_DIR_LISTING_END", flush=True)


if __name__ == "__main__":
    main()
