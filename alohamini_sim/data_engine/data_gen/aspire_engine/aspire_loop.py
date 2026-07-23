"""ASPIRE agentic repair loop: diagnose a filtered episode from its traces, match the
failure against the skill library, apply a parameterized repair, and retry.

This closes the loop that was previously manual (coordinator-in-the-loop): the same
trace -> library-match -> repair -> validate cycle now runs automatically, and every
validated (failure-signature -> repair) pair is admitted to a repairs log — ASPIRE
§2.2's "only validated repairs are admitted" rule.

Usage:
    from data_gen.aspire_engine.aspire_loop import run_with_repairs
    episode, report = run_with_repairs(command, seed, cfg, max_attempts=4)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from data_gen.aspire_engine.engine import run_episode

# the skill library loader lives next to the intern_engine skills
from data_gen.intern_engine.skills.library.library import match as library_match

DEFAULT_LOG = Path(__file__).parent / "output" / "repairs_log.jsonl"


# ---------------------------------------------------------------------------
# Repair actions: library-entry name -> ordered list of command/cfg mutations.
# Each action is a dict: {"command": {...updates}, "cfg": {...updates}, "why": str}.
# Mutations are cumulative per attempt (applied to a fresh copy of the originals).
# ---------------------------------------------------------------------------
def _pitch_actions() -> list[dict[str, Any]]:
    return [
        {"command": {"pitch_deg": 65.0}, "why": "tilted_approach: steeper pitch clears palm/reach"},
        {"command": {"pitch_deg": 55.0}, "why": "tilted_approach: shallower pitch"},
        {"command": {"pitch_deg": 70.0}, "why": "tilted_approach: steepest validated pitch"},
    ]


def _station_actions() -> list[dict[str, Any]]:
    return [
        {"cfg": {"base_xy": (-0.35, 0.22)}, "why": "base_reposition: different start -> new station draw"},
        {"cfg": {"base_xy": (-0.50, 0.14)}, "why": "base_reposition: alternate start"},
    ]


def _push_actions(command: dict[str, Any]) -> list[dict[str, Any]]:
    """Push-specific: shorten the push (target closer to the object's start)."""
    out = []
    tgt = command.get("target_xy") or (command.get("goal") or {}).get("xy")
    pick = command.get("pick_xy")
    if tgt is not None:
        tgt = np.asarray(tgt, np.float64)
        # without the live object pos, contract toward the sampled/nominal pick point
        anchor = np.asarray(pick if pick is not None else [-0.13, -0.31], np.float64)
        for frac in (0.7, 0.5):
            new = anchor + frac * (tgt - anchor)
            out.append({
                "command": {"target_xy": new.tolist(),
                            "goal": {"type": "zone", "xy": new.tolist()}},
                "why": f"linear_push: contract push distance to {frac:.0%}",
            })
    return out


def _generic_actions(seed: int) -> list[dict[str, Any]]:
    return [
        {"seed": seed + 101, "why": "generic: reroll domain randomization"},
        {"cfg": {"object_xy_noise": 0.0}, "why": "generic: disable object noise"},
    ]


def propose_repairs(command: dict[str, Any], seed: int, sink: dict[str, Any]) -> list[dict[str, Any]]:
    """Rank repair actions for a failure: library-matched actions first, generic last."""
    stage = str(sink.get("stage", ""))
    failed = sink.get("failed_trace") or {}
    info = failed.get("info", {})
    evidence = failed.get("evidence", {})
    # symptom text for the library's keyword matcher
    symptoms = " ".join([
        f"stage {stage}", f"skill {info.get('skill', '')}",
        " ".join(f"{k} {v}" for k, v in list(evidence.items())[:8]),
    ])
    matches = library_match(symptoms, top_k=4)
    matched_names = [m["name"] for m in matches]

    actions: list[dict[str, Any]] = []
    skill = str(info.get("skill", stage))
    if "push" in skill or "push" in stage:
        actions += _push_actions(command)
    for name in matched_names:
        if name in ("tilted_approach", "desc_first_ik_branch", "cartesian_line_descent"):
            actions += _pitch_actions()
        elif name in ("base_reposition", "station_relative_approach_dir",
                      "station_physics_validation"):
            actions += _station_actions()
    if stage in ("preselect_pick", "preselect_place") and not actions:
        actions += _station_actions() + _pitch_actions()
    actions += _generic_actions(seed)

    # dedupe by repr, keep order
    seen: set[str] = set()
    unique = []
    for a in actions:
        key = json.dumps({k: a.get(k) for k in ("command", "cfg", "seed")}, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(a)
    for a in unique:
        a["matched_library"] = matched_names
    return unique


def run_with_repairs(command: dict[str, Any], seed: int, cfg: Any,
                     max_attempts: int = 4,
                     log_path: str | Path = DEFAULT_LOG,
                     verbose: bool = True) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Execute a command; on filtered failure, diagnose -> match library -> repair -> retry.

    Returns (episode_or_None, report). The report lists every attempt with its applied
    repair and outcome; validated repairs are appended to the repairs log (JSONL).
    """
    base_cfg = dict(cfg) if isinstance(cfg, dict) else dict(getattr(cfg, "__dict__", {}) or {})
    report: dict[str, Any] = {"command": command, "seed": seed, "attempts": []}
    pending: list[dict[str, Any]] | None = None
    cur_cmd, cur_cfg, cur_seed = dict(command), dict(base_cfg), int(seed)

    for attempt in range(max_attempts):
        sink: dict[str, Any] = {}
        cur_cfg["failure_sink"] = sink
        t0 = time.time()
        episode = run_episode(cur_cmd, cur_seed, cur_cfg)
        entry: dict[str, Any] = {
            "attempt": attempt,
            "seed": cur_seed,
            "duration_s": round(time.time() - t0, 1),
            "ok": episode is not None,
            "stage": sink.get("stage"),
            "repair": report["attempts"][-1].get("next_repair") if report["attempts"] else None,
        }
        report["attempts"].append(entry)
        if episode is not None:
            if verbose:
                print(f"[LOOP] attempt {attempt} OK"
                      + (f" (repair: {entry['repair']['why']})" if entry.get("repair") else ""),
                      flush=True)
            if entry.get("repair"):
                _admit(log_path, cur_cmd, sinks=report["attempts"], repair=entry["repair"])
            report["success"] = True
            return episode, report

        failed = sink.get("failed_trace") or {}
        if verbose:
            print(f"[LOOP] attempt {attempt} FAILED at stage={sink.get('stage')} "
                  f"skill={failed.get('info', {}).get('skill')}", flush=True)
        if pending is None:
            pending = propose_repairs(cur_cmd, cur_seed, sink)
            if verbose and pending:
                print(f"[LOOP] library matched: {pending[0].get('matched_library')} — "
                      f"{len(pending)} repair candidates", flush=True)
        if not pending:
            break
        repair = pending.pop(0)
        # apply to fresh copies of the ORIGINALS (repairs don't stack across attempts)
        cur_cmd, cur_cfg, cur_seed = dict(command), dict(base_cfg), int(seed)
        cur_cmd.update(repair.get("command", {}))
        cur_cfg.update(repair.get("cfg", {}))
        cur_seed = int(repair.get("seed", cur_seed))
        report["attempts"][-1]["next_repair"] = repair
        if verbose:
            print(f"[LOOP] applying repair: {repair['why']}", flush=True)

    report["success"] = False
    return None, report


def _admit(log_path: str | Path, command: dict[str, Any],
           sinks: list[dict[str, Any]], repair: dict[str, Any]) -> None:
    """Append a VALIDATED (failure -> repair) pair to the repairs log."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "validated": True,
        "command_verb": command.get("verb"),
        "failed_stage": next((a.get("stage") for a in sinks if not a.get("ok")), None),
        "repair": {k: repair.get(k) for k in ("command", "cfg", "seed", "why", "matched_library")},
        "attempts": len(sinks),
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    # Regression case: the push command that the plain engine FILTERED.
    from data_gen.aspire_engine.engine import _default_cfg

    cmd = {"verb": "push", "object": "077_rubiks_cube",
           "goal": {"type": "zone", "xy": [-0.13, -0.42]},
           "target_xy": [-0.13, -0.42]}
    episode, report = run_with_repairs(cmd, seed=7, cfg=_default_cfg(), max_attempts=4)
    print("FINAL:", "SUCCESS" if episode else "EXHAUSTED",
          "after", len(report["attempts"]), "attempts")
    if episode:
        print("annotation:", episode.get("annotation"),
              "xy_error:", episode["final"]["xy_error"])
