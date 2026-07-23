"""Unit tests for the ASPIRE repair loop (mocked engine — no simulation).

Validates the full agentic path deterministically: failure sink -> symptom text ->
library match -> parameterized repair -> retry -> success -> validated-repair
admission to the repairs log.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ASPIRE_ENGINE_DIR = Path(__file__).resolve().parents[1]
MANISKILL_ROOT = ASPIRE_ENGINE_DIR.parents[1]
sys.path.insert(0, str(MANISKILL_ROOT))


def _fake_run_episode_factory(fail_specs):
    """run_episode stub: consult fail_specs per call; fill the sink like the engine."""
    calls = []

    def fake(command, seed, cfg):
        idx = len(calls)
        calls.append({"command": dict(command), "seed": seed})
        if idx < len(fail_specs):
            sink = cfg.get("failure_sink")
            if isinstance(sink, dict):
                sink.update(fail_specs[idx])
            return None
        return {
            "steps": [], "annotation": "ok", "command": dict(command),
            "seed": seed, "success": True, "traces": [],
            "final": {"xy_error": 0.01},
        }

    fake.calls = calls
    return fake


def test_repair_path_and_admission():
    import data_gen.aspire_engine.aspire_loop as loop

    # attempt 0 fails at pick with palm-press evidence (matches tilted_approach)
    fail = {
        "stage": "skill:pick",
        "failed_trace": {
            "ok": False,
            "info": {"skill": "pick"},
            "evidence": {"error": "object pushed away during descent, palm contact impulse"},
        },
        "traces": [],
    }
    fake = _fake_run_episode_factory([fail])
    orig = loop.run_episode
    loop.run_episode = fake
    try:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "repairs.jsonl"
            cmd = {"verb": "move", "object": "077_rubiks_cube",
                   "target_xy": [0.06, -0.31], "pitch_deg": 90.0}
            episode, report = loop.run_with_repairs(
                cmd, seed=3, cfg={}, max_attempts=3, log_path=log, verbose=False)
            assert episode is not None, "loop should heal the mocked failure"
            assert len(report["attempts"]) == 2, report["attempts"]
            repair = report["attempts"][0]["next_repair"]
            assert repair is not None
            # library must have matched a motion-primitive entry and mutated pitch
            assert "pitch_deg" in repair.get("command", {}), repair
            assert repair["command"]["pitch_deg"] != 90.0
            # retry must have used the repaired command
            assert fake.calls[1]["command"]["pitch_deg"] == repair["command"]["pitch_deg"]
            # validated repair admitted to the log
            rec = json.loads(log.read_text().strip().splitlines()[-1])
            assert rec["validated"] is True
            assert rec["failed_stage"] == "skill:pick"
    finally:
        loop.run_episode = orig
    print("PASSED test_repair_path_and_admission")


def test_exhaustion_returns_none():
    import data_gen.aspire_engine.aspire_loop as loop

    fail = {"stage": "preselect_pick",
            "failed_trace": {"ok": False, "info": {"skill": "preselect_station"},
                             "evidence": {"error": "no physically-valid station"}},
            "traces": []}
    fake = _fake_run_episode_factory([fail] * 10)   # never succeeds
    orig = loop.run_episode
    loop.run_episode = fake
    try:
        episode, report = loop.run_with_repairs(
            {"verb": "move", "object": "077_rubiks_cube", "target_xy": [0, -0.45]},
            seed=1, cfg={}, max_attempts=3, verbose=False)
        assert episode is None
        assert report["success"] is False
        assert len(report["attempts"]) == 3
    finally:
        loop.run_episode = orig
    print("PASSED test_exhaustion_returns_none")


if __name__ == "__main__":
    failures = 0
    for fn in (test_repair_path_and_admission, test_exhaustion_returns_none):
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAILED {fn.__name__}: {exc}")
    print(f"PASSED {2 - failures} FAILED {failures}")
