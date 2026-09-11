import json
import time
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from examples.alohamini.safety_utils import (
    EvaluationSafetyGuard,
    SafetyRecorder,
    preserve_dataset,
    stop_inference,
)
from lerobot.rollout.inference.rtc import RTCInferenceEngine
from tests.robots.test_alohamini_command_safety import status


def test_sidecar_frame_indices_survive_missing_status_and_events(tmp_path):
    dataset = SimpleNamespace(root=tmp_path, meta=SimpleNamespace(total_episodes=2))
    snapshot = status(accepted_targets={"joint.pos": 1.0})
    with SafetyRecorder(dataset) as recorder:
        recorder.write(safety={})
        recorder.write(safety=snapshot, event={"type": "capture_wait"})
        recorder.write(safety=snapshot, requested_action={"joint.pos": 9.0})
        snapshot["accepted_targets"]["joint.pos"] = 100.0
    path = tmp_path / "meta/safety/episode_000002.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["frame_index"] for row in rows] == [0, None, 1, None]
    assert rows[2]["requested_action"] == {"joint.pos": 9.0}
    assert rows[2]["safety"]["accepted_targets"] == {"joint.pos": 1.0}
    with SafetyRecorder(dataset) as recorder:
        recorder.write(safety=status())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["frame_index"] == 0


@pytest.mark.parametrize("exception", [KeyboardInterrupt, RuntimeError])
def test_interrupted_recording_saves_partial_episode_and_finalizes(exception):
    dataset = Mock()
    dataset.has_pending_frames.return_value = True
    with pytest.raises(exception), preserve_dataset(dataset):
        raise exception()
    dataset.save_episode.assert_called_once_with()
    dataset.finalize.assert_called_once_with()


def test_finalize_still_runs_when_save_fails():
    dataset = Mock()
    dataset.save_episode.side_effect = OSError("disk full")
    with pytest.raises(OSError), preserve_dataset(dataset):
        pass
    dataset.finalize.assert_called_once_with()


@pytest.mark.parametrize(
    "change",
    [
        {"joint_holds": {"elbow": 1.0}},
        {"watchdog_active": True},
        {"joint_hold_events": 1},
        {"watchdog_events": 1},
        {"host_session_id": "host-2"},
    ],
)
def test_eval_pauses_for_active_or_transient_protection(change):
    guard = EvaluationSafetyGuard()
    robot = SimpleNamespace(latest_safety_status=status(), _last_safety_received_at=time.monotonic())
    assert guard.reason(robot) is None
    robot.latest_safety_status.update(change)
    assert guard.reason(robot) is not None


def test_eval_allows_gripper_contact_but_not_stale_feedback():
    guard = EvaluationSafetyGuard()
    robot = SimpleNamespace(
        latest_safety_status=status(gripper_holds={"gripper": 5.0}),
        _last_safety_received_at=time.monotonic(),
    )
    assert guard.reason(robot) is None
    robot._last_safety_received_at -= 2.0
    assert guard.reason(robot) == "Host 反馈中断"


def test_eval_rechecks_feedback_against_host_watchdog_after_slow_inference():
    robot = SimpleNamespace(
        latest_safety_status=status(command_watchdog_timeout_s=0.5),
        _last_safety_received_at=time.monotonic() - 0.6,
    )
    assert EvaluationSafetyGuard().reason(robot) == "Host 反馈中断"


def test_eval_manual_recovery_discards_old_inference_before_resume(monkeypatch):
    robot = SimpleNamespace(
        latest_safety_status=status(watchdog_active=True),
        _last_safety_received_at=time.monotonic(),
        observation_sequence=0,
        last_sent_command={},
    )
    robot.feedback_fresh = True
    robot.command_permitted = True

    def send_action(_action):
        robot.last_sent_command = {"client_id": "pc", "sequence": 1}

    def get_observation():
        robot.observation_sequence += 1
        robot.latest_safety_status = status(command=robot.last_sent_command.copy())
        robot._last_safety_received_at = time.monotonic()
        return {"joint.pos": 1.0}

    robot.send_action = send_action
    robot.get_observation = get_observation
    robot.refresh_observation = get_observation
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    engine = Mock()
    engine._rtc_thread = None
    guard = EvaluationSafetyGuard()
    guard.recover(robot, engine, Mock(), Mock(), {"joint.pos": 1.0}, "test")
    assert [call[0] for call in engine.mock_calls] == ["pause", "stop", "reset", "start", "resume"]
    assert guard.reason(robot) is None


def test_rtc_stop_timeout_keeps_thread_reference():
    engine = object.__new__(RTCInferenceEngine)
    engine._shutdown_event = Event()
    engine._policy_active = Event()
    thread = Mock()
    thread.is_alive.return_value = True
    engine._rtc_thread = thread
    with pytest.raises(RuntimeError, match="still running"):
        stop_inference(engine)
    assert engine._rtc_thread is thread
    assert engine._shutdown_event.is_set()


def test_partial_episode_is_readable_after_interrupt(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "dataset"
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
        "action": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
    }
    dataset = LeRobotDataset.create(
        repo_id="local/recovery", fps=10, root=root, features=features, use_videos=False
    )
    with pytest.raises(KeyboardInterrupt), preserve_dataset(dataset):
        for index in range(2):
            dataset.add_frame(
                {
                    "observation.state": np.array([index], dtype=np.float32),
                    "action": np.array([index + 1], dtype=np.float32),
                    "task": "recovery",
                }
            )
        raise KeyboardInterrupt()
    restored = LeRobotDataset(repo_id="local/recovery", root=root)
    assert len(restored) == 2
    assert restored.num_episodes == 1
    assert restored[1]["action"].item() == 2.0
