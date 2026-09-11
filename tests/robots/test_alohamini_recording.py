#!/usr/bin/env python

import importlib
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.alohamini import record_utils_multirate
from examples.alohamini.record_utils_multirate import (
    ControlSample,
    ControlSampleBuffer,
    FreshCameraGate,
)
from lerobot.robots.alohamini.alohamini_client import AlohaMiniClient


def test_control_sample_buffer_selects_more_than_one_tick_of_history() -> None:
    samples = ControlSampleBuffer(max_samples=10)
    for index, timestamp in enumerate((1.00, 1.02, 1.04, 1.06)):
        samples.append(
            ControlSample(
                observation={"joint": index},
                action={"arm_joint.pos": index},
                state_timestamp=timestamp,
            )
        )

    selected, error_s = samples.nearest(1.001, max_error_s=0.1)

    assert selected.observation == {"joint": 0}
    assert selected.action == {"arm_joint.pos": 0}
    assert error_s == pytest.approx(0.001)


def test_control_sample_buffer_rejects_excessive_alignment_error() -> None:
    samples = ControlSampleBuffer(max_samples=2)
    samples.append(ControlSample({}, {}, 1.0))

    with pytest.raises(RuntimeError, match="alignment exceeded"):
        samples.nearest(1.2, max_error_s=0.05)


def test_camera_gate_requires_every_camera_to_advance() -> None:
    gate = FreshCameraGate(
        ("forward", "wrist_right"),
        started_at=0.0,
        stall_timeout_s=1.0,
        max_skew_s=0.05,
    )

    assert gate.observe({"forward": 1.00, "wrist_right": 1.02}, now=0.1) == pytest.approx(1.01)
    assert gate.observe({"forward": 1.04, "wrist_right": 1.02}, now=0.2) is None
    assert gate.observe({"forward": 1.04, "wrist_right": 1.05}, now=0.3) == pytest.approx(1.045)


def test_camera_gate_reports_stall_and_excessive_skew() -> None:
    stalled = FreshCameraGate(
        ("forward", "wrist_right"),
        started_at=0.0,
        stall_timeout_s=0.5,
        max_skew_s=0.05,
    )
    stalled.observe({"forward": 1.0}, now=0.1)
    with pytest.raises(RuntimeError, match="wrist_right"):
        stalled.observe({}, now=0.6)

    skewed = FreshCameraGate(
        ("forward", "wrist_right"),
        started_at=0.0,
        stall_timeout_s=1.0,
        max_skew_s=0.05,
    )
    with pytest.raises(RuntimeError, match="capture skew"):
        skewed.observe({"forward": 1.0, "wrist_right": 1.2}, now=0.1)


def test_client_primes_request_window_with_one_payload_kind() -> None:
    client = object.__new__(AlohaMiniClient)
    client._observation_request_tokens = deque((b"1:camera", b"2:camera"))
    client.polling_timeout_ms = 200
    client.latest_host_timing = {"old": True}
    drained = []
    refilled = []
    client._receive_observation_response = lambda token, timeout_ms: drained.append((token, timeout_ms))
    client._fill_observation_request_window = lambda *, include_cameras: refilled.append(include_cameras)

    AlohaMiniClient.prime_observation_request_window(client, include_cameras=False)

    assert drained == [(b"1:camera", 200), (b"2:camera", 200)]
    assert refilled == [False]
    assert client.latest_host_timing == {}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(float(seconds), 1e-6)


class FakeDataset:
    def __init__(self, fps: int, *, with_camera: bool, root: Path) -> None:
        self.fps = fps
        self.root = root
        self.meta = SimpleNamespace(total_episodes=0)
        self.frames = []
        self.features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["joint"],
            },
            "action": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["arm_joint.pos"],
            },
        }
        if with_camera:
            self.features["observation.images.forward"] = {
                "dtype": "image",
                "shape": (2, 2, 3),
            }

    def add_frame(self, frame) -> None:
        self.frames.append(frame)


class FakeRobot:
    def __init__(self, clock: FakeClock, *, with_camera: bool, duplicate_once: bool = False) -> None:
        self.clock = clock
        self._cameras_ft = {"forward": (2, 2, 3)} if with_camera else {}
        self.duplicate_once = duplicate_once
        self.camera_requests = 0
        self.last_camera_index = 0
        self.last_camera_timestamp = 0.0
        self.latest_host_timing = {}
        self.logs = {}
        self.sent_actions = []
        self.primed = []

    def prime_observation_request_window(self, *, include_cameras: bool) -> None:
        self.primed.append(include_cameras)

    def get_observation(self, *, include_cameras: bool = True):
        timestamp = self.clock.now
        camera_timestamps = {}
        if include_cameras and self._cameras_ft:
            self.camera_requests += 1
            if not self.duplicate_once or self.camera_requests == 1 or self.camera_requests % 2 == 1:
                self.last_camera_index += 1
                self.last_camera_timestamp = timestamp
            camera_timestamps["forward"] = self.last_camera_timestamp
        self.latest_host_timing = {
            "state_sample_started_monotonic_s": timestamp,
            "state_sample_finished_monotonic_s": timestamp,
            "camera_capture_monotonic_s": camera_timestamps,
        }
        observation = {"joint": timestamp}
        if self._cameras_ft:
            observation["forward"] = np.full((2, 2, 3), self.last_camera_index, dtype=np.uint8)
        return observation

    def _from_keyboard_to_base_action(self, _keys):
        return {}

    def _from_keyboard_to_lift_action(self, _keys):
        return {}

    def send_action(self, action) -> None:
        self.sent_actions.append(action)


def run_fake_recording(
    monkeypatch,
    tmp_path,
    *,
    with_camera: bool,
    duplicate_once: bool = False,
    dataset_with_camera: bool | None = None,
    camera_gap: bool = False,
    stop_during_gap: bool = False,
) -> tuple[FakeRobot, FakeDataset, FakeClock]:
    clock = FakeClock()
    robot = FakeRobot(clock, with_camera=with_camera, duplicate_once=duplicate_once)
    dataset = FakeDataset(
        10,
        with_camera=with_camera if dataset_with_camera is None else dataset_with_camera,
        root=tmp_path,
    )
    events = {"exit_early": False}
    if camera_gap:
        get_observation = robot.get_observation

        def interrupted_camera(**kwargs):
            obs = get_observation(**kwargs)
            if 0.05 < clock.now < 0.8:
                robot.latest_host_timing["camera_capture_monotonic_s"] = {"forward": 0.0}
            if stop_during_gap and clock.now > 0.7:
                events["exit_early"] = True
            return obs

        robot.get_observation = interrupted_camera
    monkeypatch.setattr(record_utils_multirate.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(record_utils_multirate, "precise_sleep", clock.sleep)

    def identity(value):
        return value

    record_utils_multirate.record_loop(
        robot=robot,
        events=events,
        fps=10,
        leader_arm=SimpleNamespace(get_action=lambda: {"joint.pos": clock.now}),
        keyboard=SimpleNamespace(get_action=lambda: set()),
        teleop_action_processor=lambda value: identity(value[0]),
        robot_action_processor=lambda value: identity(value[0]),
        robot_observation_processor=identity,
        dataset=dataset,
        control_time_s=0.2,
        single_task="test",
        control_fps=50,
        camera_stall_timeout_s=0.5,
        min_capture_rate_ratio=0.5,
    )
    return robot, dataset, clock


def test_default_recorder_keeps_the_original_single_rate_loop(monkeypatch) -> None:
    examples_dir = Path(__file__).parents[2] / "examples" / "alohamini"
    monkeypatch.syspath_prepend(str(examples_dir))
    record_bi = importlib.import_module("record_bi")
    default_record_utils = importlib.import_module("record_utils")

    assert record_bi.record_loop is default_record_utils.record_loop


def test_record_loop_records_state_only_dataset_at_dataset_rate(monkeypatch, tmp_path) -> None:
    robot, dataset, _clock = run_fake_recording(monkeypatch, tmp_path, with_camera=False)

    assert len(dataset.frames) == 2
    assert all("observation.state" in frame for frame in dataset.frames)
    assert len(robot.sent_actions) > len(dataset.frames)
    assert robot.primed == [False]


def test_record_loop_uses_dataset_features_not_connected_cameras(monkeypatch, tmp_path) -> None:
    robot, dataset, _clock = run_fake_recording(
        monkeypatch,
        tmp_path,
        with_camera=True,
        dataset_with_camera=False,
    )

    assert len(dataset.frames) == 2
    assert robot.camera_requests == 0


def test_record_loop_skips_duplicate_images_but_reaches_target_frame_count(monkeypatch, tmp_path) -> None:
    robot, dataset, clock = run_fake_recording(
        monkeypatch,
        tmp_path,
        with_camera=True,
        duplicate_once=True,
    )

    assert len(dataset.frames) == 2
    image_values = [int(frame["observation.images.forward"][0, 0, 0]) for frame in dataset.frames]
    assert image_values == [1, 2]
    assert robot.camera_requests >= 3
    assert clock.now > 0.2


def test_camera_gap_keeps_control_running_then_completes_episode(monkeypatch, tmp_path):
    robot, dataset, clock = run_fake_recording(monkeypatch, tmp_path, with_camera=True, camera_gap=True)

    assert len(dataset.frames) == 2
    assert clock.now >= 0.8
    assert len(robot.sent_actions) >= 40
    metadata = (tmp_path / "meta/safety/episode_000000.jsonl").read_text()
    assert "capture_wait" in metadata
    assert "capture_recovered" in metadata


def test_operator_can_end_stalled_camera_episode_without_losing_frames(monkeypatch, tmp_path):
    robot, dataset, clock = run_fake_recording(
        monkeypatch,
        tmp_path,
        with_camera=True,
        camera_gap=True,
        stop_during_gap=True,
    )

    assert len(dataset.frames) == 1
    assert clock.now < 0.8
    assert len(robot.sent_actions) > 30


def test_default_recorder_continues_through_joint_protection(monkeypatch, tmp_path):
    from examples.alohamini import record_utils

    clock = FakeClock()
    robot = FakeRobot(clock, with_camera=False)
    robot.latest_safety_status = {"version": 1, "joint_holds": {"elbow": 1.0}}
    dataset = FakeDataset(10, with_camera=False, root=tmp_path)
    monkeypatch.setattr(record_utils.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(record_utils, "precise_sleep", clock.sleep)
    record_utils.record_loop(
        robot=robot,
        events={"exit_early": False},
        fps=10,
        leader_arm=SimpleNamespace(get_action=lambda: {"joint.pos": clock.now}),
        keyboard=SimpleNamespace(get_action=lambda: set()),
        teleop_action_processor=lambda value: value[0],
        robot_action_processor=lambda value: value[0],
        robot_observation_processor=lambda value: value,
        dataset=dataset,
        control_time_s=0.2,
        single_task="test",
    )
    assert len(dataset.frames) == 2
    assert len(robot.sent_actions) == 2
    assert '"joint_holds": {"elbow": 1.0}' in (tmp_path / "meta/safety/episode_000000.jsonl").read_text()


@pytest.mark.parametrize("multirate", [False, True])
@pytest.mark.parametrize("stop_early", [False, True])
def test_recorders_wait_through_lost_feedback_without_commands_or_cached_frames(
    monkeypatch,
    tmp_path,
    multirate,
    stop_early,
):
    from examples.alohamini import record_utils

    module = record_utils_multirate if multirate else record_utils
    clock = FakeClock()
    robot = FakeRobot(clock, with_camera=True)
    robot.observation_sequence = 0
    get_observation = robot.get_observation
    cached = None
    events = {"exit_early": False}

    def interrupted_observation(**kwargs):
        nonlocal cached
        if clock.now < 0.05 or clock.now >= 0.6:
            cached = get_observation(**kwargs)
            robot.observation_sequence += 1
        elif stop_early and clock.now >= 0.3:
            events["exit_early"] = True
        return cached

    robot.get_observation = interrupted_observation
    dataset = FakeDataset(10, with_camera=True, root=tmp_path)
    monkeypatch.setattr(module.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(module, "precise_sleep", clock.sleep)
    module.record_loop(
        robot=robot,
        events=events,
        fps=10,
        dataset=dataset,
        control_time_s=0.2,
        single_task="test",
        leader_arm=SimpleNamespace(get_action=lambda: {"joint.pos": clock.now}),
        keyboard=SimpleNamespace(get_action=lambda: set()),
        teleop_action_processor=lambda value: value[0],
        robot_action_processor=lambda value: value[0],
        robot_observation_processor=lambda value: value,
    )
    assert len(dataset.frames) == (1 if stop_early else 2)
    assert all(not 0.05 <= action["arm_joint.pos"] < 0.6 for action in robot.sent_actions)
    metadata = (tmp_path / "meta/safety/episode_000000.jsonl").read_text()
    assert "feedback_wait" in metadata
    if not stop_early:
        assert "feedback_recovered" in metadata
        assert dataset.frames[-1]["observation.state"][0] >= 0.6


def test_default_recorder_waits_for_new_images(monkeypatch, tmp_path):
    from examples.alohamini import record_utils

    clock = FakeClock()
    robot = FakeRobot(clock, with_camera=True, duplicate_once=True)
    dataset = FakeDataset(10, with_camera=True, root=tmp_path)
    monkeypatch.setattr(record_utils.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(record_utils, "precise_sleep", clock.sleep)
    record_utils.record_loop(
        robot=robot,
        events={"exit_early": False},
        fps=10,
        dataset=dataset,
        control_time_s=0.2,
        single_task="test",
        leader_arm=SimpleNamespace(get_action=lambda: {"joint.pos": clock.now}),
        keyboard=SimpleNamespace(get_action=lambda: set()),
        teleop_action_processor=lambda value: value[0],
        robot_action_processor=lambda value: value[0],
        robot_observation_processor=lambda value: value,
    )
    assert [int(frame["observation.images.forward"][0, 0, 0]) for frame in dataset.frames] == [1, 2]
    assert len(robot.sent_actions) == 3
