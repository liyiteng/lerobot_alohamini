#!/usr/bin/env python

import json
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.alohamini.alohamini import AlohaMini
from lerobot.robots.alohamini.alohamini_host import (
    build_observation_multipart,
    build_robot_metadata,
    observation_request_includes_cameras,
)
from lerobot.robots.alohamini.camera_stream import (
    CameraSnapshot,
    encode_camera_stream_message,
    read_camera_snapshot,
)
from lerobot.robots.alohamini.config_alohamini import (
    AlohaMiniHostConfig,
    alohamini_cameras_config,
)


def test_50hz_branch_keeps_original_camera_defaults_and_ros_stream_opt_in() -> None:
    assert set(alohamini_cameras_config()) == {"forward", "wrist_right"}
    config = AlohaMiniHostConfig()
    assert config.max_loop_freq_hz == 50
    assert config.camera_stream_enabled is False


@pytest.mark.parametrize(
    ("token", "expected"),
    [(None, False), (b"1", True), (b"1:camera", True), (b"ros-1:state", False)],
)
def test_request_suffix_is_backward_compatible(token, expected) -> None:
    assert observation_request_includes_cameras(token) is expected


def test_state_only_observation_has_no_jpeg_frames() -> None:
    parts = build_observation_multipart(
        {
            "arm_left_shoulder_pan.pos": 1.0,
            "forward": object(),
            "_robot_metadata": {"schema_version": 1},
        },
        camera_keys=("forward",),
        encoded_camera_keys=(),
    )

    assert len(parts) == 1
    state = json.loads(parts[0])
    assert state["_images"] == []
    assert state["_image_encoding"] == "jpeg"
    assert "forward" not in state
    assert state["_robot_metadata"] == {"schema_version": 1}


def test_full_observation_keeps_legacy_camera_multipart(monkeypatch) -> None:
    encoded = SimpleNamespace(tobytes=lambda: b"jpeg-data")
    monkeypatch.setattr(
        "lerobot.robots.alohamini.alohamini_host.cv2.imencode",
        lambda *_args, **_kwargs: (True, encoded),
    )

    parts = build_observation_multipart(
        {"x.vel": 0.0, "forward": object()},
        camera_keys=("forward",),
    )

    assert len(parts) == 3
    assert json.loads(parts[0])["_images"] == ["forward"]
    assert parts[1:] == [b"forward", b"jpeg-data"]


def test_camera_stream_protocol_carries_capture_time_and_rgb_color(monkeypatch) -> None:
    encoded_inputs = []
    encoded = SimpleNamespace(tobytes=lambda: b"jpeg-data")
    monkeypatch.setattr(
        "lerobot.robots.alohamini.camera_stream.cv2.imencode",
        lambda _extension, frame, _options: (
            encoded_inputs.append(frame.copy()) or True,
            encoded,
        ),
    )
    rgb_frame = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb_frame[0, 0] = [250, 20, 10]

    parts, encode_ms = encode_camera_stream_message(
        "forward",
        CameraSnapshot(frame=rgb_frame, capture_monotonic_s=8.0),
        sequence=4,
        jpeg_quality=70,
        host_monotonic_s=10.0,
        host_unix_ns=20_000_000_000,
    )

    metadata = json.loads(parts[1])
    assert parts[0] == b"camera/forward"
    assert parts[2] == b"jpeg-data"
    assert metadata["schema_version"] == 1
    assert metadata["capture_unix_ns"] == 18_000_000_000
    assert encode_ms >= 0.0
    assert encoded_inputs[0][0, 0].tolist() == [10, 20, 250]


def test_camera_stream_snapshot_keeps_frame_and_timestamp_atomic() -> None:
    class FakeCamera:
        def __init__(self) -> None:
            import threading

            self.frame_lock = threading.Lock()
            self.latest_frame = "new-frame"
            self.latest_timestamp = 2.0

        def read_latest(self, max_age_ms):
            assert max_age_ms == 500
            return "old-frame"

    assert read_camera_snapshot(FakeCamera(), 500) == CameraSnapshot("new-frame", 2.0)


def test_state_only_robot_observation_skips_camera_and_has_wall_timestamp() -> None:
    class FakeBus:
        def sync_read(self, register, motors):
            if register == "Present_Velocity":
                return dict.fromkeys(motors, 0.0)
            return {}

    class FailCamera:
        def async_read(self):
            raise AssertionError("state-only observation must not read a camera")

    robot = object.__new__(AlohaMini)
    robot.id = "test"
    robot.left_bus = FakeBus()
    robot.right_bus = None
    robot.left_arm_motors = []
    robot.right_arm_motors = []
    robot.base_motors = ["base_left_wheel", "base_back_wheel", "base_right_wheel"]
    robot._wheel_raw_to_body = lambda *_args: {
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
    }
    robot.lift = SimpleNamespace(contribute_observation=lambda _obs: None)
    robot.read_and_check_currents = lambda **_kwargs: {}
    robot.cameras = {"forward": FailCamera()}
    robot.logs = {}

    observation = AlohaMini.get_observation.__wrapped__(robot, include_cameras=False)

    assert "forward" not in observation
    assert observation["_host_timing"]["state_sample_unix_ns"] > 0
    assert observation["_host_timing"]["camera_capture_monotonic_s"] == {}


def test_robot_metadata_describes_normalization_and_lift_limits() -> None:
    motor = SimpleNamespace(
        id=1,
        model="sts3250",
        norm_mode=SimpleNamespace(value="range_m100_100"),
    )
    calibration = SimpleNamespace(drive_mode=1, range_min=100, range_max=3900)
    robot = SimpleNamespace(
        left_bus=SimpleNamespace(
            motors={"arm_left_shoulder_pan": motor},
            calibration={"arm_left_shoulder_pan": calibration},
        ),
        right_bus=None,
        config=SimpleNamespace(robot_model="alohamini2pro"),
        lift=SimpleNamespace(
            cfg=SimpleNamespace(
                soft_min_mm=0.0,
                soft_max_mm=600.0,
                descent_floor_mm=5.0,
            )
        ),
    )

    metadata = build_robot_metadata(robot)

    assert metadata["schema_version"] == 1
    assert metadata["robot_model"] == "alohamini2pro"
    assert metadata["motors"]["arm_left_shoulder_pan"]["normalization"] == "range_m100_100"
    assert metadata["lift_axis"] == {
        "soft_min_mm": 0.0,
        "soft_max_mm": 600.0,
        "descent_floor_mm": 5.0,
    }
