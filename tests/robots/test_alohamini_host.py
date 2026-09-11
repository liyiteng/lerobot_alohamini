#!/usr/bin/env python

import json
from types import SimpleNamespace

import pytest

from lerobot.motors import MotorNormMode
from lerobot.robots.alohamini import (
    alohamini as alohamini_module,
    alohamini_client as alohamini_client_module,
)
from lerobot.robots.alohamini.alohamini import AlohaMini
from lerobot.robots.alohamini.alohamini_client import AlohaMiniClient
from lerobot.robots.alohamini.alohamini_host import (
    build_observation_multipart,
    build_robot_metadata,
)
from lerobot.robots.alohamini.config_alohamini import AlohaMiniConfig
from lerobot.robots.alohamini.lift_axis import LiftAxis, LiftAxisConfig


def test_arm_motion_profile_config_rejects_unsafe_register_values() -> None:
    with pytest.raises(ValueError, match="arm_goal_velocity"):
        AlohaMiniConfig(arm_goal_velocity=0)
    with pytest.raises(ValueError, match="arm_acceleration"):
        AlohaMiniConfig(arm_acceleration=255)


def test_configure_applies_native_arm_motion_profile() -> None:
    class ConfigBus:
        def __init__(self) -> None:
            self.writes = []

        def disable_torque(self) -> None:
            pass

        def configure_motors(self) -> None:
            pass

        def write(self, register, motor, value, **kwargs) -> None:
            self.writes.append((register, motor, value, kwargs))

    bus = ConfigBus()
    robot = object.__new__(AlohaMini)
    robot.config = SimpleNamespace(arm_goal_velocity=2000, arm_acceleration=100)
    robot.left_bus = bus
    robot.right_bus = None
    robot.left_arm_motors = ["arm_left_shoulder_pan"]
    robot.base_motors = []

    robot.configure()

    assert (
        "Goal_Velocity",
        "arm_left_shoulder_pan",
        2000,
        {"normalize": False},
    ) in bus.writes
    assert (
        "Acceleration",
        "arm_left_shoulder_pan",
        100,
        {"normalize": False},
    ) in bus.writes


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


@pytest.mark.parametrize(
    ("include_cameras", "expected"),
    [(True, b"1:camera"), (False, b"1:state")],
)
def test_client_observation_token_selects_payload(include_cameras: bool, expected: bytes) -> None:
    sent = []
    client = object.__new__(AlohaMiniClient)
    client._zmq = SimpleNamespace(NOBLOCK=1, ZMQError=RuntimeError)
    client._observation_request_id = 0
    client._request_times = {}
    client._observation_request_tokens = []
    client.zmq_observation_socket = SimpleNamespace(send=lambda token, flags: sent.append((token, flags)))

    token = AlohaMiniClient._send_observation_request(client, include_cameras=include_cameras)

    assert token == expected
    assert sent == [(expected, 1)]


def test_full_observation_keeps_camera_multipart_compatibility(monkeypatch) -> None:
    encoded = SimpleNamespace(tobytes=lambda: b"jpeg-data")
    monkeypatch.setattr(
        "lerobot.robots.alohamini.alohamini_host.cv2.imencode",
        lambda *_args, **_kwargs: (True, encoded),
    )

    timings = {}
    parts = build_observation_multipart(
        {"x.vel": 0.0, "forward": object()},
        camera_keys=("forward",),
        encoding_timings_ms=timings,
    )

    assert len(parts) == 3
    assert json.loads(parts[0])["_images"] == ["forward"]
    assert parts[1:] == [b"forward", b"jpeg-data"]
    assert timings["encode_forward"] >= 0.0


def test_control_observation_peeks_camera_cache_without_waiting_for_new_frame() -> None:
    class FakeCamera:
        def __init__(self) -> None:
            self.max_age_ms = None
            self.latest_timestamp = 12.5

        def read_latest(self, max_age_ms: int):
            self.max_age_ms = max_age_ms
            return "cached-frame"

    class FakeBus:
        def sync_read(self, register, motors):
            if register == "Present_Velocity":
                return dict.fromkeys(motors, 0.0)
            return {}

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
    camera = FakeCamera()
    robot.cameras = {"forward": camera}
    robot.logs = {}

    observation = AlohaMini.get_observation.__wrapped__(robot, include_cameras=True)

    assert observation["forward"] == "cached-frame"
    assert camera.max_age_ms == 500
    assert observation["_host_timing"]["camera_capture_monotonic_s"] == {"forward": 12.5}


def test_robot_metadata_describes_normalization_and_calibration() -> None:
    motor = SimpleNamespace(
        id=1,
        model="sts3250",
        norm_mode=SimpleNamespace(value="range_m100_100"),
    )
    calibration = SimpleNamespace(drive_mode=1, range_min=100, range_max=3900)
    left_bus = SimpleNamespace(
        motors={"arm_left_shoulder_pan": motor},
        calibration={"arm_left_shoulder_pan": calibration},
    )
    lift = SimpleNamespace(cfg=SimpleNamespace(soft_min_mm=0.0, soft_max_mm=600.0, descent_floor_mm=5.0))
    robot = SimpleNamespace(
        left_bus=left_bus,
        right_bus=None,
        config=SimpleNamespace(robot_model="alohamini2pro"),
        lift=lift,
    )

    metadata = build_robot_metadata(robot)

    assert metadata["schema_version"] == 1
    assert metadata["robot_model"] == "alohamini2pro"
    assert metadata["motors"]["arm_left_shoulder_pan"] == {
        "id": 1,
        "model": "sts3250",
        "normalization": "range_m100_100",
        "drive_mode": 1,
        "range_min": 100,
        "range_max": 3900,
    }
    assert metadata["lift_axis"] == {
        "soft_min_mm": 0.0,
        "soft_max_mm": 600.0,
        "descent_floor_mm": 5.0,
    }


def test_client_lift_target_is_absolute_bounded_and_has_one_control_semantic(
    monkeypatch,
) -> None:
    client = object.__new__(AlohaMiniClient)
    client.teleop_keys = {"lift_up": "u", "lift_down": "j"}
    client.last_remote_state = {"lift_axis.height_mm": 100.0}
    client.latest_robot_metadata = {"lift_axis": {"soft_min_mm": 0.0, "soft_max_mm": 600.0}}
    client.config = SimpleNamespace(
        lift_target_speed_mm_s=150.0,
        lift_target_max_lead_mm=5.0,
    )
    client._lift_target_mm = None
    client._lift_last_update_t = None
    client._lift_direction = 0
    times = iter((1.0, 1.02, 1.04))
    monkeypatch.setattr(alohamini_client_module.time, "monotonic", lambda: next(times))

    first = client._from_keyboard_to_lift_action({"u"})
    second = client._from_keyboard_to_lift_action({"u"})
    released = client._from_keyboard_to_lift_action(set())

    assert first == {"lift_axis.height_mm": pytest.approx(103.0)}
    assert second == {"lift_axis.height_mm": pytest.approx(105.0)}
    assert released == {"lift_axis.height_mm": pytest.approx(100.0)}
    assert "lift_axis.vel" not in released


def test_lift_axis_clamps_absolute_target_and_prioritizes_position() -> None:
    writes = []
    bus = SimpleNamespace(
        motors={"lift_axis": object()},
        write=lambda register, motor, value: writes.append((register, motor, value)),
    )
    lift = LiftAxis(
        LiftAxisConfig(soft_min_mm=0.0, soft_max_mm=600.0, dir_sign=-1),
        bus_left=bus,
        bus_right=None,
    )

    accepted = lift.apply_action(
        {"lift_axis.height_mm": 700.0, "lift_axis.vel": 0.0},
        current_height_mm=100.0,
    )

    assert accepted == {"lift_axis.height_mm": 600.0}
    assert writes == [("Goal_Velocity", "lift_axis", -1300)]


def test_lift_axis_returns_empty_mapping_without_lift_command() -> None:
    bus = SimpleNamespace(motors={"lift_axis": object()})
    lift = LiftAxis(LiftAxisConfig(), bus_left=bus, bus_right=None)

    assert lift.apply_action({"arm_left_shoulder_pan.pos": 12.0}) == {}


class FakeBus:
    def __init__(self, *, current_raw: float = 0.0, motor_model: str = "sts3095") -> None:
        self.motors = {
            "arm_left_elbow_flex": SimpleNamespace(model=motor_model, norm_mode=MotorNormMode.DEGREES)
        }
        self.current_raw = current_raw
        self.reads: list[str] = []
        self.writes = []
        self.model_resolution_table = {motor_model: 4096}
        self.calibration = {"arm_left_elbow_flex": SimpleNamespace(range_min=0, range_max=4095)}

    def sync_read(self, register: str, motors: list[str]) -> dict[str, float]:
        self.reads.append(register)
        if register == "Present_Current":
            return dict.fromkeys(motors, self.current_raw)
        return dict.fromkeys(motors, 1.0)

    def sync_write(self, register, values, **kwargs):
        self.writes.append((register, dict(values)))


def make_robot_feedback_stub(bus: FakeBus) -> AlohaMini:
    robot = object.__new__(AlohaMini)
    robot.left_bus = bus
    robot.right_bus = None
    robot._initialize_current_protection()
    robot._feedback_currents_raw = {"arm_left_elbow_flex": 0.0}
    robot._feedback_positions = {"arm_left_elbow_flex": 1.0}
    robot._joint_release_margin = 1.0
    robot._feedback_lift_height_mm = None
    robot.config = SimpleNamespace(max_relative_target=None)
    robot.lift = SimpleNamespace(apply_action=lambda *_args, **_kwargs: {})
    robot._body_to_wheel_raw = lambda *_args: {}
    robot.logs = {}
    return robot


def test_current_limiter_reuses_observe_act_feedback() -> None:
    bus = FakeBus()
    robot = make_robot_feedback_stub(bus)

    result = robot._limit_joint_goal_by_current(bus, {"arm_left_elbow_flex.pos": 2.0})

    assert result == {"arm_left_elbow_flex.pos": 2.0}
    assert bus.reads == []


def test_current_limiter_keeps_read_through_fallback() -> None:
    bus = FakeBus()
    robot = make_robot_feedback_stub(bus)
    robot._feedback_currents_raw.clear()
    robot._feedback_positions.clear()

    result = robot._limit_joint_goal_by_current(bus, {"arm_left_elbow_flex.pos": 2.0})

    assert result == {"arm_left_elbow_flex.pos": 2.0}
    assert bus.reads == ["Present_Current", "Present_Position"]


def test_joint_current_limiter_uses_elapsed_time(monkeypatch) -> None:
    bus = FakeBus()
    robot = make_robot_feedback_stub(bus)
    robot._feedback_currents_raw["arm_left_elbow_flex"] = 600.0
    goal = {"arm_left_elbow_flex.pos": 10.0}
    times = iter((1.0, 1.149, 1.151))
    monkeypatch.setattr(alohamini_module.time, "monotonic", lambda: next(times))

    first = robot._limit_joint_goal_by_current(bus, goal)
    second = robot._limit_joint_goal_by_current(bus, goal)
    held = robot._limit_joint_goal_by_current(bus, goal)

    assert first == goal
    assert second == goal
    assert held == {"arm_left_elbow_flex.pos": 1.0}


def test_joint_current_limiter_does_not_hold_a_moving_joint(monkeypatch) -> None:
    bus = FakeBus()
    robot = make_robot_feedback_stub(bus)
    robot._feedback_currents_raw["arm_left_elbow_flex"] = 600.0
    goal = {"arm_left_elbow_flex.pos": 10.0}
    times = iter((1.0, 1.160, 1.320))
    monkeypatch.setattr(alohamini_module.time, "monotonic", lambda: next(times))

    assert robot._limit_joint_goal_by_current(bus, goal) == goal
    robot._feedback_positions["arm_left_elbow_flex"] = 2.0
    assert robot._limit_joint_goal_by_current(bus, goal) == goal
    robot._feedback_positions["arm_left_elbow_flex"] = 3.0
    assert robot._limit_joint_goal_by_current(bus, goal) == goal
    assert robot._joint_hold_goal == {}


def test_joint_current_limiter_ignores_small_command_error(monkeypatch) -> None:
    bus = FakeBus()
    robot = make_robot_feedback_stub(bus)
    robot._feedback_currents_raw["arm_left_elbow_flex"] = 600.0
    monkeypatch.setattr(alohamini_module.time, "monotonic", lambda: 1.0)

    goal = {"arm_left_elbow_flex.pos": 2.0}
    assert robot._limit_joint_goal_by_current(bus, goal) == goal
    assert robot._joint_stall_candidates == {}


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("last_current", [0.0, 10.0])
def test_current_units_do_not_depend_on_last_motor(raw, last_current) -> None:
    bus = FakeBus()
    bus.motors["lift_axis"] = SimpleNamespace(model="sts3095")
    values = {"arm_left_elbow_flex": 100.0, "lift_axis": last_current}
    bus.sync_read = lambda *_args: dict(values)
    robot = make_robot_feedback_stub(bus)

    result = robot.read_and_check_currents(raw=raw)

    assert result == {name: value * (1.0 if raw else 6.5) for name, value in values.items()}


def test_zero_current_on_other_bus_does_not_inflate_gripper_current() -> None:
    robot, bus = make_gripper_feedback_stub(present=30.0, current_raw=20.0)
    bus.motors["arm_left_gripper"] = SimpleNamespace(model="sts3250")
    bus.current_raw = 20.0
    robot.left_bus = bus
    robot.right_bus = FakeBus(current_raw=0.0)
    robot._initialize_current_protection()
    robot._feedback_currents_raw = robot.read_and_check_currents(raw=True)

    goal = {"arm_left_gripper.pos": 0.0}
    assert robot._limit_gripper_goal_by_current(bus, goal) == goal
    assert robot._gripper_hold_goal == {}


@pytest.mark.parametrize("frequency_hz", [30, 50])
@pytest.mark.parametrize(
    "norm_mode,calibration_span",
    [
        (MotorNormMode.DEGREES, 4095),
        (MotorNormMode.RANGE_M100_100, 1024),
        (MotorNormMode.RANGE_M100_100, 4095),
    ],
)
def test_slow_motion_uses_physical_degrees(monkeypatch, frequency_hz, norm_mode, calibration_span) -> None:
    bus = FakeBus()
    motor = "arm_left_elbow_flex"
    bus.motors[motor].norm_mode = norm_mode
    bus.calibration[motor].range_max = calibration_span
    units_per_degree = 1.0 if norm_mode is MotorNormMode.DEGREES else 200 * 4095 / (calibration_span * 360)
    robot = make_robot_feedback_stub(bus)
    robot._feedback_currents_raw[motor] = 600.0
    goal = {motor + ".pos": 10.0 * units_per_degree}
    for sample in range(frequency_hz // 2 + 1):
        now = sample / frequency_hz
        monkeypatch.setattr(alohamini_module.time, "monotonic", lambda now=now: now)
        robot._feedback_positions[motor] = (1.0 + 2.0 * now) * units_per_degree
        assert robot._limit_joint_goal_by_current(bus, goal) == goal
    assert robot._joint_hold_goal == {}


@pytest.mark.parametrize("frequency_hz", [30, 50])
def test_idle_host_cycles_hold_stalled_joint_once_and_allow_retreat(monkeypatch, frequency_hz) -> None:
    bus = FakeBus(current_raw=600.0)
    robot = make_robot_feedback_stub(bus)
    motor = "arm_left_elbow_flex"
    key = motor + ".pos"
    now = 0.0
    monkeypatch.setattr(alohamini_module.time, "monotonic", lambda: now)
    robot._feedback_currents_raw = robot.read_and_check_currents(raw=True)
    assert AlohaMini.send_action.__wrapped__(robot, {key: 10.0})[key] == 10.0
    bus.writes.clear()

    trip_time = None
    for sample in range(1, frequency_hz // 2):
        now = sample / frequency_hz
        robot._feedback_positions = {motor: 1.0}
        robot._feedback_currents_raw = robot.read_and_check_currents(raw=True)
        corrections = robot.supervise_arm_motion()
        if corrections:
            assert corrections == {key: 1.0}
            assert trip_time is None
            trip_time = now
    assert 0.150 <= trip_time < 0.150 + 1 / frequency_hz
    assert bus.writes == [("Goal_Position", {motor: 1.0})]

    bus.current_raw = 0.0
    robot._feedback_currents_raw = robot.read_and_check_currents(raw=True)
    assert robot.supervise_arm_motion() == {}
    assert robot._joint_hold_goal == {motor: 1.0}
    assert AlohaMini.send_action.__wrapped__(robot, {key: 0.0})[key] == 0.0
    assert robot._joint_hold_goal == {}


def test_normal_observation_clears_pending_stall(monkeypatch) -> None:
    bus = FakeBus(current_raw=600.0)
    robot = make_robot_feedback_stub(bus)
    motor = "arm_left_elbow_flex"
    now = 0.0
    monkeypatch.setattr(alohamini_module.time, "monotonic", lambda: now)
    goal = {motor + ".pos": 10.0}
    robot._feedback_currents_raw = robot.read_and_check_currents(raw=True)
    robot._limit_joint_goal_by_current(bus, goal)
    assert motor in robot._joint_stall_candidates

    now = 0.1
    bus.current_raw = 0.0
    robot.read_and_check_currents(raw=True)
    assert robot._joint_stall_candidates == {}
    now = 0.2
    bus.current_raw = 600.0
    robot._feedback_currents_raw = robot.read_and_check_currents(raw=True)
    assert robot._limit_joint_goal_by_current(bus, goal) == goal


def test_target_reversal_restarts_stall_window(monkeypatch) -> None:
    bus = FakeBus()
    robot = make_robot_feedback_stub(bus)
    motor = "arm_left_elbow_flex"
    robot._feedback_currents_raw[motor] = 600.0
    times = iter((0.0, 0.14, 0.16))
    monkeypatch.setattr(alohamini_module.time, "monotonic", lambda: next(times))
    robot._limit_joint_goal_by_current(bus, {motor + ".pos": 10.0})
    retreat = {motor + ".pos": -10.0}
    assert robot._limit_joint_goal_by_current(bus, retreat) == retreat
    assert robot._limit_joint_goal_by_current(bus, retreat) == retreat


def test_encoder_jitter_does_not_release_a_stalled_joint(monkeypatch) -> None:
    bus = FakeBus()
    robot = make_robot_feedback_stub(bus)
    motor = "arm_left_elbow_flex"
    robot._feedback_currents_raw[motor] = 600.0
    goal = {motor + ".pos": 10.0}
    for sample in range(10):
        now = sample / 50
        monkeypatch.setattr(alohamini_module.time, "monotonic", lambda now=now: now)
        robot._feedback_positions[motor] = 1.0 + (0.08 if sample % 2 else 0.0)
        result = robot._limit_joint_goal_by_current(bus, goal)
    assert motor in robot._joint_hold_goal
    assert result[motor + ".pos"] == robot._joint_hold_goal[motor]


def test_partial_command_keeps_other_active_arm_targets(monkeypatch) -> None:
    bus = FakeBus()
    robot = make_robot_feedback_stub(bus)
    key = "arm_left_elbow_flex.pos"
    monkeypatch.setattr(alohamini_module.time, "monotonic", lambda: 0.0)
    AlohaMini.send_action.__wrapped__(robot, {key: 10.0})
    result = AlohaMini.send_action.__wrapped__(robot, {"x.vel": 0.1})
    assert result[key] == 10.0
    assert robot._arm_goal_positions == {key: 10.0}


@pytest.mark.parametrize("gripper", [False, True])
def test_feedback_failure_preserves_existing_hold(gripper) -> None:
    if gripper:
        robot, bus = make_gripper_feedback_stub(present=1.0, current_raw=0.0)
        motor = "arm_left_gripper"
        robot._gripper_hold_goal[motor] = 1.0
        robot._gripper_hold_direction[motor] = -1.0
        limiter = robot._limit_gripper_goal_by_current
    else:
        bus = FakeBus()
        robot = make_robot_feedback_stub(bus)
        motor = "arm_left_elbow_flex"
        robot._joint_hold_goal[motor] = 1.0
        robot._joint_hold_direction[motor] = -1.0
        limiter = robot._limit_joint_goal_by_current

    def fail_read(*_args):
        raise OSError("serial read failed")

    robot._feedback_positions.clear()
    robot._feedback_currents_raw.clear()
    bus.sync_read = fail_read
    assert limiter(bus, {motor + ".pos": 10.0}) == {motor + ".pos": 1.0}


@pytest.mark.parametrize("frequency_hz", [30, 50])
def test_overcurrent_duration_is_independent_of_sample_frequency(frequency_hz: int) -> None:
    duration_s = 0.650
    started_at = {}
    trip_time = None

    for sample in range(frequency_hz * 2):
        now = sample / frequency_hz
        if alohamini_module._has_sustained_overcurrent(
            started_at,
            "arm_left_elbow_flex",
            current_ma=4500.0,
            limit_ma=4400.0,
            now=now,
            duration_s=duration_s,
        ):
            trip_time = now
            break

    assert trip_time is not None
    assert duration_s <= trip_time < duration_s + 1 / frequency_hz


def test_normal_current_clears_pending_overcurrent_duration() -> None:
    started_at = {}

    assert not alohamini_module._has_sustained_overcurrent(started_at, "motor", 3000.0, 2000.0, 0.0, 0.150)
    assert not alohamini_module._has_sustained_overcurrent(started_at, "motor", 1000.0, 2000.0, 0.100, 0.150)
    assert started_at == {}
    assert not alohamini_module._has_sustained_overcurrent(started_at, "motor", 3000.0, 2000.0, 0.200, 0.150)


@pytest.mark.parametrize(
    ("motor_model", "expected"),
    [
        ("sts3095", (3300.0, 4400.0, 7840.0)),
        ("sts3250", (2100.0, 2800.0, 3360.0)),
        ("sts3215", (1350.0, 1800.0, 2160.0)),
    ],
)
def test_current_limits_follow_motor_ratings(motor_model: str, expected: tuple[float, float, float]) -> None:
    motor = SimpleNamespace(model=motor_model)

    limits = alohamini_module._current_limits_for_motor(motor)
    assert (limits.collision_ma, limits.sustained_ma, limits.near_stall_ma) == expected


def test_current_limits_reject_unknown_motor_model() -> None:
    with pytest.raises(ValueError, match="Missing current ratings"):
        alohamini_module._current_limits_for_motor(SimpleNamespace(model="unknown"))


@pytest.mark.parametrize(("current_raw", "duration"), [(-700.0, 0.650), (-1300.0, 0.080)])
def test_overcurrent_stops_and_disconnects_after_elapsed_time(monkeypatch, current_raw, duration) -> None:
    bus = FakeBus(current_raw=current_raw)
    robot = make_robot_feedback_stub(bus)
    events = []
    robot.stop_motion = lambda: events.append("stop")
    robot.disconnect = lambda: events.append("disconnect")
    times = iter((1.0, 1.0 + duration - 0.001, 1.0 + duration + 0.001))
    monkeypatch.setattr(alohamini_module.time, "monotonic", lambda: next(times))

    assert robot.read_and_check_currents(raw=True)["arm_left_elbow_flex"] == current_raw
    assert robot.read_and_check_currents(raw=True)["arm_left_elbow_flex"] == current_raw
    assert events == []
    with pytest.raises(SystemExit, match="1"):
        robot.read_and_check_currents(raw=True)
    assert events == ["stop", "disconnect"]


def make_gripper_feedback_stub(*, present: float, current_raw: float) -> tuple[AlohaMini, FakeBus]:
    bus = FakeBus()
    bus.motors = {"arm_left_gripper": SimpleNamespace(model="sts3250", norm_mode=MotorNormMode.RANGE_0_100)}
    robot = object.__new__(AlohaMini)
    robot.left_bus = bus
    robot.right_bus = None
    robot._initialize_current_protection()
    robot._feedback_currents_raw = {"arm_left_gripper": current_raw}
    robot._feedback_positions = {"arm_left_gripper": present}
    robot._gripper_current_limit_ma = 500.0
    robot._gripper_release_margin = 1.0
    robot._gripper_hold_close_step = 3.0
    robot._gripper_open_direction = {"arm_left_gripper": 1.0}
    robot._gripper_hold_goal = {}
    robot._gripper_hold_direction = {}
    return robot, bus


def test_gripper_open_endpoint_overcurrent_releases_on_close_command() -> None:
    robot, bus = make_gripper_feedback_stub(present=90.0, current_raw=100.0)

    held = robot._limit_gripper_goal_by_current(bus, {"arm_left_gripper.pos": 100.0})
    assert held["arm_left_gripper.pos"] == pytest.approx(90.0)

    robot._feedback_currents_raw["arm_left_gripper"] = 0.0
    closing = robot._limit_gripper_goal_by_current(bus, {"arm_left_gripper.pos": 0.0})
    assert closing["arm_left_gripper.pos"] == pytest.approx(0.0)
    assert robot._gripper_hold_goal == {}


def test_gripper_closing_contact_retains_squeeze_and_releases_on_open() -> None:
    robot, bus = make_gripper_feedback_stub(present=30.0, current_raw=100.0)

    held = robot._limit_gripper_goal_by_current(bus, {"arm_left_gripper.pos": 0.0})
    assert held["arm_left_gripper.pos"] == pytest.approx(27.0)

    robot._feedback_currents_raw["arm_left_gripper"] = 0.0
    opening = robot._limit_gripper_goal_by_current(bus, {"arm_left_gripper.pos": 100.0})
    assert opening["arm_left_gripper.pos"] == pytest.approx(100.0)
    assert robot._gripper_hold_goal == {}
