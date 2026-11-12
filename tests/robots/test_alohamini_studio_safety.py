"""采集检查和停止协议的离线回归，不连接真实硬件。"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import zmq

from lerobot.robots.alohamini.command_owner import CommandOwner
from lerobot.robots.alohamini.software_stop import SoftwareStop

spec = importlib.util.spec_from_file_location(
    "studio_safety", Path(__file__).parents[2] / "examples/alohamini/studio_safety.py"
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def observation():
    return {
        "_robot_metadata": {"robot_model": "alohamini2pro", "cameras": ["forward", "wrist_right"]},
        "_safety": {"host_session_id": "host", "software_stop_supported": True},
        "_host_timing": {
            "host_clock_reference": {"monotonic_s": 10.0},
            "camera_capture_monotonic_s": {"forward": 9.9, "wrist_right": 9.9},
        },
    }


def test_network_requires_matching_model_session_and_unlocked_host():
    tool.validate_host(observation(), "alohamini2pro")
    for update in [
        {"_robot_metadata": {"robot_model": "alohamini2"}},
        {"_safety": {}},
        {"_safety": {"host_session_id": "host", "software_stop": {"latched": True}}},
        {"_safety": {"host_session_id": "host", "control_owner": "other-client"}},
    ]:
        with pytest.raises(ValueError):
            tool.validate_host({**observation(), **update}, "alohamini2pro")


@pytest.mark.parametrize("problem", [None, "missing", "corrupt", "stale", "no_timestamp", "duplicate"])
def test_cameras_require_two_fresh_robot_images(problem):
    data = observation()
    jpeg = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))[1].tobytes()
    frames = [b"forward", jpeg, b"wrist_right", jpeg]
    names = ["forward", "wrist_right"]
    if problem == "missing": frames = frames[:2]
    if problem == "corrupt": frames[-1] = b"invalid"
    if problem == "stale": data["_host_timing"]["camera_capture_monotonic_s"]["forward"] = 1
    if problem == "no_timestamp": data["_host_timing"] = {}
    if problem == "duplicate": names = ["forward", "forward"]
    if problem:
        with pytest.raises(ValueError): tool.validate_cameras(data, frames, names)
    else:
        tool.validate_cameras(data, frames, names)


def test_calibration_checks_local_leaders_and_live_follower_metadata(tmp_path):
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]
    calibration = {name: {"range_min": 200, "range_max": 3800} for name in joints}
    directory = tmp_path / "teleoperators/so_leader"
    directory.mkdir(parents=True)
    for side in ("left", "right"):
        (directory / f"so101_leader_bi_{side}.json").write_text(json.dumps(calibration))
    data = observation()
    motors = {f"arm_{side}_{name}": value for side in ("left", "right") for name, value in calibration.items()}
    data["_robot_metadata"]["motors"] = motors
    data.update({f"{name}.pos": 0.0 for name in motors})
    tool.validate_calibration(data, calibration_root=tmp_path)
    data["_robot_metadata"]["motors"] = {}
    with pytest.raises(ValueError): tool.validate_calibration(data, calibration_root=tmp_path)
    (directory / "so101_leader_bi_left.json").write_text("invalid")
    with pytest.raises(ValueError): tool.validate_calibration(data, calibration_root=tmp_path)


@pytest.mark.parametrize("token", [b"1:state", b"studio-stop:old:" + b"a" * 32 + b":state", b"studio-stop:host:invalid:state"])
def test_stop_ignores_ordinary_or_wrong_session_requests(token):
    stop = SoftwareStop()
    stop.request(token, "host")
    assert not stop.latched


@pytest.mark.parametrize("fails", [False, True])
def test_stop_holds_measured_positions_and_only_confirms_success(fails):
    stop, robot, owner = SoftwareStop(), Mock(), CommandOwner()
    owner.owner = "inference"
    stop.request(b"studio-stop:host:" + b"a" * 32 + b":state", "host")
    if fails: robot.send_action.side_effect = RuntimeError("电机写入失败")
    stop.apply(robot, {"joint.pos": 3.0, "x.vel": 1.0}, owner)
    robot.send_action.assert_called_once_with({"joint.pos": 3.0, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
    robot.stop_motion.assert_called_once()
    assert stop.latched
    assert stop.applied is not fails
    assert (owner.owner is None) is not fails
    if fails:
        robot.send_action.side_effect = None
        stop.apply(robot, {"joint.pos": 3.0}, owner)
        assert stop.applied


def test_invalid_feedback_still_stops_base_without_confirming():
    stop, robot = SoftwareStop(), Mock()
    stop.latched = True
    stop.apply(robot, {"joint.pos": float("nan")}, CommandOwner())
    assert not stop.applied
    robot.stop_motion.assert_called_once()
    robot.send_action.assert_not_called()


@pytest.mark.parametrize("problem", [None, "legacy", "wrong_id", "restart", "not_applied"])
def test_stop_client_requires_matching_execution_ack(monkeypatch, problem):
    data = observation()
    socket = Mock()
    if problem == "legacy": data["_safety"].pop("software_stop_supported")
    def request(_socket, token):
        request_id = token.decode().split(":")[2]
        return {"_safety": {
            "host_session_id": "restarted" if problem == "restart" else "host",
            "software_stop": {"latched": True, "applied": problem != "not_applied", "request_id": "wrong" if problem == "wrong_id" else request_id},
        }}, []
    monkeypatch.setattr(tool, "request_observation", request)
    assert tool.stop_robot(socket, data)["success"] is (problem is None)


def test_request_ignores_unmatched_reply_and_times_out():
    socket = Mock()
    socket.poll.side_effect = [True, False]
    socket.recv_multipart.return_value = [b"old", b"{}"]
    with pytest.raises(TimeoutError): tool.request_observation(socket, b"current", 0.01)


def test_host_stop_blocks_concurrent_and_subsequent_motion(monkeypatch):
    from lerobot.robots.alohamini import alohamini_host

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(alohamini_host.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(alohamini_host.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(alohamini_host.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
    monkeypatch.setattr("sys.argv", ["alohamini_host"])
    robot = Mock(cameras={}, logs={}, _feedback_currents_raw={})
    robot.action_features = {"joint.pos": float}
    robot.get_observation.return_value = {"joint.pos": 1.0}
    robot.get_safety_status.return_value = {"host_session_id": "host"}
    robot.send_action.side_effect = lambda action: action
    host = Mock(max_loop_freq_hz=50, connection_time_s=0.055, watchdog_timeout_ms=1000)
    host.zmq_cmd_socket.recv_string.return_value = json.dumps({"joint.pos": 20.0})
    token = b"studio-stop:host:" + b"a" * 32 + b":state"
    host.zmq_observation_socket.recv_multipart.side_effect = [[b"client", token], [b"client", b"2:state"], [b"client", b"3:state"]]
    monkeypatch.setattr(alohamini_host, "AlohaMini", lambda _: robot)
    monkeypatch.setattr(alohamini_host, "AlohaMiniHost", lambda _: host)
    monkeypatch.setattr(alohamini_host, "build_robot_metadata", lambda _: {})
    alohamini_host.main()
    robot.send_action.assert_called_once_with({"joint.pos": 1.0, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
    responses = [json.loads(call.args[0][2])["_safety"] for call in host.zmq_observation_socket.send_multipart.call_args_list]
    assert all(response["software_stop"]["applied"] for response in responses)
    assert all(response["control_owner"] is None for response in responses)


@pytest.mark.parametrize("problem", [None, "missing", "conversion", "movement"])
def test_leader_alignment_rejects_incomplete_targets_and_propagates_thread_failures(problem):
    from lerobot.teleoperators.bi_so_leader.bi_so_leader import BiSOLeader

    leader = object.__new__(BiSOLeader)
    for side in ("left", "right"):
        bus = SimpleNamespace(calibration={"joint": object()}, motors={"joint": SimpleNamespace(id=1)},
                              _unnormalize=lambda values: values, _id_to_name=lambda _: "joint")
        arm = SimpleNamespace(is_connected=True, bus=bus, smooth_move_to_raw=Mock())
        setattr(leader, f"{side}_arm", arm)
    targets = {"arm_left_joint.pos": 1.0, "arm_right_joint.pos": 2.0}
    if problem == "missing": targets.pop("arm_right_joint.pos")
    if problem == "conversion": leader.right_arm.bus._unnormalize = Mock(side_effect=ValueError("标定转换失败"))
    if problem == "movement": leader.right_arm.smooth_move_to_raw.side_effect = RuntimeError("串口写入失败")
    if problem:
        with pytest.raises((ValueError, RuntimeError)):
            leader.sync_to_pose(targets)
        if problem != "movement":
            leader.left_arm.smooth_move_to_raw.assert_not_called()
            leader.right_arm.smooth_move_to_raw.assert_not_called()
    else:
        leader.sync_to_pose(targets)
        assert leader.left_arm.smooth_move_to_raw.call_args.kwargs["target_raw"] == {"joint": 1}
        assert leader.right_arm.smooth_move_to_raw.call_args.kwargs["target_raw"] == {"joint": 2}


@pytest.mark.parametrize("problem", ["yield", "current_read", "write"])
def test_strict_alignment_failure_releases_leader_torque(monkeypatch, problem):
    from lerobot.teleoperators.so_leader.so_leader import SOLeader

    leader = object.__new__(SOLeader)
    leader.id = "test"
    leader.bus = Mock(motors={"joint": object()})
    def read(register, *_args, **_kwargs):
        if register == "Present_Current" and problem == "current_read":
            raise RuntimeError("电流反馈失败")
        return {"joint": 66 if register == "Present_Current" else 100}
    leader.bus.sync_read.side_effect = read
    if problem == "write": leader.bus.write.side_effect = RuntimeError("写入失败")
    monkeypatch.setattr("lerobot.teleoperators.so_leader.so_leader.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError):
        leader.smooth_move_to_raw({"joint": 100}, duration_s=0.01, hold_s=0, fail_on_yield=True)
    leader.bus.disable_torque.assert_called_once()
