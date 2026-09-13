import json
import runpy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import zmq

from examples.alohamini import teleop_monitor


@pytest.fixture
def robot():
    return SimpleNamespace(
        feedback_fresh=True,
        command_permitted=True,
        latest_safety_status={
            "requested_targets": {"arm_left_gripper.pos": 0.54},
            "accepted_targets": {
                "arm_left_gripper.pos": 0.54,
                "arm_left_elbow_flex.pos": 3.0,
                "arm_left_shoulder_pan.pos": 2.0,
            },
            "currents_ma": {"arm_left_gripper": 58.5},
            "joint_holds": {},
        },
        latest_robot_metadata={
            "motors": {
                "arm_left_gripper": {"normalization": "range_0_100"},
                "arm_left_elbow_flex": {"normalization": "range_m100_100"},
                "arm_left_shoulder_pan": {"normalization": "range_m100_100"},
            }
        },
        last_remote_state={
            "arm_left_gripper.pos": 1.73,
            "arm_left_elbow_flex.pos": 1.0,
            "arm_left_shoulder_pan.pos": 1.0,
        },
        get_observation=Mock(),
        send_action=Mock(),
    )


def test_monitor_paces_reports_and_keeps_gripper_units_separate(robot, monkeypatch, capsys):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(teleop_monitor.time, "perf_counter", lambda: clock.now)
    monitor = teleop_monitor.TeleopMonitor(robot)
    for i in range(1, 50):
        clock.now = i / 50
        monitor.update(sent=True)
    assert not capsys.readouterr().out
    clock.now = 1.0
    monitor.update(sent=False)
    output = capsys.readouterr().out
    assert "loop_hz=50.0 sent_hz=49.0" in output
    assert "state=active" in output
    assert "[arm_left_gripper] target=0.54 command=0.54 measured=1.73 gap=-1.19" in output
    assert "current=+58.5mA" in output
    assert "[arm_left_elbow_flex]" in output
    assert "[arm_left_shoulder_pan]" not in output
    assert len(output.splitlines()) == 3
    robot.get_observation.assert_not_called()
    robot.send_action.assert_not_called()


@pytest.mark.parametrize(
    "fresh,permitted,expected", [(False, True, "waiting_feedback"), (True, False, "control_unavailable")]
)
def test_monitor_reports_waiting_without_claiming_commands_sent(
    robot, monkeypatch, capsys, fresh, permitted, expected
):
    robot.feedback_fresh = fresh
    robot.command_permitted = permitted
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(teleop_monitor.time, "perf_counter", lambda: clock.now)
    monitor = teleop_monitor.TeleopMonitor(robot)
    clock.now = 1.0
    monitor.update(sent=False)
    output = capsys.readouterr().out
    assert "sent_hz=0.0" in output
    assert f"state={expected}" in output
    if not fresh:
        assert "TRACKING" not in output
        assert "joint_holds=n/a" in output


def test_old_host_without_tracking_metadata_is_supported(robot, monkeypatch, capsys):
    robot.latest_safety_status = {}
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(teleop_monitor.time, "perf_counter", lambda: clock.now)
    monitor = teleop_monitor.TeleopMonitor(robot)
    clock.now = 1.0
    monitor.update(sent=True)
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_old_host_currents_are_not_invented(robot):
    robot.latest_safety_status.pop("currents_ma")
    assert all("current=n/a" in row[3] for row in teleop_monitor.tracking_rows(robot))


def test_nonfinite_feedback_is_not_reported_as_tracking(robot):
    robot.last_remote_state["arm_left_gripper.pos"] = float("nan")
    assert all(row[0] != "arm_left_gripper" for row in teleop_monitor.tracking_rows(robot))


@pytest.mark.parametrize("mode", ["active", "waiting_feedback", "waiting_send"])
def test_actual_teleop_entry_reports_each_control_path(robot, monkeypatch, capsys, mode):
    clock = SimpleNamespace(now=0.0, cycles=0)
    monkeypatch.setattr(teleop_monitor.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(
        "lerobot.utils.robot_utils.precise_sleep", lambda seconds: setattr(clock, "now", clock.now + seconds)
    )
    robot.feedback_fresh = mode != "waiting_feedback"
    robot.is_connected = True
    robot.connect = Mock()
    robot.disconnect = Mock()
    robot._from_keyboard_to_base_action = Mock(return_value={})
    robot._from_keyboard_to_lift_action = Mock(return_value={})
    robot.send_action.return_value = {"arm_left_gripper.pos": 0.54} if mode == "active" else {}

    def observe(**_kwargs):
        clock.cycles += 1
        if clock.cycles > 60:
            raise KeyboardInterrupt
        return robot.last_remote_state

    robot.get_observation.side_effect = observe
    monkeypatch.setattr("lerobot.robots.alohamini.AlohaMiniClient", lambda _cfg: robot)
    leader = Mock(is_connected=True)
    leader.get_action.return_value = {"left_gripper.pos": 0.54}
    monkeypatch.setattr("lerobot.teleoperators.bi_so_leader.BiSOLeader", lambda _cfg: leader)
    keyboard = Mock(is_connected=True)
    keyboard.get_action.return_value = []
    monkeypatch.setattr(
        "lerobot.teleoperators.keyboard.teleop_keyboard.KeyboardTeleop", lambda _cfg: keyboard
    )
    monkeypatch.setattr("lerobot.utils.visualization_utils.init_rerun", Mock())
    monkeypatch.setattr("lerobot.utils.visualization_utils.log_rerun_data", Mock())
    monkeypatch.setattr(
        "examples.alohamini.safety_utils.RecordingGate",
        lambda *_args: SimpleNamespace(state_ready=lambda: robot.feedback_fresh),
    )
    monkeypatch.setattr("sys.argv", ["teleoperate_bi"])
    with pytest.raises(KeyboardInterrupt):
        runpy.run_module("examples.alohamini.teleoperate_bi", run_name="__main__")
    output = capsys.readouterr().out
    assert f"state={mode}" in output
    assert output.count("[TELEOP]") == 1
    robot.disconnect.assert_called_once()
    leader.disconnect.assert_called_once()
    keyboard.disconnect.assert_called_once()
    if mode != "active":
        assert "sent_hz=0.0" in output


def test_host_keeps_timing_and_publishes_currents_without_tracking_prints(monkeypatch, capsys):
    from lerobot.robots.alohamini import alohamini_host

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(alohamini_host.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(alohamini_host.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(
        alohamini_host.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds)
    )
    monkeypatch.setattr("sys.argv", ["alohamini_host", "--profile-timing", "true"])
    robot = Mock(cameras={}, logs={}, _feedback_currents_raw={"arm_left_gripper": 9.0})
    robot.get_observation.return_value = {"arm_left_gripper.pos": 1.73}
    robot.get_safety_status.return_value = {"version": 1, "host_session_id": "test"}
    robot.supervise_arm_motion.return_value = {}
    host = Mock(max_loop_freq_hz=50, connection_time_s=1.1, watchdog_timeout_ms=1000)
    host.zmq_cmd_socket.recv_string.side_effect = zmq.Again()
    host.zmq_observation_socket.recv_multipart.return_value = [b"client", b"1:state"]
    monkeypatch.setattr(alohamini_host, "AlohaMini", lambda _cfg: robot)
    monkeypatch.setattr(alohamini_host, "AlohaMiniHost", lambda _cfg: host)
    monkeypatch.setattr(alohamini_host, "build_robot_metadata", lambda _robot: {})
    alohamini_host.main()
    output = capsys.readouterr().out
    assert "[HOST TIMING avg ms/loop] Hz=50.0" in output
    assert "HOST TRACKING" not in output
    response = json.loads(host.zmq_observation_socket.send_multipart.call_args.args[0][2])
    assert response["_safety"]["currents_ma"] == {"arm_left_gripper": 58.5}
