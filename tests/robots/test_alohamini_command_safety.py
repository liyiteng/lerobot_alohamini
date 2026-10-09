import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import zmq

from lerobot.robots.alohamini.alohamini_client import AlohaMiniClient
from lerobot.robots.alohamini.command_owner import CommandOwner
from tests.robots.test_alohamini_host import FakeBus, make_robot_feedback_stub


def status(**updates):
    return {
        "version": 1,
        "host_session_id": "host-1",
        "joint_hold_events": 0,
        "watchdog_events": 0,
        "joint_holds": {},
        "gripper_holds": {},
        "watchdog_active": False,
        **updates,
    }


def test_host_status_copies_targets_without_reading_bus():
    bus = FakeBus()
    robot = make_robot_feedback_stub(bus)
    robot._gripper_hold_goal = {}
    robot._arm_goal_positions = {"arm_left_elbow_flex.pos": 9.0}
    robot._arm_sent_positions = {"arm_left_elbow_flex.pos": 1.0}
    snapshot = robot.get_safety_status()
    snapshot["accepted_targets"].clear()
    assert robot._arm_sent_positions == {"arm_left_elbow_flex.pos": 1.0}
    assert snapshot["requested_targets"] == robot._arm_goal_positions
    assert bus.reads == []


@pytest.mark.parametrize("supported", [True, False])
def test_client_adds_command_identity_only_for_updated_host(supported):
    client = object.__new__(AlohaMiniClient)
    client._is_connected = True
    client._feedback_valid = True
    client._feedback_requested_at = time.monotonic()
    client._zmq = zmq
    client.latest_safety_status = status() if supported else {}
    client._command_sequence = 0
    client._client_id = "test-client"
    client.last_sent_command = {}
    client._state_order = ["joint.pos"]
    client.zmq_cmd_socket = Mock()
    action = {"joint.pos": 3.0}
    client.send_action(action)
    payload = json.loads(client.zmq_cmd_socket.send_string.call_args.args[0])
    assert action == {"joint.pos": 3.0}
    if supported:
        assert payload.pop("_command") == {"client_id": "test-client", "sequence": 1}
    assert payload == action


def test_failed_image_decode_does_not_mark_cached_frame_fresh():
    client = object.__new__(AlohaMiniClient)
    client._response_includes_cameras = False
    client.logs = {}
    client.latest_safety_status = {}
    client._response_requested_at = time.monotonic()
    client.last_frames = {"forward": "cached"}
    client.last_remote_state = {}
    client._observation_sequence = 0
    client._poll_and_get_latest_message = lambda **_kwargs: [b"response"]
    client._parse_observation_message = lambda _parts: (
        {"_host_timing": {"camera_capture_monotonic_s": {"forward": 12.0}}, "_safety": status()},
        {},
    )
    client._remote_state_from_obs = lambda *_args: ({}, {"joint.pos": 1.0})
    frames, _state = client._get_data()
    assert frames == {"forward": "cached"}
    assert client.latest_host_timing["camera_capture_monotonic_s"] == {}
    assert client.latest_safety_status == status()


@pytest.mark.parametrize("competing_client", [False, True])
def test_watchdog_cycle_does_not_supervise_twice(monkeypatch, competing_client):
    import zmq

    from lerobot.robots.alohamini import alohamini_host

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(alohamini_host.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(alohamini_host.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(
        alohamini_host.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds)
    )
    monkeypatch.setattr("sys.argv", ["alohamini_host"])
    robot = Mock()
    robot.cameras = {}
    robot.action_features = {"joint.pos": float}
    robot.logs = {}
    robot._feedback_currents_raw = {}
    robot.get_observation.return_value = {"joint.pos": 1.0}
    robot.get_safety_status.side_effect = lambda: status()
    robot.send_action.side_effect = lambda action: dict(action)
    robot.supervise_arm_motion.return_value = {}
    command = {"client_id": "pc", "sequence": 1, "control_epoch": 0, "host_session_id": "host-1"}
    other_command = {"client_id": "ros", "sequence": 1, "control_epoch": 1, "host_session_id": "host-1"}
    competing_message = json.dumps({"joint.pos": 7.0, "_command": other_command})
    host = Mock(max_loop_freq_hz=50, connection_time_s=0.055, watchdog_timeout_ms=10)
    host.zmq_cmd_socket.recv_string.side_effect = [
        json.dumps({"joint.pos": 5.0, "_command": command}),
        json.dumps({"joint.pos": 7.0, "_command": {**other_command, "control_epoch": 0}})
        if competing_client
        else zmq.Again(),
        competing_message if competing_client else zmq.Again(),
    ]
    host.zmq_observation_socket.recv_multipart.return_value = [b"client", b"1:state"]
    monkeypatch.setattr(alohamini_host, "AlohaMini", lambda _config: robot)
    monkeypatch.setattr(alohamini_host, "AlohaMiniHost", lambda _config: host)
    monkeypatch.setattr(alohamini_host, "build_robot_metadata", lambda _robot: {})

    alohamini_host.main()

    assert robot.get_observation.call_count == 3
    assert robot.send_action.call_count == (3 if competing_client else 2)
    assert robot.supervise_arm_motion.call_count == (0 if competing_client else 1)
    responses = [
        json.loads(call.args[0][2])["_safety"]
        for call in host.zmq_observation_socket.send_multipart.call_args_list
    ]
    assert responses[0]["command"] == command
    assert responses[1]["command"] == {}
    assert responses[1]["watchdog_active"]
    assert responses[2]["watchdog_events"] == 1
    assert [row["control_owner"] for row in responses] == ["pc", None, "ros" if competing_client else None]
    if competing_client:
        calls = [call[0] for call in robot.mock_calls]
        assert calls.index("stop_motion") < max(
            index for index, call in enumerate(calls) if call == "send_action"
        )
        assert responses[2]["command"] == other_command


def test_command_owner_rejects_competing_clients_until_watchdog_release():
    owner = CommandOwner()
    pc = {"client_id": "pc", "sequence": 1, "control_epoch": 0, "host_session_id": "host"}
    ros = {**pc, "client_id": "ros"}
    assert owner.accept(pc, "host")
    assert not owner.accept(ros, "host")
    assert not owner.accept({}, "host")
    assert owner.accept({**pc, "sequence": 2}, "host")
    owner.release()
    assert not owner.accept({**pc, "sequence": 3}, "host")
    assert owner.accept({**ros, "control_epoch": 1}, "host")
    assert owner.owner == "ros"


@pytest.mark.parametrize("incoming", ["current", "wrong_epoch", "duplicate", "invalid"])
def test_host_accepts_available_command_before_expiry_but_invalid_commands_do_not_renew(
    monkeypatch, incoming
):
    from lerobot.robots.alohamini import alohamini_host

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(alohamini_host.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(alohamini_host.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(
        alohamini_host.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds)
    )
    monkeypatch.setattr("sys.argv", ["alohamini_host"])
    robot = Mock(cameras={}, logs={}, _feedback_currents_raw={})
    robot.action_features = {"joint.pos": float}
    robot.get_observation.return_value = {"joint.pos": 1.0}
    robot.get_safety_status.side_effect = lambda: status()
    robot.supervise_arm_motion.return_value = {}
    identity = {"client_id": "pc", "sequence": 1, "control_epoch": 0, "host_session_id": "host-1"}
    second = {**identity, "sequence": 2}
    if incoming == "wrong_epoch":
        second["control_epoch"] = 99
    if incoming == "duplicate":
        second["sequence"] = 1
    target = float("nan") if incoming == "invalid" else 7.0
    host = Mock(max_loop_freq_hz=50, connection_time_s=0.055, watchdog_timeout_ms=10)
    host.zmq_cmd_socket.recv_string.side_effect = [
        json.dumps({"joint.pos": 5.0, "_command": identity}),
        json.dumps({"joint.pos": target, "_command": second}),
        json.dumps(
            {
                "joint.pos": 9.0,
                "_command": {
                    **identity,
                    "sequence": 3,
                    "control_epoch": 0 if incoming == "current" else 1,
                },
            }
        ),
    ]
    host.zmq_observation_socket.recv_multipart.return_value = [b"client", b"1:state"]
    monkeypatch.setattr(alohamini_host, "AlohaMini", lambda config: robot)
    monkeypatch.setattr(alohamini_host, "AlohaMiniHost", lambda config: host)
    monkeypatch.setattr(alohamini_host, "build_robot_metadata", lambda robot: {})
    alohamini_host.main()
    replies = [
        json.loads(call.args[0][2])["_safety"]
        for call in host.zmq_observation_socket.send_multipart.call_args_list
    ]
    assert replies[1]["watchdog_active"] == (incoming != "current")
    assert replies[2]["watchdog_events"] == int(incoming != "current")
    assert not replies[2]["watchdog_active"]
    assert replies[2]["command"]["sequence"] == 3
    assert robot.stop_motion.call_count == int(incoming != "current")


def test_command_owner_rejects_replay_and_previous_host_session():
    owner = CommandOwner()
    command = {"client_id": "pc", "sequence": 2, "host_session_id": "host", "control_epoch": 0}
    assert not owner.accept(command, "restarted-host")
    assert owner.accept(command, "host")
    owner.release()
    assert not owner.accept(command, "host")
    assert not owner.accept({**command, "sequence": 1}, "host")


def test_legacy_owner_cannot_be_preempted_by_identified_client():
    owner = CommandOwner()
    assert owner.accept({}, "host")
    assert owner.accept({}, "host")
    assert not owner.accept({"client_id": "pc", "sequence": 1}, "host")
    owner.release()
    assert owner.accept(
        {"client_id": "pc", "sequence": 1, "host_session_id": "host", "control_epoch": 1}, "host"
    )


def test_client_does_not_send_when_control_belongs_to_ros():
    client = object.__new__(AlohaMiniClient)
    client._is_connected = True
    client._client_id = "pc"
    client.latest_safety_status = status(control_owner="ros")
    client.zmq_cmd_socket = Mock()
    assert client.send_action({"joint.pos": 3.0}) == {}
    client.zmq_cmd_socket.send_string.assert_not_called()


def test_first_pc_command_uses_identity_from_handshake():
    client = object.__new__(AlohaMiniClient)
    client._is_connected = False
    client._feedback_valid = False
    client._feedback_requested_at = None
    client._client_id = "pc"
    client._command_sequence = 0
    client._state_order = ["joint.pos"]
    client._zmq = Mock()
    client.remote_ip = "test"
    client.port_zmq_cmd = 5555
    client.port_zmq_observations = 5556
    client.observation_request_window = 3
    client.connect_timeout_s = 1
    client._request_observation = lambda _timeout: [
        json.dumps({"_safety": status(control_owner=None)}).encode()
    ]
    client._fill_observation_request_window = Mock()
    client.connect()
    assert client.send_action({"joint.pos": 1.0}) == {}
    client.zmq_cmd_socket.send_string.assert_not_called()
    client._feedback_valid = True
    client._feedback_requested_at = time.monotonic()
    client.send_action({"joint.pos": 1.0})
    payload = json.loads(client.zmq_cmd_socket.send_string.call_args.args[0])
    assert payload["_command"] == {"client_id": "pc", "sequence": 1, "host_session_id": "host-1"}


def client_stub():
    client = object.__new__(AlohaMiniClient)
    client._is_connected = True
    client._feedback_valid = True
    client._feedback_requested_at = time.monotonic()
    client._client_id = "pc"
    client._command_sequence = 0
    client._state_order = ("joint.pos", "x.vel")
    client.config = SimpleNamespace(robot_model="alohamini1")
    client.latest_safety_status = {
        "version": 1,
        "host_session_id": "host",
        "control_owner": None,
        "control_epoch": 4,
    }
    client.last_sent_command = {}
    client.zmq_cmd_socket = Mock()
    client._zmq = zmq
    return client


@pytest.mark.parametrize("failure", ["stale", "missing", "blocked"])
def test_unsent_action_is_not_reported_as_sent(failure):
    client = client_stub()
    if failure == "stale":
        client._feedback_requested_at -= 1
    elif failure == "missing":
        client._feedback_requested_at = None
    else:
        client.zmq_cmd_socket.send_string.side_effect = zmq.Again()
    assert client.send_action({"joint.pos": 2.0}) == {}
    assert client.last_sent_command == {}
    if failure != "blocked":
        client.zmq_cmd_socket.send_string.assert_not_called()


def test_short_feedback_gap_does_not_block_new_leader_targets(monkeypatch):
    client = client_stub()
    client._feedback_requested_at = 10.0
    client._feedback_valid = False
    monkeypatch.setattr(time, "monotonic", lambda: 10.4)
    assert not client.observation_updated
    assert client.send_action({"joint.pos": 2.0})
    assert client.send_action({"joint.pos": 3.0})
    assert [
        json.loads(call.args[0])["joint.pos"] for call in client.zmq_cmd_socket.send_string.call_args_list
    ] == [2.0, 3.0]


def test_cached_reads_cannot_renew_control_feedback_deadline(monkeypatch):
    client = client_stub()
    client._feedback_requested_at = 10.0
    client._last_safety_received_at = 10.9
    monkeypatch.setattr(time, "monotonic", lambda: 11.1)
    assert not client.control_feedback_valid
    assert client.send_action({"joint.pos": 2.0}) == {}
    client.zmq_cmd_socket.send_string.assert_not_called()


@pytest.mark.parametrize("delay", [0.4, 2.0])
def test_delayed_valid_reply_is_a_sample_but_cannot_extend_control_deadline(monkeypatch, delay):
    client = client_stub()
    client.logs = {}
    client._response_requested_at = 10.0
    client._response_includes_cameras = False
    client._observation_sequence = 0
    client.last_frames = {}
    client.last_remote_state = {"joint.pos": 1.0}
    client._poll_and_get_latest_message = lambda **kwargs: [b"reply"]
    client._parse_observation_message = lambda parts: ({"_safety": status()}, {})
    client._remote_state_from_obs = lambda *args: ({}, {"joint.pos": 2.0})
    monkeypatch.setattr(time, "monotonic", lambda: 10.0 + delay)
    assert client._get_data()[1] == {"joint.pos": 2.0}
    assert client.control_feedback_valid == (delay < 1.0)
    assert client.observation_updated
    assert client.observation_sequence == 1


def test_command_echoes_epoch_and_uses_nonblocking_transport():
    client = client_stub()
    assert client.send_action({"joint.pos": 2.0})
    call = client.zmq_cmd_socket.send_string.call_args
    assert call.kwargs["flags"] == zmq.NOBLOCK
    assert json.loads(call.args[0])["_command"]["control_epoch"] == 4


@pytest.mark.parametrize(
    "observation",
    [
        {"joint.pos": 1},
        {"joint.pos": float("nan"), "x.vel": 0},
        {"joint.pos": 1, "x.vel": 0, "_robot_metadata": {"robot_model": "alohamini2pro"}},
    ],
)
def test_missing_nonfinite_or_wrong_model_feedback_is_rejected(observation):
    with pytest.raises((KeyError, ValueError)):
        client_stub()._remote_state_from_obs(observation, {})


def test_unmatched_response_cannot_refresh_feedback():
    client = client_stub()
    client.logs = {}
    client.last_frames = {}
    client.last_remote_state = {"joint.pos": 1}
    client._response_requested_at = None
    client._observation_sequence = 8
    client._poll_and_get_latest_message = lambda **_kwargs: [b"late"]
    client._parse_observation_message = Mock()
    assert client._get_data()[1] == {"joint.pos": 1}
    assert client.observation_sequence == 8
    assert not client.observation_updated
    client._parse_observation_message.assert_not_called()


@pytest.mark.parametrize("write_fails", [False, True])
def test_host_watchdog_lease_begins_after_slow_successful_write(monkeypatch, write_fails):
    import zmq

    from lerobot.robots.alohamini import alohamini_host

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(alohamini_host.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(alohamini_host.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(
        alohamini_host.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds)
    )
    monkeypatch.setattr("sys.argv", ["alohamini_host"])
    robot = Mock(cameras={}, logs={}, _feedback_currents_raw={})
    robot.action_features = {"joint.pos": float}
    robot.get_observation.return_value = {"joint.pos": 1.0}
    robot.get_safety_status.side_effect = lambda: status()
    robot.supervise_arm_motion.return_value = {}

    def write(action):
        clock.now += 0.030
        if write_fails:
            raise OSError("servo write failed")
        return dict(action)

    robot.send_action.side_effect = write
    host = Mock(max_loop_freq_hz=50, connection_time_s=0.080, watchdog_timeout_ms=25)
    host.zmq_cmd_socket.recv_string.side_effect = [
        json.dumps({"joint.pos": 5.0}),
        zmq.Again(),
        zmq.Again(),
        zmq.Again(),
    ]
    host.zmq_observation_socket.recv_multipart.return_value = [b"client", b"1:state"]
    monkeypatch.setattr(alohamini_host, "AlohaMini", lambda config: robot)
    monkeypatch.setattr(alohamini_host, "AlohaMiniHost", lambda config: host)
    monkeypatch.setattr(alohamini_host, "build_robot_metadata", lambda robot: {})

    if write_fails:
        with pytest.raises(OSError, match="servo write failed"):
            alohamini_host.main()
        host.zmq_observation_socket.send_multipart.assert_not_called()
        robot.disconnect.assert_called_once()
        host.disconnect.assert_called_once()
    else:
        alohamini_host.main()
        replies = [
            json.loads(call.args[0][2])["_safety"]
            for call in host.zmq_observation_socket.send_multipart.call_args_list
        ]
        assert [reply["watchdog_active"] for reply in replies] == [False, False, False, True]
        assert replies[-1]["watchdog_events"] == 1
