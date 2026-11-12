import json
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from examples.alohamini import evaluate_bi
from examples.alohamini.evaluation_safety import EvaluationSafetyGuard
from lerobot.robots.alohamini.alohamini_client import AlohaMiniClient
from tests.robots.test_alohamini_command_safety import client_stub, status


def test_refresh_discards_prefetched_tokens_without_waiting_for_each_reply():
    client = client_stub()
    client._observation_request_tokens = deque([b"1:camera", b"2:camera"])
    client._request_times = {b"1:camera": 1.0, b"2:camera": 1.0}

    def receive():
        assert not client._observation_request_tokens
        assert not client._request_times
        assert not client.observation_updated
        return {"joint.pos": 3.0}

    client.get_observation = receive
    assert client.refresh_observation() == {"joint.pos": 3.0}
    client.zmq_cmd_socket.send_string.assert_not_called()


@pytest.mark.parametrize("delay", [0.05, 0.3, 0.6, 1.2])
@pytest.mark.parametrize(
    "fault", [None, "feedback_lost", "feedback_gap", "watchdog", "restart", "owner", "joint_event", "epoch_jump", "joint_hold", "initial_hold"]
)
def test_actual_evaluation_loop_rechecks_feedback_after_sync_inference(monkeypatch, tmp_path, delay, fault):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(evaluate_bi.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(evaluate_bi.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(
        evaluate_bi, "precise_sleep", lambda seconds: setattr(clock, "now", clock.now + seconds)
    )
    client = client_stub()
    client._state_order = ("joint.pos", "x.vel", "y.vel", "theta.vel")
    client.config.cameras = {}
    client._left_arm_state_keys = ("joint.pos",)
    client._right_arm_state_keys = ()
    client._observation_request_tokens = deque()
    client._request_times = {}
    client._observation_sequence = 0
    client._last_safety_received_at = clock.now
    client.last_remote_state = {"joint.pos": 1.0}
    client.latest_safety_status = status(control_owner=None, control_epoch=0)
    client.latest_host_timing = {}
    client.connect = Mock()
    client.disconnect = Mock()
    computed = []
    frames = []
    refreshed = []
    missed_reads = []
    held_reads = []

    def get_observation():
        if computed and fault == "feedback_gap" and len(missed_reads) < 2:
            missed_reads.append(clock.now)
            client._feedback_valid = False
            return client.last_remote_state
        if computed and fault == "feedback_lost":
            client._feedback_valid = False
            return client.last_remote_state
        # Prefetched replies are from before the blocking policy call.
        if client._observation_request_tokens and clock.now - client._feedback_requested_at >= 0.25:
            client._feedback_valid = False
            return client.last_remote_state
        client._observation_sequence += 1
        client._feedback_valid = True
        client._feedback_requested_at = clock.now
        client._last_safety_received_at = clock.now
        client._observation_request_tokens.append(b"prefetched")
        if fault == "initial_hold":
            client.latest_safety_status["joint_holds"] = {"elbow": 1.0} if not held_reads else {}
            held_reads.append(clock.now)
        if computed:
            updates = {
                "watchdog": {
                    "watchdog_active": True,
                    "watchdog_events": len(computed),
                    "control_epoch": len(computed),
                },
                "restart": {"host_session_id": "host-restarted"},
                "owner": {"control_owner": "another-client"},
                "joint_event": {"joint_hold_events": 1},
                "epoch_jump": {"control_epoch": 3, "watchdog_events": 3},
            }
            client.latest_safety_status.update(updates.get(fault, {}))
            if fault == "joint_hold":
                client.latest_safety_status["joint_hold_events"] = 1
                client.latest_safety_status["joint_holds"] = {"elbow": 1.0} if not held_reads else {}
                held_reads.append(clock.now)
        return client.last_remote_state

    def refresh():
        refreshed.append(clock.now)
        return AlohaMiniClient.refresh_observation(client)

    client.get_observation = get_observation
    client.refresh_observation = refresh
    engine = Mock()
    engine._rtc_thread = None

    def infer(_frame):
        computed.append(clock.now)
        clock.now += delay
        # A fresh response is required to discover a fault; force a refresh in fault cases.
        if fault:
            client._feedback_valid = False
        return torch.tensor([7.0])

    engine.get_action.side_effect = infer
    dataset = Mock(root=tmp_path, meta=SimpleNamespace(total_episodes=0, stats={}, info={}))
    dataset.features = {}
    dataset.has_pending_frames.return_value = False
    dataset.add_frame.side_effect = frames.append
    monkeypatch.setattr(evaluate_bi, "AlohaMiniClient", lambda _config: client)
    monkeypatch.setattr(evaluate_bi, "ThreadSafeRobot", lambda robot: robot)
    monkeypatch.setattr(AlohaMiniClient, "action_features", property(lambda _self: {"joint.pos": float}))
    monkeypatch.setattr(evaluate_bi, "auto_select_torch_device", lambda: "cpu")
    monkeypatch.setattr(
        evaluate_bi.PreTrainedConfig, "from_pretrained", lambda _path: SimpleNamespace(type="act")
    )
    monkeypatch.setattr(evaluate_bi, "get_policy_class", lambda _type: Mock())
    monkeypatch.setattr(
        evaluate_bi,
        "make_default_processors",
        lambda: (lambda pair: pair[0], lambda pair: pair[0], lambda obs: obs),
    )
    monkeypatch.setattr(evaluate_bi, "aggregate_pipeline_dataset_features", lambda **_kwargs: {})
    monkeypatch.setattr(evaluate_bi, "hw_to_dataset_features", lambda *_args: {})
    monkeypatch.setattr(
        evaluate_bi, "build_dataset_frame", lambda _features, data, prefix: {prefix: dict(data)}
    )
    monkeypatch.setattr(evaluate_bi, "make_pre_post_processors", lambda **_kwargs: (None, None))
    monkeypatch.setattr(evaluate_bi, "create_inference_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(evaluate_bi.LeRobotDataset, "create", lambda **_kwargs: dataset)
    monkeypatch.setattr(evaluate_bi, "log_say", lambda *_args: None)
    # 此处只验证推理与反馈时序，不能启动受系统权限和真实时钟影响的键盘监听。
    monkeypatch.setattr(evaluate_bi, "init_keyboard_listener", lambda: (None, {"exit_early": False, "stop_recording": False}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_bi",
            "--policy.path=test",
            "--dataset.repo_id=test/test",
            "--num_episodes=1",
            "--episode_time=2",
            "--dataset.push_to_hub=false",
        ],
    )
    prompt = Mock(side_effect=AssertionError("Evaluation must not require Enter"))
    monkeypatch.setattr("builtins.input", prompt)

    if fault in ("restart", "owner"):
        with pytest.raises(RuntimeError, match="评估停止"):
            evaluate_bi.main()
    else:
        evaluate_bi.main()
    policy_commands = [
        json.loads(call.args[0])
        for call in client.zmq_cmd_socket.send_string.call_args_list
        if json.loads(call.args[0]).get("joint.pos") == 7.0
    ]
    if fault in ("restart", "owner") or (fault == "feedback_lost" and delay >= 1.0):
        assert not policy_commands
        assert not frames
        engine.pause.assert_called()
    else:
        assert len(policy_commands) >= 1
        assert len(frames) <= len(policy_commands)
        if fault in ("joint_event", "joint_hold"):
            assert engine.reset.call_count == 2
            assert engine.get_action.call_count == len(policy_commands) + 1
        if fault in ("watchdog", "epoch_jump"):
            assert engine.reset.call_count == 1
            epochs = [row["_command"]["control_epoch"] for row in policy_commands]
            assert epochs == (list(range(1, len(epochs) + 1)) if fault == "watchdog" else [3] * len(epochs))
            assert engine.get_action.call_count == len(policy_commands)
        if delay > 0.25:
            assert len(refreshed) >= len(policy_commands)
        if fault == "feedback_lost":
            assert all(start + delay < 11.0 for start in computed[:len(policy_commands)])
            assert not frames
    prompt.assert_not_called()
    if not frames:
        dataset.save_episode.assert_not_called()
    client.disconnect.assert_called_once()
    dataset.finalize.assert_called_once()


def test_recording_delay_limit_does_not_gate_evaluation(monkeypatch):
    client = client_stub()
    client.latest_safety_status = status()
    client._feedback_requested_at = 10.0
    client._last_safety_received_at = 10.4
    monkeypatch.setattr(time, "monotonic", lambda: 10.4)
    client.refresh_observation = Mock()
    assert EvaluationSafetyGuard().check_observation(client, {}) == ({}, None)
    client.refresh_observation.assert_not_called()


def test_fresh_feedback_needs_no_extra_round_trip():
    robot = SimpleNamespace(
        control_feedback_valid=True,
        latest_safety_status=status(),
        _last_safety_received_at=time.monotonic(),
        refresh_observation=Mock(),
    )
    observation = {"joint.pos": 1.0}
    assert EvaluationSafetyGuard().check_observation(robot, observation) == (observation, None)
    robot.refresh_observation.assert_not_called()
