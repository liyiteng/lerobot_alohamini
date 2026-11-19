import json
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import zmq

from lerobot.robots.alohamini.alohamini_client import AlohaMiniClient
from lerobot.robots.alohamini.config_alohamini import AlohaMiniClientConfig


@pytest.fixture
def client():
    robot = AlohaMiniClient(AlohaMiniClientConfig(remote_ip="127.0.0.1", cameras={}))
    robot._is_connected = True
    robot.zmq_observation_socket = Mock()
    robot.zmq_cmd_socket = Mock()
    return robot


def test_queue_backpressure_does_not_log_error_or_register_unsent_request(client, caplog):
    client._observation_request_tokens.append(b"0:state")
    client._request_times[b"0:state"] = 1.0
    client.zmq_observation_socket.send.side_effect = zmq.Again()

    assert client._send_observation_request() is None

    assert not caplog.records
    assert client._request_times == {b"0:state": 1.0}
    assert list(client._observation_request_tokens) == [b"0:state"]
    client.zmq_observation_socket.send.assert_called_once_with(b"1:camera", flags=zmq.NOBLOCK)


def test_request_window_stops_on_backpressure_and_refills_later(client, caplog):
    client.zmq_observation_socket.send.side_effect = [None, zmq.Again()]
    client._fill_observation_request_window(include_cameras=False)
    assert list(client._observation_request_tokens) == [b"1:state"]
    assert set(client._request_times) == {b"1:state"}
    assert client.zmq_observation_socket.send.call_count == 2

    client.zmq_observation_socket.send.side_effect = None
    client._fill_observation_request_window(include_cameras=False)
    assert list(client._observation_request_tokens) == [b"1:state", b"3:state", b"4:state"]
    assert set(client._request_times) == set(client._observation_request_tokens)
    assert not caplog.records


def test_valid_response_is_returned_even_when_replenishment_is_backpressured(client, caplog):
    client._observation_request_tokens.append(b"0:state")
    client._request_times[b"0:state"] = time.monotonic()
    client._receive_observation_response = Mock(return_value=[b"fresh response"])
    client.zmq_observation_socket.send.side_effect = zmq.Again()

    assert client._poll_and_get_latest_message(include_cameras=False) == [b"fresh response"]
    client._receive_observation_response.assert_called_once_with(b"0:state", client.polling_timeout_ms)
    assert not client._observation_request_tokens
    assert not client._request_times
    assert not caplog.records


def test_backpressure_after_prolonged_feedback_loss_prevents_commands(client):
    client._feedback_valid = True
    client._feedback_requested_at = time.monotonic() - 2.0
    client.zmq_observation_socket.send.side_effect = zmq.Again()

    client.get_observation(include_cameras=False)

    assert not client.observation_updated
    assert client.send_action({"x.vel": 0.1}) == {}
    client.zmq_cmd_socket.send_string.assert_not_called()
    assert not client._request_times


@pytest.mark.parametrize("errno", [zmq.ENOTSOCK, zmq.ETERM])
def test_real_transport_error_is_reported_without_registering_request(client, caplog, errno):
    client.zmq_observation_socket.send.side_effect = zmq.ZMQError(errno)
    assert client._send_observation_request() is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert "ZMQ observation request failed" in caplog.text
    assert not client._request_times


class DelayedLink:
    """Serialize replies through a slow link while exercising the real client receive loop."""

    def __init__(self, client, monkeypatch, delay):
        self.client = client
        self.delay = delay
        self.now = 10.0
        self.replies = deque()
        self.sent = []
        self.drop = False
        self.epoch = 0
        monkeypatch.setattr(time, "monotonic", lambda: self.now)
        monkeypatch.setattr(
            zmq, "Poller", lambda: SimpleNamespace(register=lambda *_args: None, poll=self.poll)
        )
        client.zmq_observation_socket.send.side_effect = self.send
        client.zmq_observation_socket.recv_multipart.side_effect = self.receive

    def send(self, token, **_kwargs):
        self.sent.append(token)
        if self.drop:
            return
        state = dict.fromkeys(self.client._state_order, 0.0)
        state["_safety"] = {
            "version": 1,
            "host_session_id": "test-host",
            "control_owner": None,
            "control_epoch": self.epoch,
            "command_watchdog_timeout_s": 1.0,
        }
        deliver_at = max(self.now, self.replies[-1][0] if self.replies else self.now) + self.delay
        self.replies.append((deliver_at, [token, json.dumps(state).encode()]))

    def poll(self, timeout_ms):
        deadline = self.now + timeout_ms / 1000
        if self.replies and self.replies[0][0] <= deadline:
            self.now = max(self.now, self.replies[0][0])
            return [(self.client.zmq_observation_socket, zmq.POLLIN)]
        self.now = deadline
        return []

    def receive(self, *_args):
        if self.replies and self.replies[0][0] <= self.now:
            return self.replies.popleft()[1]
        raise zmq.Again()


def test_fast_link_keeps_configured_request_window(client, monkeypatch):
    DelayedLink(client, monkeypatch, 0.02)
    for _ in range(10):
        client.get_observation(include_cameras=False)
        assert client.observation_updated
        assert client.control_feedback_valid
        assert client._active_observation_request_window == 3
        assert len(client._request_times) == 3


@pytest.mark.parametrize("delay", [0.35, 0.6])
@pytest.mark.parametrize("window", [1, 3])
def test_slow_link_eventually_sends_commands_without_replacing_delayed_requests(
    client, monkeypatch, delay, window
):
    link = DelayedLink(client, monkeypatch, delay)
    client._active_observation_request_window = window
    command_times = []
    for _ in range(60):
        client.get_observation(include_cameras=False)
        if client.observation_updated and client.send_action({"x.vel": 0.1}):
            command_times.append(link.now)
        assert len(client._request_times) <= window
        if len(command_times) == 8:
            break

    assert len(command_times) == 8
    assert client._active_observation_request_window == 1
    # After draining the initial burst, command delivery follows link throughput
    # and stays within the Host watchdog. A single request avoids the initial burst.
    steady_times = command_times if window == 1 else command_times[-5:]
    assert all(b - a < 1.0 for a, b in zip(steady_times, steady_times[1:], strict=False))


def test_poll_timeout_preserves_token_and_original_request_time(client, monkeypatch):
    link = DelayedLink(client, monkeypatch, 0.35)
    client._active_observation_request_window = 1

    client.get_observation(include_cameras=False)
    assert not client.observation_updated
    assert list(client._observation_request_tokens) == [b"1:state"]
    assert client._request_times == {b"1:state": 10.0}
    assert link.sent == [b"1:state"]
    assert client.send_action({"x.vel": 0.1}) == {}

    client.get_observation(include_cameras=False)
    assert client.observation_updated
    assert client._feedback_requested_at == 10.0
    assert client.send_action({"x.vel": 0.1})
    assert link.sent == [b"1:state", b"2:state"]


def test_missing_response_expires_without_dropping_following_tokens(client, monkeypatch):
    link = DelayedLink(client, monkeypatch, 0.35)
    link.drop = True
    client._fill_observation_request_window(include_cameras=False)

    link.now = 10.9
    assert client._poll_and_get_latest_message(include_cameras=False) is None
    assert list(client._observation_request_tokens) == [b"2:state", b"3:state"]
    assert client._request_times == {b"2:state": 10.0, b"3:state": 10.0}
    assert link.sent == [b"1:state", b"2:state", b"3:state"]


def test_link_loss_stops_commands_then_recovers_with_current_epoch(client, monkeypatch):
    link = DelayedLink(client, monkeypatch, 0.35)
    client._active_observation_request_window = 1
    for _ in range(2):
        client.get_observation(include_cameras=False)
    assert client.send_action({"x.vel": 0.1})

    link.drop = True
    link.replies.clear()
    for _ in range(12):
        client.get_observation(include_cameras=False)
        assert len(client._request_times) <= 1
    assert not client.control_feedback_valid
    assert client._feedback_requested_at == 10.0
    client.zmq_cmd_socket.reset_mock()
    assert client.send_action({"x.vel": 0.1}) == {}
    client.zmq_cmd_socket.send_string.assert_not_called()

    link.drop = False
    link.epoch = 1
    for _ in range(12):
        client.get_observation(include_cameras=False)
        if client.send_action({"x.vel": 0.1}):
            break
    assert client.control_feedback_valid
    payload = json.loads(client.zmq_cmd_socket.send_string.call_args.args[0])
    assert payload["_command"]["control_epoch"] == 1


def test_delayed_tcp_reply_is_consumed_on_next_poll(client):
    """A real DEALER/ROUTER reply arriving after 200 ms must still reach control."""
    with zmq.Context() as context, context.socket(zmq.ROUTER) as host, context.socket(zmq.DEALER) as peer:
        host.setsockopt(zmq.LINGER, 0)
        peer.setsockopt(zmq.LINGER, 0)
        port = host.bind_to_random_port("tcp://127.0.0.1")
        peer.connect(f"tcp://127.0.0.1:{port}")
        client.zmq_observation_socket = peer
        client._active_observation_request_window = 1

        client.get_observation(include_cameras=False)
        assert not client.observation_updated
        assert host.poll(1000)
        identity, token = host.recv_multipart()
        assert token == b"1:state"
        # The first poll already used 200 ms. Deliver the reply after another 150 ms.
        time.sleep(0.15)
        state = dict.fromkeys(client._state_order, 0.0)
        host.send_multipart([identity, token, json.dumps(state).encode()])

        client.get_observation(include_cameras=False)
        assert client.observation_updated
        assert client.control_feedback_valid
        assert client.send_action({"x.vel": 0.1})
