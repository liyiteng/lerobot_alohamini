import time
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
    client._receive_observation_response = Mock(return_value=[b"fresh response"])
    client.zmq_observation_socket.send.side_effect = zmq.Again()

    assert client._poll_and_get_latest_message(include_cameras=False) == [b"fresh response"]
    client._receive_observation_response.assert_called_once_with(b"0:state", client.polling_timeout_ms)
    assert not client._observation_request_tokens
    assert not client._request_times
    assert not caplog.records


def test_backpressure_without_feedback_still_prevents_commands(client):
    client._feedback_valid = True
    client._feedback_requested_at = time.monotonic()
    client.zmq_observation_socket.send.side_effect = zmq.Again()

    client.get_observation(include_cameras=False)

    assert not client.feedback_fresh
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
