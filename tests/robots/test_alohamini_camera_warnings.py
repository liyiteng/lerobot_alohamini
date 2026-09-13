import json
import logging
import time
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.robots.alohamini.alohamini_client import AlohaMiniClient
from lerobot.robots.alohamini.config_alohamini import AlohaMiniClientConfig


@pytest.fixture
def client():
    robot = AlohaMiniClient(AlohaMiniClientConfig(remote_ip="127.0.0.1"))
    robot._is_connected = True
    robot._cameras_ft = dict.fromkeys(("forward", "backward", "chest", "wrist_left"), (2, 2, 3))
    robot._state_order = ("joint.pos",)
    robot._poll_and_get_latest_message = Mock()
    return robot


def receive(client, *, enabled=None, frames=None, cameras=True):
    frame_names = ("forward",) if frames is None else frames
    observation = {
        "joint.pos": 1.0,
        "_robot_metadata": {"robot_model": client.config.robot_model},
        "_host_timing": {"camera_capture_monotonic_s": dict.fromkeys(frame_names, 1.0)},
    }
    if enabled is not None:
        observation["_robot_metadata"]["cameras"] = enabled
    client._response_requested_at = time.monotonic()
    client._response_includes_cameras = cameras
    client._parse_observation_message = lambda _parts: (
        observation,
        {name: np.ones((2, 2, 3), dtype=np.uint8) for name in frame_names},
    )
    return client.get_observation(include_cameras=cameras)


def test_disabled_cameras_do_not_warn_or_change_schema(client, caplog):
    features = dict(client.observation_features)
    for _ in range(5):
        observation = receive(client, enabled=["forward"])
    assert not caplog.records
    assert client.observation_features == features
    assert not observation["backward"].any()
    assert "backward" not in client.latest_host_timing["camera_capture_monotonic_s"]


def test_enabled_missing_camera_warns_once_and_again_only_after_recovery(client, caplog):
    for _ in range(5):
        receive(client, enabled=["forward"], frames=[])
    assert len(caplog.records) == 1
    assert "forward" in caplog.text
    receive(client, enabled=["forward"])
    receive(client, enabled=["forward"], frames=[])
    assert len(caplog.records) == 2
    # The cached display image must not acquire a fresh capture timestamp.
    assert client.latest_host_timing["camera_capture_monotonic_s"] == {}


def test_state_only_response_does_not_warn_or_clear_missing_state(client, caplog):
    receive(client, enabled=["forward"], frames=[], cameras=False)
    assert not caplog.records
    receive(client, enabled=["forward"], frames=[])
    receive(client, enabled=["forward"], frames=[], cameras=False)
    receive(client, enabled=["forward"], frames=[])
    assert len(caplog.records) == 1


def test_camera_free_host_does_not_warn(client, caplog):
    receive(client, enabled=[], frames=[])
    assert not caplog.records


def test_failed_jpeg_decode_warns_once_without_fresh_timestamp(client, caplog):
    state = {
        "joint.pos": 1.0,
        "_robot_metadata": {"robot_model": client.config.robot_model, "cameras": ["forward"]},
        "_host_timing": {"camera_capture_monotonic_s": {"forward": 1.0}},
    }
    client._poll_and_get_latest_message.return_value = [
        json.dumps(state).encode(),
        b"forward",
        b"invalid jpeg",
    ]
    client._response_includes_cameras = True
    for _ in range(5):
        client._response_requested_at = time.monotonic()
        client.get_observation()
    assert len(caplog.records) == 1
    assert client.latest_host_timing["camera_capture_monotonic_s"] == {}


@pytest.mark.parametrize("enabled", [None, "invalid", [None]])
def test_legacy_or_invalid_camera_metadata_reports_configuration_once(client, caplog, enabled):
    for _ in range(5):
        receive(client, enabled=enabled)
    assert len(caplog.records) == 3
    assert all("configuration" in record.message for record in caplog.records)


def test_disabling_missing_camera_is_not_reported_as_recovery(client, caplog):
    caplog.set_level(logging.INFO)
    receive(client, enabled=["forward", "chest"])
    receive(client, enabled=["forward"])
    assert len(caplog.records) == 1


def test_no_response_does_not_repeat_warnings(client, caplog):
    receive(client, enabled=["forward"], frames=[])
    client._poll_and_get_latest_message.return_value = None
    for _ in range(5):
        client.get_observation()
    assert len(caplog.records) == 1


def test_response_kind_follows_matched_token_not_next_request(client, monkeypatch):
    poller = Mock()
    client.zmq_observation_socket = Mock()
    poller.poll.return_value = [(client.zmq_observation_socket, client._zmq.POLLIN)]
    monkeypatch.setattr(client._zmq, "Poller", lambda: poller)
    for token, expected in [(b"1:state", False), (b"2:camera", True)]:
        client._request_times[token] = time.monotonic()
        client.zmq_observation_socket.recv_multipart.return_value = [token, json.dumps({}).encode()]
        assert client._receive_observation_response(token, 200) == [b"{}"]
        assert client._response_includes_cameras is expected
