import time
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from examples.alohamini.safety_utils import RecordingCadence, RecordingGate, SafetyRecorder, preserve_dataset
from lerobot.robots.alohamini.alohamini_client import AlohaMiniClient
from lerobot.robots.alohamini.config_alohamini import AlohaMiniClientConfig


@pytest.mark.parametrize("multiplier", [1, 2, 3, 4])
def test_recording_cadence_is_independent_of_interpolation(multiplier):
    cadence = RecordingCadence(30)
    samples = [i / (30 * multiplier) for i in range(30 * multiplier)]
    assert sum(cadence.ready(stamp) for stamp in samples) == 30
    assert cadence.ready(10)
    assert not cadence.ready(10.001)


def test_slow_safety_writer_is_bounded_and_does_not_block_close(tmp_path, monkeypatch):
    entered, release = Event(), Event()
    monkeypatch.setattr(SafetyRecorder, "QUEUE_SIZE", 2)
    monkeypatch.setattr(SafetyRecorder, "CLOSE_TIMEOUT_S", 0.01)

    def slow_writer(*_args):
        entered.set()
        release.wait(2)

    monkeypatch.setattr(SafetyRecorder, "_write_loop", slow_writer)
    dataset = SimpleNamespace(root=tmp_path, meta=SimpleNamespace(total_episodes=0, info={}))
    recorder = SafetyRecorder(dataset)
    try:
        recorder.write(safety={})
        assert entered.wait(1)
        for _ in range(20):
            recorder.write(safety={})
        started = time.monotonic()
        recorder.__exit__()
        assert time.monotonic() - started < 0.5
        assert recorder._queue.qsize() == 2
        assert recorder._dropped == 19
        assert dataset.meta.info["safety_format"] == 1
    finally:
        release.set()
        recorder._thread.join(1)


def test_cleanup_does_not_mask_original_error_or_retry_partial_commit():
    dataset = Mock()
    dataset.writer.save_failed = True
    dataset.finalize.side_effect = OSError("footer failed")
    with pytest.raises(RuntimeError, match="original"), preserve_dataset(dataset):
        raise RuntimeError("original")
    dataset.save_episode.assert_not_called()


def test_placeholder_camera_cannot_pass_recording_gate():
    client = AlohaMiniClient(AlohaMiniClientConfig(remote_ip="127.0.0.1", cameras={}))
    client._is_connected = True
    client._cameras_ft = {"chest": (2, 2, 3)}
    client._get_data = lambda **_kwargs: ({}, {})
    dataset = SimpleNamespace(features={"observation.images.chest": {"dtype": "video"}})
    gate = RecordingGate(client, dataset, SafetyRecorder(None))
    observation = client.get_observation()
    assert "chest" in observation
    assert not gate.frame_ready(observation)
