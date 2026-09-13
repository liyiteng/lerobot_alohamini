"""Protection metadata and recovery for AlohaMini PC workflows."""

import json
import logging
import math
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread

try:
    from .evaluation_safety import (
        EvaluationSafetyGuard as EvaluationSafetyGuard,
        hold_action as hold_action,
        safety_snapshot,
        stop_inference as stop_inference,
    )
except ImportError:
    from evaluation_safety import (
        EvaluationSafetyGuard as EvaluationSafetyGuard,
        hold_action as hold_action,
        safety_snapshot,
        stop_inference as stop_inference,
    )

logger = logging.getLogger(__name__)


@contextmanager
def preserve_dataset(dataset):
    """Flush a partial episode on interruption as well as normal completion."""
    original_error = None
    try:
        yield
    except BaseException as error:
        original_error = error
        raise
    finally:
        try:
            if getattr(getattr(dataset, "writer", None), "save_failed", False) is True:
                logger.error("Not retrying an interrupted dataset save; recovery files were retained")
            elif dataset.has_pending_frames():
                dataset.save_episode()
        except BaseException as error:
            if original_error is None:
                original_error = error
                raise
            logger.exception("Unable to save the partial episode during cleanup")
        finally:
            try:
                dataset.finalize()
            except BaseException:
                if original_error is None:
                    raise
                logger.exception("Unable to finalize dataset during cleanup")


class SafetyRecorder:
    """Bounded, nonblocking logging with a closing record to attest completeness."""

    QUEUE_SIZE = 2048
    CLOSE_TIMEOUT_S = 1.0

    def __init__(self, dataset):
        self.dataset = dataset
        if dataset is not None and isinstance(getattr(dataset.meta, "info", None), dict):
            dataset.meta.info["safety_format"] = 1
        self._queue = Queue(maxsize=self.QUEUE_SIZE)
        self._thread = None
        self._closing = Event()
        self._frame_index = 0
        self._dropped = 0
        self._write_errors = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self._closing.set()
        if self._thread is not None:
            self._thread.join(timeout=self.CLOSE_TIMEOUT_S)
            if self._thread.is_alive() or self._write_errors or self._dropped:
                logger.error(
                    "Safety log incomplete: dropped=%d, write_errors=%d, writer_pending=%s",
                    self._dropped,
                    self._write_errors,
                    self._thread.is_alive(),
                )

    def _write_loop(self, path, episode):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", buffering=1) as stream:
                while not self._closing.is_set() or not self._queue.empty():
                    try:
                        row = self._queue.get(timeout=0.05)
                    except Empty:
                        continue
                    stream.write(row)
                stream.write(
                    json.dumps(
                        {
                            "episode_index": episode,
                            "frame_index": None,
                            "event": {
                                "type": "recorder_closed",
                                "frame_count": self._frame_index,
                                "dropped_records": self._dropped,
                            },
                        }
                    )
                    + "\n"
                )
        except OSError:
            self._write_errors += 1
            logger.exception("Safety metadata write failed; the log is incomplete")

    def write(
        self,
        *,
        safety,
        requested_action=None,
        issued_command=None,
        event=None,
        alignment_error_s=None,
        host_timing=None,
    ):
        if self.dataset is None:
            return
        if self._closing.is_set():
            raise RuntimeError("Safety recorder is closed")
        frame_index = self._frame_index if event is None else None
        if event is None:
            self._frame_index += 1
        episode = int(self.dataset.meta.total_episodes)
        if self._thread is None:
            path = Path(self.dataset.root) / "meta" / "safety" / f"episode_{episode:06d}.jsonl"
            self._thread = Thread(
                target=self._write_loop, args=(path, episode), daemon=True, name="SafetyRecorder"
            )
            self._thread.start()
        row = {
            "episode_index": episode,
            "frame_index": frame_index,
            "client_monotonic_s": time.monotonic(),
            "safety": safety,
            "feedback_phase": "before_issued_command",
            "requested_action": requested_action,
            "issued_command": issued_command,
            "event": event,
            "alignment_error_s": alignment_error_s,
            "host_timing": host_timing,
        }
        try:
            self._queue.put_nowait(
                json.dumps(row, default=lambda value: value.item(), ensure_ascii=False) + "\n"
            )
        except Full:
            self._dropped += 1


class RecordingGate:
    """Keep cached feedback out of recorded frames and out of new motor commands."""

    def __init__(self, robot, dataset, recorder):
        self.robot = robot
        self.recorder = recorder
        self._sequence = getattr(robot, "observation_sequence", None)
        self._waiting = {}
        self._wait_started = {}
        self._warned = set()
        self._camera_stamps = {}
        self.cameras = tuple(
            key.removeprefix("observation.images.")
            for key, feature in getattr(dataset, "features", {}).items()
            if key.startswith("observation.images.") and feature.get("dtype") in {"image", "video"}
        )

    def _ready(self, kind, reason):
        previous = self._waiting.get(kind)
        if reason != previous:
            self._waiting[kind] = reason
            self._wait_started[kind] = time.perf_counter()
            if reason is None and kind in self._warned:
                print("\nCapture feedback recovered.", flush=True)
                self._warned.discard(kind)
            self.recorder.write(
                safety=safety_snapshot(self.robot),
                event={"type": f"{kind}_{'wait' if reason else 'recovered'}", "reason": reason},
            )
        if reason and kind not in self._warned and time.perf_counter() - self._wait_started[kind] >= 0.5:
            print(f"\n{reason} Press -> to finish and save.", flush=True)
            self._warned.add(kind)
        return reason is None

    def state_ready(self):
        sequence = getattr(self.robot, "observation_sequence", None)
        fresh = sequence is None or sequence != self._sequence
        self._sequence = sequence
        if not fresh or not getattr(self.robot, "feedback_fresh", True):
            return self._ready("feedback", "Waiting for new Host feedback; no new actions sent.")
        if not getattr(self.robot, "command_permitted", True):
            return self._ready("feedback", "Another client controls the robot; capture is waiting.")
        return self._ready("feedback", None)

    def frame_ready(self, observation):
        if not self.cameras:
            return True
        timing = getattr(self.robot, "latest_host_timing", {})
        stamps = timing.get("camera_capture_monotonic_s", {})
        if not all(name in observation and name in stamps for name in self.cameras):
            return self._ready("camera", "Waiting for complete camera frames.")
        values = [float(stamps[name]) for name in self.cameras]
        if not all(math.isfinite(value) for value in values) or max(values) - min(values) > 0.05:
            return self._ready("camera", "Waiting for aligned camera frames.")
        if any(stamps[name] == self._camera_stamps.get(name) for name in self.cameras):
            return False
        state_time = timing.get("state_sample_finished_monotonic_s")
        if state_time is not None and abs(statistics.median(values) - state_time) > 0.1:
            return self._ready("camera", "Waiting for camera/state alignment.")
        self._camera_stamps = {name: stamps[name] for name in self.cameras}
        return self._ready("camera", None)


class RecordingCadence:
    """Sample at dataset FPS without creating catch-up frames after a pause."""

    def __init__(self, fps):
        self.interval = 1.0 / fps
        self.next_sample = None

    def ready(self, now):
        if self.next_sample is not None and now + 1e-9 < self.next_sample:
            return False
        if self.next_sample is None or now - self.next_sample >= self.interval:
            self.next_sample = now + self.interval
        else:
            self.next_sample += self.interval
        return True
