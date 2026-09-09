#!/usr/bin/env python

"""Camera-only ZMQ stream kept outside the AlohaMini Host control loop."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import zmq

CAMERA_STREAM_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CameraSnapshot:
    frame: Any
    capture_monotonic_s: float


def read_camera_snapshot(camera: Any, max_age_ms: int) -> CameraSnapshot:
    """Read an exact frame/timestamp pair without consuming the camera event."""
    frame = camera.read_latest(max_age_ms=max_age_ms)
    timestamp = getattr(camera, "latest_timestamp", None)
    frame_lock = getattr(camera, "frame_lock", None)
    if frame_lock is not None:
        with frame_lock:
            latest_frame = getattr(camera, "latest_frame", None)
            latest_timestamp = getattr(camera, "latest_timestamp", None)
        if latest_frame is not None and latest_timestamp is not None:
            frame = latest_frame
            timestamp = latest_timestamp
    if timestamp is None:
        raise RuntimeError("camera does not expose a capture timestamp")
    return CameraSnapshot(frame=frame, capture_monotonic_s=float(timestamp))


def encode_camera_stream_message(
    camera_name: str,
    snapshot: CameraSnapshot,
    sequence: int,
    jpeg_quality: int,
    *,
    host_monotonic_s: float | None = None,
    host_unix_ns: int | None = None,
) -> tuple[list[bytes], float]:
    """Build ``[topic, metadata_json, jpeg]`` for one newly captured frame."""
    host_monotonic_s = time.perf_counter() if host_monotonic_s is None else host_monotonic_s
    host_unix_ns = time.time_ns() if host_unix_ns is None else host_unix_ns
    capture_unix_ns = host_unix_ns - round(
        (host_monotonic_s - snapshot.capture_monotonic_s) * 1e9
    )
    encode_started = time.perf_counter()
    # LeRobot OpenCVCamera returns RGB by default, while cv2.imencode interprets
    # three-channel input as BGR. This stream is consumed as a normal JPEG by
    # ROS, so make the color-space boundary explicit. The separate 5556
    # teleoperation/recording path is intentionally unchanged.
    bgr_frame = cv2.cvtColor(snapshot.frame, cv2.COLOR_RGB2BGR)
    encoded, buffer = cv2.imencode(
        ".jpg",
        bgr_frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    encode_ms = (time.perf_counter() - encode_started) * 1e3
    if not encoded:
        raise RuntimeError(f"failed to JPEG encode camera {camera_name}")
    height, width = snapshot.frame.shape[:2]
    metadata = {
        "schema_version": CAMERA_STREAM_SCHEMA_VERSION,
        "camera_name": camera_name,
        "sequence": int(sequence),
        "encoding": "jpeg",
        "width": int(width),
        "height": int(height),
        "capture_monotonic_s": snapshot.capture_monotonic_s,
        "capture_unix_ns": int(capture_unix_ns),
        "host_clock_reference": {
            "monotonic_s": float(host_monotonic_s),
            "unix_ns": int(host_unix_ns),
        },
    }
    return [
        f"camera/{camera_name}".encode(),
        json.dumps(metadata, separators=(",", ":")).encode("utf-8"),
        buffer.tobytes(),
    ], encode_ms


class CameraStreamPublisher:
    """Publish each unique camera capture from a socket-owning background thread."""

    def __init__(
        self,
        cameras: dict[str, Any],
        *,
        port: int,
        jpeg_quality: int = 70,
        max_age_ms: int = 500,
        poll_interval_s: float = 0.002,
    ) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if port <= 0 or max_age_ms <= 0 or poll_interval_s <= 0.0:
            raise ValueError("camera stream port, max age, and poll interval must be positive")
        self.cameras = cameras
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.max_age_ms = int(max_age_ms)
        self.poll_interval_s = float(poll_interval_s)
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None
        self._stats_lock = threading.Lock()
        self._stats: dict[str, float] = {
            "published": 0.0,
            "dropped": 0.0,
            "errors": 0.0,
            "encode_ms_total": 0.0,
        }

    def start(self, timeout_s: float = 2.0) -> None:
        if self._thread is not None:
            raise RuntimeError("camera stream publisher is already started")
        self._thread = threading.Thread(
            target=self._run,
            name="alohamini-camera-stream",
            daemon=True,
        )
        self._thread.start()
        if not self._started_event.wait(timeout_s):
            raise TimeoutError("camera stream publisher did not start")
        if self._startup_error is not None:
            raise RuntimeError("camera stream publisher failed to start") from self._startup_error

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                logging.warning("Camera stream publisher did not stop within %.1fs", timeout_s)
        self._thread = None

    def stats(self) -> dict[str, float]:
        with self._stats_lock:
            result = dict(self._stats)
        published = result["published"]
        result["average_encode_ms"] = (
            result["encode_ms_total"] / published if published else 0.0
        )
        return result

    def _increment_stats(self, **values: float) -> None:
        with self._stats_lock:
            for name, value in values.items():
                self._stats[name] += value

    def _run(self) -> None:
        context = zmq.Context()
        # XPUB is wire-compatible with SUB and exposes subscription events. Avoid
        # spending Pi CPU on JPEG encoding while no ROS camera consumer exists.
        socket = context.socket(zmq.XPUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, max(4, len(self.cameras) * 2))
        socket.setsockopt(zmq.XPUB_VERBOSE, 1)
        try:
            socket.bind(f"tcp://*:{self.port}")
        except BaseException as error:
            self._startup_error = error
            self._started_event.set()
            socket.close(linger=0)
            context.term()
            return
        self._started_event.set()
        sequences = dict.fromkeys(self.cameras, 0)
        last_timestamps: dict[str, float] = {}
        last_warning: dict[str, float] = {}
        subscriptions: set[bytes] = set()
        try:
            while not self._stop_event.is_set():
                while True:
                    try:
                        subscription = socket.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    if not subscription:
                        continue
                    prefix = subscription[1:]
                    if subscription[0] == 1:
                        subscriptions.add(prefix)
                    else:
                        subscriptions.discard(prefix)
                published_any = False
                for camera_name, camera in self.cameras.items():
                    topic = f"camera/{camera_name}".encode()
                    if not any(topic.startswith(prefix) for prefix in subscriptions):
                        continue
                    try:
                        snapshot = read_camera_snapshot(camera, self.max_age_ms)
                        if last_timestamps.get(camera_name) == snapshot.capture_monotonic_s:
                            continue
                        sequences[camera_name] += 1
                        parts, encode_ms = encode_camera_stream_message(
                            camera_name,
                            snapshot,
                            sequences[camera_name],
                            self.jpeg_quality,
                        )
                        socket.send_multipart(parts, flags=zmq.NOBLOCK)
                        last_timestamps[camera_name] = snapshot.capture_monotonic_s
                        self._increment_stats(published=1.0, encode_ms_total=encode_ms)
                        published_any = True
                    except zmq.Again:
                        self._increment_stats(dropped=1.0)
                    except Exception as error:
                        self._increment_stats(errors=1.0)
                        now = time.monotonic()
                        if now - last_warning.get(camera_name, 0.0) >= 1.0:
                            logging.warning("Camera stream %s unavailable: %s", camera_name, error)
                            last_warning[camera_name] = now
                if not published_any:
                    self._stop_event.wait(self.poll_interval_s)
        finally:
            socket.close(linger=0)
            context.term()
