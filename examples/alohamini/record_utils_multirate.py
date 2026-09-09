#!/usr/bin/env python3

import logging
import math
import statistics
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from lerobot.datasets import LeRobotDataset, safe_stop_image_writer
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_visualization_data


@dataclass(frozen=True)
class ControlSample:
    """One coherent processed state and teleoperation action sample."""

    observation: dict[str, Any]
    action: dict[str, Any]
    state_timestamp: float | None


class ControlSampleBuffer:
    """Keep enough Host-timestamped control history to align delayed camera frames."""

    def __init__(self, max_samples: int) -> None:
        if max_samples < 2:
            raise ValueError("max_samples must be at least 2")
        self._samples: deque[ControlSample] = deque(maxlen=max_samples)

    def append(self, sample: ControlSample) -> None:
        self._samples.append(sample)

    def nearest(self, timestamp: float, *, max_error_s: float) -> tuple[ControlSample, float]:
        timestamped = [sample for sample in self._samples if sample.state_timestamp is not None]
        if not timestamped:
            if not self._samples:
                raise RuntimeError("no control sample is available for dataset alignment")
            return self._samples[-1], math.nan

        sample = min(
            timestamped,
            key=lambda item: abs(float(item.state_timestamp) - timestamp),
        )
        error_s = abs(float(sample.state_timestamp) - timestamp)
        if error_s > max_error_s:
            raise RuntimeError(
                "camera/state alignment exceeded the allowed error: "
                f"{error_s * 1000:.1f} ms > {max_error_s * 1000:.1f} ms"
            )
        return sample, error_s


class FreshCameraGate:
    """Accept complete, fresh multi-camera snapshots and diagnose stalled cameras."""

    def __init__(
        self,
        camera_names: tuple[str, ...],
        *,
        started_at: float,
        stall_timeout_s: float,
        max_skew_s: float,
    ) -> None:
        if not camera_names:
            raise ValueError("FreshCameraGate requires at least one camera")
        if stall_timeout_s <= 0.0 or max_skew_s < 0.0:
            raise ValueError("camera timeout must be positive and skew must be non-negative")
        self.camera_names = camera_names
        self.stall_timeout_s = float(stall_timeout_s)
        self.max_skew_s = float(max_skew_s)
        self._last_seen: dict[str, float] = {}
        self._last_recorded: dict[str, float] = {}
        self._last_advanced_at = dict.fromkeys(camera_names, float(started_at))
        self.last_skew_s = 0.0

    def observe(self, timestamps: dict[str, float], *, now: float) -> float | None:
        for name in self.camera_names:
            if name not in timestamps:
                continue
            timestamp = float(timestamps[name])
            if self._last_seen.get(name) != timestamp:
                self._last_seen[name] = timestamp
                self._last_advanced_at[name] = now

        stalled = [
            name
            for name in self.camera_names
            if now - self._last_advanced_at[name] > self.stall_timeout_s
        ]
        if stalled:
            raise RuntimeError(
                "camera capture timestamp stalled for "
                f"> {self.stall_timeout_s:.1f} s: {', '.join(stalled)}"
            )

        if not all(name in timestamps for name in self.camera_names):
            return None
        if not all(
            float(timestamps[name]) != self._last_recorded.get(name)
            for name in self.camera_names
        ):
            return None

        values = [float(timestamps[name]) for name in self.camera_names]
        self.last_skew_s = max(values) - min(values)
        if self.last_skew_s > self.max_skew_s:
            raise RuntimeError(
                "multi-camera capture skew exceeded the allowed limit: "
                f"{self.last_skew_s * 1000:.1f} ms > {self.max_skew_s * 1000:.1f} ms"
            )

        self._last_recorded = {
            name: float(timestamps[name]) for name in self.camera_names
        }
        return float(statistics.median(values))


def _advance_deadline(deadline: float, interval: float, now: float) -> float:
    while deadline <= now:
        deadline += interval
    return deadline


def _dataset_camera_names(robot: Any, dataset: LeRobotDataset | None) -> tuple[str, ...]:
    configured = tuple(getattr(robot, "_cameras_ft", {}))
    if dataset is None:
        return configured

    prefix = f"{OBS_STR}.images."
    requested = tuple(
        key.removeprefix(prefix)
        for key, feature in dataset.features.items()
        if key.startswith(prefix) and feature.get("dtype") in {"image", "video"}
    )
    unknown = [name for name in requested if name not in configured]
    if unknown:
        raise ValueError(
            "dataset requests cameras that are not configured on the robot: "
            f"{', '.join(unknown)}"
        )
    return requested


@safe_stop_image_writer
def record_loop(
    robot: Any,
    events: dict,
    fps: int,
    leader_arm: Any,
    keyboard: Any,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset: LeRobotDataset | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    timing_callback: Callable[[dict[str, float]], None] | None = None,
    display_data: bool = False,
    control_fps: int = 50,
    sample_buffer_s: float = 1.0,
    camera_stall_timeout_s: float = 1.0,
    max_camera_skew_s: float = 0.05,
    max_state_alignment_s: float = 0.10,
    min_capture_rate_ratio: float = 0.90,
) -> None:
    """Run fast control while recording complete, fresh camera snapshots at ``fps``.

    LeRobot assigns timestamps as ``frame_index / dataset.fps``. For a camera
    dataset this loop records the requested number of fresh frames, allowing a
    small wall-time overrun instead of silently shortening the dataset timeline
    when a nominal 30 Hz camera actually produces 29.5 Hz.
    """
    if fps <= 0 or control_fps <= 0:
        raise ValueError("fps and control_fps must be positive")
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")
    if control_time_s is None or control_time_s <= 0:
        raise ValueError("control_time_s must be positive")
    if control_fps < fps:
        raise ValueError(f"control_fps must be at least dataset fps ({control_fps} < {fps}).")
    if sample_buffer_s <= 0.0 or max_state_alignment_s <= 0.0:
        raise ValueError("sample buffer and state alignment limits must be positive")
    if not 0.0 < min_capture_rate_ratio <= 1.0:
        raise ValueError("min_capture_rate_ratio must be in (0, 1]")

    # Remove camera requests prefetched during setup/reset. Priming state-only
    # requests prevents a burst of stale camera responses at the episode start.
    prime_requests = getattr(robot, "prime_observation_request_window", None)
    if dataset is not None and callable(prime_requests):
        prime_requests(include_cameras=False)

    control_interval = 1.0 / control_fps
    dataset_interval = 1.0 / fps
    start_episode_t = time.perf_counter()
    next_camera_request_t = start_episode_t
    next_state_only_sample_t = start_episode_t
    expected_cameras = _dataset_camera_names(robot, dataset)
    sample_buffer = ControlSampleBuffer(
        max_samples=max(2, math.ceil(control_fps * sample_buffer_s))
    )
    camera_gate = (
        FreshCameraGate(
            expected_cameras,
            started_at=start_episode_t,
            stall_timeout_s=camera_stall_timeout_s,
            max_skew_s=max_camera_skew_s,
        )
        if dataset is not None and expected_cameras
        else None
    )
    target_dataset_frames = (
        max(1, round(control_time_s * fps)) if dataset is not None else None
    )
    max_recording_wall_time_s = (
        control_time_s / min_capture_rate_ratio + control_interval * 2
        if dataset is not None and expected_cameras
        else control_time_s + control_interval * 2
    )

    control_report_count = 0
    dataset_report_count = 0
    dataset_frames_written = 0
    fps_report_start_t = start_episode_t
    exited_early = False
    last_alignment_error_s = math.nan
    timing_totals_s = {
        "observation": 0.0,
        "observation_processing": 0.0,
        "frame_build": 0.0,
        "teleop": 0.0,
        "send_action": 0.0,
        "dataset_write": 0.0,
        "display": 0.0,
        "sleep": 0.0,
        "loop": 0.0,
    }

    while True:
        start_loop_t = time.perf_counter()
        elapsed_s = start_loop_t - start_episode_t
        if target_dataset_frames is None:
            if elapsed_s >= control_time_s:
                break
        elif dataset_frames_written >= target_dataset_frames:
            break
        elif elapsed_s > max_recording_wall_time_s:
            raise RuntimeError(
                "could not collect the requested number of fresh dataset frames: "
                f"{dataset_frames_written}/{target_dataset_frames} in {elapsed_s:.2f} s"
            )

        if events["exit_early"]:
            events["exit_early"] = False
            exited_early = True
            break

        request_cameras = bool(expected_cameras) and start_loop_t >= next_camera_request_t
        if request_cameras:
            next_camera_request_t = _advance_deadline(
                next_camera_request_t, dataset_interval, start_loop_t
            )
        state_only_sample_due = not expected_cameras and start_loop_t >= next_state_only_sample_t
        if state_only_sample_due:
            next_state_only_sample_t = _advance_deadline(
                next_state_only_sample_t, dataset_interval, start_loop_t
            )

        obs = robot.get_observation(include_cameras=request_cameras)
        observation_done_t = time.perf_counter()
        for name, value_ms in getattr(robot, "logs", {}).get("observation_timing_ms", {}).items():
            timing_totals_s[name] = timing_totals_s.get(name, 0.0) + value_ms / 1000

        obs_processed = robot_observation_processor(obs)
        observation_processing_done_t = time.perf_counter()

        arm_action = {f"arm_{key}": value for key, value in leader_arm.get_action().items()}
        keyboard_action = keyboard.get_action()
        action = {
            **arm_action,
            **robot._from_keyboard_to_base_action(keyboard_action),
            **robot._from_keyboard_to_lift_action(keyboard_action),
        }
        action_values = teleop_action_processor((action, obs))
        robot_action_to_send = robot_action_processor((action_values, obs))
        teleop_done_t = time.perf_counter()

        robot.send_action(robot_action_to_send)
        send_action_done_t = time.perf_counter()

        host_timing = dict(getattr(robot, "latest_host_timing", {}))
        state_started_t = host_timing.get("state_sample_started_monotonic_s")
        state_finished_t = host_timing.get("state_sample_finished_monotonic_s")
        state_timestamp = (
            (float(state_started_t) + float(state_finished_t)) / 2.0
            if state_started_t is not None and state_finished_t is not None
            else None
        )
        state_observation = {
            key: value for key, value in obs_processed.items() if key not in expected_cameras
        }
        current_sample = ControlSample(
            observation=state_observation,
            action=dict(action_values),
            state_timestamp=state_timestamp,
        )
        sample_buffer.append(current_sample)

        selected_sample: ControlSample | None = None
        alignment_error_s = math.nan
        if dataset is not None and not expected_cameras and state_only_sample_due:
            selected_sample = current_sample
        elif dataset is not None and camera_gate is not None:
            camera_timestamps = {
                str(name): float(value)
                for name, value in host_timing.get("camera_capture_monotonic_s", {}).items()
            }
            camera_timestamp = camera_gate.observe(camera_timestamps, now=start_loop_t)
            if camera_timestamp is not None:
                selected_sample, alignment_error_s = sample_buffer.nearest(
                    camera_timestamp,
                    max_error_s=max_state_alignment_s,
                )
                last_alignment_error_s = alignment_error_s

        frame_build_started_t = time.perf_counter()
        if dataset is not None and selected_sample is not None:
            aligned_observation = dict(selected_sample.observation)
            for camera_name in expected_cameras:
                if camera_name not in obs_processed:
                    raise RuntimeError(
                        f"fresh timestamp was received without image data for {camera_name}"
                    )
                aligned_observation[camera_name] = obs_processed[camera_name]
            observation_frame = build_dataset_frame(
                dataset.features, aligned_observation, prefix=OBS_STR
            )
            action_frame = build_dataset_frame(
                dataset.features, selected_sample.action, prefix=ACTION
            )
            dataset.add_frame({**observation_frame, **action_frame, "task": single_task})
            dataset_frames_written += 1
            dataset_report_count += 1
        dataset_write_done_t = time.perf_counter()

        if display_data:
            log_visualization_data(
                "rerun",
                observation=obs_processed,
                action=action_values,
            )
        work_done_t = time.perf_counter()
        work_duration_s = work_done_t - start_loop_t
        sleep_time_s = control_interval - work_duration_s
        if sleep_time_s < 0:
            logging.warning(
                "AlohaMini record loop is running slower (%.1f Hz) than the target control rate (%d Hz).",
                1 / work_duration_s,
                control_fps,
            )
        precise_sleep(max(sleep_time_s, 0.0))
        loop_done_t = time.perf_counter()

        timing_totals_s["observation"] += observation_done_t - start_loop_t
        timing_totals_s["observation_processing"] += (
            observation_processing_done_t - observation_done_t
        )
        timing_totals_s["frame_build"] += dataset_write_done_t - frame_build_started_t
        timing_totals_s["teleop"] += teleop_done_t - observation_processing_done_t
        timing_totals_s["send_action"] += send_action_done_t - teleop_done_t
        timing_totals_s["dataset_write"] += dataset_write_done_t - frame_build_started_t
        timing_totals_s["display"] += work_done_t - dataset_write_done_t
        timing_totals_s["sleep"] += loop_done_t - work_done_t
        timing_totals_s["loop"] += loop_done_t - start_loop_t

        control_report_count += 1
        report_now_t = time.perf_counter()
        fps_report_elapsed_s = report_now_t - fps_report_start_t
        if fps_report_elapsed_s >= 1.0:
            control_rate = control_report_count / fps_report_elapsed_s
            dataset_rate = dataset_report_count / fps_report_elapsed_s
            if timing_callback is not None:
                timing_callback(
                    {
                        "capture_fps": dataset_rate,
                        "control_fps": control_rate,
                        "camera_skew_ms": (
                            camera_gate.last_skew_s * 1000 if camera_gate is not None else 0.0
                        ),
                        "state_alignment_ms": (
                            last_alignment_error_s * 1000
                            if math.isfinite(last_alignment_error_s)
                            else 0.0
                        ),
                        **{
                            name: total_s * 1000 / control_report_count
                            for name, total_s in timing_totals_s.items()
                        },
                    }
                )
            for name in timing_totals_s:
                timing_totals_s[name] = 0.0
            control_report_count = 0
            dataset_report_count = 0
            fps_report_start_t = report_now_t

    if (
        dataset is not None
        and not exited_early
        and dataset_frames_written != target_dataset_frames
    ):
        raise RuntimeError(
            "recording ended with an incomplete dataset frame count: "
            f"{dataset_frames_written}/{target_dataset_frames}"
        )
