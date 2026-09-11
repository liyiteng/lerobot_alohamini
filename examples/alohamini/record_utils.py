#!/usr/bin/env python3

import logging
import time
from collections.abc import Callable
from typing import Any

from lerobot.datasets import LeRobotDataset, safe_stop_image_writer
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_visualization_data

try:
    from .safety_utils import RecordingGate, SafetyRecorder, safety_snapshot
except ImportError:
    from safety_utils import RecordingGate, SafetyRecorder, safety_snapshot


@safe_stop_image_writer
def record_loop(
    robot: Any,
    events: dict,
    fps: int,
    leader_arm: Any,
    keyboard: Any,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset: LeRobotDataset | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    timing_callback: Callable[[dict[str, float]], None] | None = None,
    display_data: bool = False,
) -> None:
    """AlohaMini-specific bimanual recording loop with optional timing diagnostics."""
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")
    if control_time_s is None:
        raise ValueError("control_time_s must be provided")

    control_interval = 1 / fps
    start_episode_t = time.perf_counter()
    timestamp = 0.0
    fps_report_count = 0
    captured_report_count = 0
    fps_report_start_t = start_episode_t
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

    with SafetyRecorder(dataset) as safety_recorder:
        gate = RecordingGate(robot, dataset, safety_recorder)
        frames_written = 0
        target_frames = max(1, round(control_time_s * fps))
        while frames_written < target_frames if dataset is not None else timestamp < control_time_s:
            start_loop_t = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                break

            obs = robot.get_observation()
            if not gate.state_ready():
                precise_sleep(max(0.0, control_interval - (time.perf_counter() - start_loop_t)))
                timestamp = time.perf_counter() - start_episode_t
                continue
            sampled_safety = safety_snapshot(robot)
            observation_done_t = time.perf_counter()
            for name, value_ms in getattr(robot, "logs", {}).get("observation_timing_ms", {}).items():
                timing_totals_s[name] = timing_totals_s.get(name, 0.0) + value_ms / 1000

            obs_processed = robot_observation_processor(obs)
            observation_processing_done_t = time.perf_counter()

            record_frame = dataset is not None and gate.frame_ready(obs_processed)
            if record_frame:
                observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            frame_build_done_t = time.perf_counter()

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

            if robot.send_action(robot_action_to_send) == {}:
                precise_sleep(max(0.0, control_interval - (time.perf_counter() - start_loop_t)))
                timestamp = time.perf_counter() - start_episode_t
                continue
            send_action_done_t = time.perf_counter()

            if record_frame:
                action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
                dataset.add_frame({**observation_frame, **action_frame, "task": single_task})
                frames_written += 1
                captured_report_count += 1
                safety_recorder.write(
                    safety=sampled_safety,
                    requested_action=action_values,
                    issued_command=dict(getattr(robot, "last_sent_command", {})),
                    host_timing=dict(getattr(robot, "latest_host_timing", {})),
                )
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
                    "AlohaMini record loop is running slower (%.1f Hz) than the target FPS (%d Hz).",
                    1 / work_duration_s,
                    fps,
                )
            precise_sleep(max(sleep_time_s, 0.0))
            loop_done_t = time.perf_counter()

            timing_totals_s["observation"] += observation_done_t - start_loop_t
            timing_totals_s["observation_processing"] += observation_processing_done_t - observation_done_t
            timing_totals_s["frame_build"] += frame_build_done_t - observation_processing_done_t
            timing_totals_s["teleop"] += teleop_done_t - frame_build_done_t
            timing_totals_s["send_action"] += send_action_done_t - teleop_done_t
            timing_totals_s["dataset_write"] += dataset_write_done_t - send_action_done_t
            timing_totals_s["display"] += work_done_t - dataset_write_done_t
            timing_totals_s["sleep"] += loop_done_t - work_done_t
            timing_totals_s["loop"] += loop_done_t - start_loop_t

            fps_report_count += 1
            report_now_t = time.perf_counter()
            fps_report_elapsed_s = report_now_t - fps_report_start_t
            if fps_report_elapsed_s >= 1.0:
                capture_fps = captured_report_count / fps_report_elapsed_s
                if timing_callback is not None:
                    timing_callback(
                        {
                            "capture_fps": capture_fps,
                            **{
                                name: total_s * 1000 / fps_report_count
                                for name, total_s in timing_totals_s.items()
                            },
                        }
                    )
                for name in timing_totals_s:
                    timing_totals_s[name] = 0.0
                fps_report_count = 0
                captured_report_count = 0
                fps_report_start_t = report_now_t

            timestamp = time.perf_counter() - start_episode_t
