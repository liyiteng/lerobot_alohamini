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


class GripperGestureDetector:
    """Detects rapid double-pinch gestures on leader arm grippers for hands-free recording."""

    def __init__(self, window_s: float = 0.85):
        self.window_s = window_s
        self.r_state = "IDLE"
        self.r_pinches: list[float] = []
        self.l_state = "IDLE"
        self.l_pinches: list[float] = []

    def update(self, r_val: float | None, l_val: float | None) -> tuple[bool, bool]:
        now = time.perf_counter()
        r_triggered = False
        l_triggered = False

        def _process_arm(val: float | None, state: str, pinches: list[float]) -> tuple[bool, str]:
            if val is None:
                return False, state
            # Retain only pinches within window
            valid_pinches = [t for t in pinches if now - t <= self.window_s]
            pinches.clear()
            pinches.extend(valid_pinches)

            # Recognize pinch whether 0 is closed or 100 is closed
            is_pinched = (val <= 30.0) or (val >= 90.0)
            is_released = (35.0 < val < 85.0)

            triggered = False
            if state == "IDLE" and is_pinched:
                state = "PINCHED"
                pinches.append(now)
                if len(pinches) >= 2:
                    triggered = True
                    pinches.clear()
            elif state == "PINCHED" and is_released:
                state = "IDLE"

            return triggered, state

        r_trig, self.r_state = _process_arm(r_val, self.r_state, self.r_pinches)
        l_trig, self.l_state = _process_arm(l_val, self.l_state, self.l_pinches)
        return r_trig, l_trig


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
    enable_hands_free: bool = True,
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
    gesture_detector = GripperGestureDetector() if enable_hands_free else None

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        observation_done_t = time.perf_counter()
        for name, value_ms in getattr(robot, "logs", {}).get("observation_timing_ms", {}).items():
            timing_totals_s[name] = timing_totals_s.get(name, 0.0) + value_ms / 1000

        obs_processed = robot_observation_processor(obs)
        observation_processing_done_t = time.perf_counter()

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
        frame_build_done_t = time.perf_counter()

        arm_action = {f"arm_{key}": value for key, value in leader_arm.get_action().items()}

        if enable_hands_free and gesture_detector is not None:
            r_grip = arm_action.get("arm_right_gripper.pos")
            l_grip = arm_action.get("arm_left_gripper.pos")
            r_trig, l_trig = gesture_detector.update(r_grip, l_grip)
            if r_trig:
                print("\r\n\x1b[1;36m✔ [免脱手手势: 右夹爪双击] 完成当前集录制，进入保存与复位...\x1b[0m", flush=True)
                if dataset is not None and getattr(dataset, "writer", None) is not None:
                    buf = dataset.writer.episode_buffer
                    if isinstance(buf, dict) and buf.get("size", 0) > 15:
                        drop_count = min(int(0.65 * fps), buf["size"] - 5)
                        if drop_count > 0:
                            for k in buf:
                                if isinstance(buf[k], list) and len(buf[k]) >= drop_count:
                                    buf[k] = buf[k][:-drop_count]
                            buf["size"] = max(0, buf["size"] - drop_count)
                events["exit_early"] = True
                break
            elif l_trig:
                print("\r\n\x1b[1;33m⚠ [免脱手手势: 左夹爪双击] 本集已作废，清空并重新开始...\x1b[0m", flush=True)
                events["rerecord_episode"] = True
                events["exit_early"] = True
                break

        keyboard_action = keyboard.get_action() if getattr(keyboard, "is_connected", True) else {}
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

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            dataset.add_frame({**observation_frame, **action_frame, "task": single_task})
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
        timing_totals_s["observation_processing"] += (
            observation_processing_done_t - observation_done_t
        )
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
            capture_fps = fps_report_count / fps_report_elapsed_s
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
            fps_report_start_t = report_now_t

        timestamp = time.perf_counter() - start_episode_t
