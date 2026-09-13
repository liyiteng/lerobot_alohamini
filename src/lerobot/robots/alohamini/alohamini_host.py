#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import zmq

from .alohamini import AlohaMini
from .camera_stream import CameraStreamPublisher
from .command_owner import CommandOwner
from .config_alohamini import AlohaMiniConfig, AlohaMiniHostConfig


class AlohaMiniHost:
    def __init__(self, config: AlohaMiniHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        # Observations are request-driven with a small bounded request window. The Host
        # consumes at most one credit per control loop, preventing unbounded accumulation
        # while allowing transport latency to overlap subsequent observation cycles.
        self.zmq_observation_socket = self.zmq_context.socket(zmq.ROUTER)
        self.zmq_observation_socket.setsockopt(zmq.SNDHWM, config.observation_request_window)
        self.zmq_observation_socket.setsockopt(zmq.RCVHWM, config.observation_request_window)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def _jsonable(value):
    """Convert numpy scalars to JSON-native values without touching normal Python values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def build_observation_multipart(
    observation: dict,
    camera_keys,
    encoded_camera_keys=None,
    encoding_timings_ms: dict[str, float] | None = None,
) -> list[bytes]:
    """Encode state as JSON and selected camera images as binary JPEG frames.

    ``camera_keys`` identifies all image fields that must be excluded from JSON.
    ``encoded_camera_keys`` selects which images to append. Passing an empty iterable
    produces the strict two-frame ROUTER response required by ROS ``:state`` requests.
    """
    camera_keys = tuple(camera_keys)
    encoded_camera_keys = camera_keys if encoded_camera_keys is None else tuple(encoded_camera_keys)
    state_observation = {
        key: _jsonable(value) for key, value in observation.items() if key not in camera_keys
    }
    state_observation["_image_encoding"] = "jpeg"

    parts = [json.dumps(state_observation).encode("utf-8")]
    image_names = []
    for cam_key in encoded_camera_keys:
        frame = observation.get(cam_key)
        if frame is None:
            continue
        encode_started = time.perf_counter()
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if encoding_timings_ms is not None:
            encoding_timings_ms[f"encode_{cam_key}"] = (time.perf_counter() - encode_started) * 1e3
        if not ret:
            logging.warning("Failed to JPEG encode camera frame %s.", cam_key)
            continue
        image_names.append(cam_key)
        parts.extend([cam_key.encode("utf-8"), buffer.tobytes()])

    state_observation["_images"] = image_names
    parts[0] = json.dumps(state_observation).encode("utf-8")
    return parts


def build_robot_metadata(robot: AlohaMini) -> dict:
    """Describe the connected Host hardware for ROS unit and model validation."""
    motors = {}
    for bus in (robot.left_bus, robot.right_bus):
        if bus is None:
            continue
        for name, motor in bus.motors.items():
            calibration = bus.calibration.get(name)
            if calibration is None:
                continue
            motors[name] = {
                "id": int(motor.id),
                "model": motor.model,
                "normalization": motor.norm_mode.value,
                "drive_mode": int(calibration.drive_mode),
                "range_min": int(calibration.range_min),
                "range_max": int(calibration.range_max),
            }

    metadata = {
        "schema_version": 1,
        "robot_model": robot.config.robot_model,
        "motors": motors,
        "cameras": list(robot.cameras),
    }
    if getattr(robot, "lift", None) is not None:
        metadata["lift_axis"] = {
            "soft_min_mm": float(robot.lift.cfg.soft_min_mm),
            "soft_max_mm": float(robot.lift.cfg.soft_max_mm),
            "descent_floor_mm": float(robot.lift.cfg.descent_floor_mm),
        }
    return metadata


def observation_request_includes_cameras(request_token: bytes | None) -> bool:
    """Keep legacy/plain requests full while allowing ROS state-only requests."""
    return request_token is not None and not request_token.endswith(b":state")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def main():
    parser = argparse.ArgumentParser(description="Run AlohaMini host process")
    parser.add_argument(
        "--robot_model",
        type=str,
        default="alohamini1",
        choices=["alohamini1", "alohamini2", "alohamini2pro"],
        help=(
            "Robot model — drives follower arm profile, base motors, lift motor, lead screw pitch, "
            "and chassis kinematics.\n"
            "  alohamini1   : so-arm-5dof,         base sts3215, lift sts3215, lead=84 mm/rev,  "
            "wheel=0.05m, radius=0.125m\n"
            "  alohamini2   : am-follower-6dof,    base sts3215, lift sts3095, lead=131 mm/rev, "
            "wheel=0.063m, radius=0.195m\n"
            "  alohamini2pro: am-follower-6dof-hd, base sts3250, lift sts3095, lead=131 mm/rev, "
            "wheel=0.063m, radius=0.195m"
        ),
    )
    parser.add_argument(
        "--no_follower",
        action="store_true",
        help="Do not connect follower arms, only operate the base and lift. Use together with --no_leader on the teleoperate side.",
    )
    parser.add_argument(
        "--profile_timing",
        "--profile-timing",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Print average Host, motor, camera, JPEG, and network timings once per second (default: false)."
        ),
    )
    parser.add_argument(
        "--camera-stream",
        action="store_true",
        help="Publish the optional non-blocking ROS camera stream on port 5557.",
    )
    parser.add_argument(
        "--camera-stream-port",
        type=int,
        default=None,
        help="Override the dedicated ROS camera PUB port.",
    )
    args = parser.parse_args()

    logging.info("Configuring AlohaMini")
    robot_config = AlohaMiniConfig()
    robot_config.id = "AlohaMiniRobot"
    robot_config.robot_model = args.robot_model
    robot_config.no_follower = args.no_follower
    if args.no_follower:
        logging.info("no_follower mode: follower arms will not connect, only base and lift operate.")
    robot = AlohaMini(robot_config)

    logging.info("Connecting AlohaMini")
    robot.connect()
    robot_metadata = build_robot_metadata(robot)

    logging.info("Starting HostAgent")
    host_config = AlohaMiniHostConfig()
    host_config.camera_stream_enabled = args.camera_stream
    if args.camera_stream_port is not None:
        host_config.port_zmq_camera_stream = args.camera_stream_port
    host = AlohaMiniHost(host_config)
    camera_stream = (
        CameraStreamPublisher(
            robot.cameras,
            port=host_config.port_zmq_camera_stream,
            jpeg_quality=host_config.camera_stream_jpeg_quality,
            max_age_ms=host_config.camera_stream_max_age_ms,
        )
        if host_config.camera_stream_enabled and robot.cameras
        else None
    )
    if camera_stream is not None:
        try:
            camera_stream.start()
        except Exception:
            host.disconnect()
            robot.disconnect()
            raise
        logging.info(
            "Camera stream publishing on tcp://*:%d",
            host_config.port_zmq_camera_stream,
        )
    jpeg_executor = ThreadPoolExecutor(
        max_workers=max(1, len(robot.cameras)),
        thread_name_prefix="alohamini-jpeg",
    )
    pending_observation_responses: deque[
        tuple[bytes, bytes, list[bytes] | Future[list[bytes]], dict[str, float]]
    ] = deque()
    max_pending_observation_responses = 8

    last_cmd_time = time.monotonic()
    watchdog_active = False
    watchdog_events = 0
    last_command_metadata = {}
    command_owner = CommandOwner()
    target_source = "none"
    has_received_command = False
    latest_action: dict[str, float] = {}
    last_sent_action: dict[str, float] = {}
    logging.info("Waiting for commands...")

    try:
        # Business logic
        start = time.perf_counter()
        duration = 0
        timing_report_start_t = start
        timing_loop_count = 0
        timing_totals_ms: dict[str, float] = {}
        timing_command_count = 0
        action_timing_totals_ms: dict[str, float] = {}
        while duration < host.connection_time_s:
            loop_start_t = time.perf_counter()
            command_received = False

            # Poll the request before sampling the robot. State-only ROS clients do not
            # put camera retrieval in the control critical path; full LeRobot clients
            # still receive the latest frames in their requested cycle.
            request_identity = None
            request_token = None
            if len(pending_observation_responses) < max_pending_observation_responses:
                try:
                    request_parts = host.zmq_observation_socket.recv_multipart(flags=zmq.NOBLOCK)
                    request_identity = request_parts[0]
                    request_token = request_parts[-1]
                except zmq.Again:
                    pass
            request_poll_done_t = time.perf_counter()
            include_cameras = observation_request_includes_cameras(request_token)

            # One feedback snapshot owns the complete observe -> act cycle.
            # send_action() reuses its position/current values for safety limits.
            last_observation = robot.get_observation(include_cameras=include_cameras)
            # send_action() consumes/clears this cycle's cached feedback. Preserve only
            # the small current snapshot needed by the once-per-second tracking report.
            tracking_currents_ma = {
                motor: float(raw) * 6.5 for motor, raw in robot._feedback_currents_raw.items()
            }
            observation_done_t = time.perf_counter()

            # Expire the lease before reading queued commands, including the same owner's.
            watchdog_tripped = (
                has_received_command
                and time.monotonic() - last_cmd_time > host.watchdog_timeout_ms / 1000
                and not watchdog_active
            )
            if watchdog_tripped:
                logging.warning("Host command watchdog expired; stopping motion")
                watchdog_active = True
                watchdog_events += 1
                target_source = "watchdog"
                last_command_metadata = {}
                hold_action = {
                    key: float(value) for key, value in last_observation.items() if key.endswith(".pos")
                }
                last_sent_action = robot.send_action(
                    {**hold_action, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
                )
                robot.stop_motion()
                has_received_command = False
                command_owner.release()

            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                command_metadata = data.pop("_command", {})
                if not isinstance(command_metadata, dict):
                    raise ValueError("Invalid command metadata")
                if command_metadata and (
                    not isinstance(command_metadata.get("client_id"), str)
                    or command_metadata.get("client_id") in {"", "legacy"}
                    or len(command_metadata["client_id"]) > 64
                    or type(command_metadata.get("sequence")) is not int
                    or command_metadata["sequence"] < 0
                ):
                    raise ValueError("Invalid command identity")
                validated_action = {}
                for key, value in data.items():
                    numeric_value = float(value)
                    if key not in robot.action_features or not math.isfinite(numeric_value):
                        raise ValueError(f"Invalid actuator target: {key}")
                    validated_action[key] = numeric_value
                if not validated_action:
                    raise ValueError("Received command contains no finite numeric action values.")
                if not watchdog_tripped and command_owner.accept(
                    command_metadata, robot.get_safety_status()["host_session_id"]
                ):
                    latest_action = validated_action
                    last_command_metadata = command_metadata
                    target_source = "command"
                    command_received = True
                    has_received_command = True
                    last_cmd_time = time.monotonic()
                    watchdog_active = False
            except zmq.Again:
                pass
            except Exception as e:
                logging.exception("Message fetching failed: %s", e)
            command_done_t = time.perf_counter()

            action_sent = False
            if command_received:
                last_sent_action = robot.send_action(latest_action)
                action_sent = True
            elif not watchdog_tripped:
                safety_corrections = robot.supervise_arm_motion()
                last_sent_action.update(safety_corrections)
                if safety_corrections:
                    target_source = "protection"
            action_done_t = time.perf_counter()

            encoding_timings_ms: dict[str, float] = {}
            if request_identity is not None and request_token is not None:
                camera_keys = tuple(robot.cameras.keys())
                encoded_camera_keys = () if request_token.endswith(b":state") else camera_keys
                response_observation = {
                    **last_observation,
                    "_robot_metadata": robot_metadata,
                    "_safety": {
                        **robot.get_safety_status(),
                        "currents_ma": tracking_currents_ma,
                        "watchdog_active": watchdog_active,
                        "watchdog_events": watchdog_events,
                        "command_watchdog_timeout_s": host.watchdog_timeout_ms / 1000,
                        "command": dict(last_command_metadata),
                        "control_owner": command_owner.owner,
                        "control_epoch": command_owner.epoch,
                        "target_source": target_source,
                    },
                }
                response_timings_ms: dict[str, float] = {}
                if encoded_camera_keys:
                    response_payload = jpeg_executor.submit(
                        build_observation_multipart,
                        response_observation,
                        camera_keys,
                        encoded_camera_keys,
                        response_timings_ms,
                    )
                else:
                    response_payload = build_observation_multipart(
                        response_observation,
                        camera_keys,
                        encoded_camera_keys,
                        response_timings_ms,
                    )
                pending_observation_responses.append(
                    (
                        request_identity,
                        request_token,
                        response_payload,
                        response_timings_ms,
                    )
                )

            encode_done_t = time.perf_counter()
            while pending_observation_responses:
                identity, token, payload, response_timings_ms = pending_observation_responses[0]
                if isinstance(payload, Future):
                    if not payload.done():
                        break
                    try:
                        observation_parts = payload.result()
                    except Exception:
                        logging.exception("JPEG observation encoding failed")
                        pending_observation_responses.popleft()
                        continue
                else:
                    observation_parts = payload
                try:
                    host.zmq_observation_socket.send_multipart(
                        [identity, token, *observation_parts], flags=zmq.NOBLOCK
                    )
                except zmq.Again:
                    break
                pending_observation_responses.popleft()
                encoding_timings_ms.update(response_timings_ms)
            response_send_done_t = time.perf_counter()

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = response_send_done_t - loop_start_t

            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            loop_done_t = time.perf_counter()

            loop_timings_ms = {
                "command": (command_done_t - observation_done_t) * 1e3,
                "robot_observation": (observation_done_t - request_poll_done_t) * 1e3,
                "robot_action": (action_done_t - command_done_t) * 1e3,
                "request_poll": (request_poll_done_t - loop_start_t) * 1e3,
                "jpeg_encode": sum(encoding_timings_ms.values()),
                "response_send": (response_send_done_t - encode_done_t) * 1e3,
                "sleep": (loop_done_t - response_send_done_t) * 1e3,
                "loop": (loop_done_t - loop_start_t) * 1e3,
                **robot.logs.get("observation_timing_ms", {}),
                **encoding_timings_ms,
            }
            for name, value_ms in loop_timings_ms.items():
                timing_totals_ms[name] = timing_totals_ms.get(name, 0.0) + value_ms
            timing_loop_count += 1
            if action_sent:
                for name, value_ms in robot.logs.get("action_timing_ms", {}).items():
                    action_timing_totals_ms[name] = action_timing_totals_ms.get(name, 0.0) + value_ms
                timing_command_count += 1

            timing_elapsed_s = loop_done_t - timing_report_start_t
            if args.profile_timing and timing_elapsed_s >= 1.0:
                averages = {name: total_ms / timing_loop_count for name, total_ms in timing_totals_ms.items()}
                image_text = " ".join(
                    f"{name}={value:.1f}"
                    for name, value in averages.items()
                    if name.startswith(("camera_", "encode_"))
                )
                print(
                    f"[HOST TIMING avg ms/loop] Hz={timing_loop_count / timing_elapsed_s:.1f} "
                    f"cmd={averages['command']:.1f} robot_obs={averages['robot_observation']:.1f} "
                    f"robot_action={averages['robot_action']:.1f} "
                    f"left={averages.get('left_arm', 0.0):.1f} base={averages.get('base', 0.0):.1f} "
                    f"right={averages.get('right_arm', 0.0):.1f} lift={averages.get('lift', 0.0):.1f} "
                    f"currents={averages.get('currents', 0.0):.1f} {image_text} "
                    f"jpeg={averages['jpeg_encode']:.1f} send={averages['response_send']:.1f} "
                    f"sleep={averages['sleep']:.1f} loop={averages['loop']:.1f}",
                    flush=True,
                )
                if camera_stream is not None:
                    camera_stats = camera_stream.stats()
                    print(
                        f"[HOST CAMERA STREAM] published={camera_stats['published']:.0f} "
                        f"dropped={camera_stats['dropped']:.0f} "
                        f"errors={camera_stats['errors']:.0f} "
                        f"encode_avg={camera_stats['average_encode_ms']:.1f}ms",
                        flush=True,
                    )
                if timing_command_count:
                    action_averages = {
                        name: total_ms / timing_command_count
                        for name, total_ms in action_timing_totals_ms.items()
                    }
                    print(
                        f"[HOST ACTION avg ms/control-cycle] n={timing_command_count} "
                        f"prepare={action_averages.get('action_prepare', 0.0):.1f} "
                        f"lift={action_averages.get('action_lift', 0.0):.1f} "
                        f"relative={action_averages.get('action_relative_limit', 0.0):.1f} "
                        f"left_gripper_limit={action_averages.get('action_left_gripper_limit', 0.0):.1f} "
                        f"left_joint_limit={action_averages.get('action_left_joint_limit', 0.0):.1f} "
                        f"right_gripper_limit={action_averages.get('action_right_gripper_limit', 0.0):.1f} "
                        f"right_joint_limit={action_averages.get('action_right_joint_limit', 0.0):.1f} "
                        f"left_write={action_averages.get('action_left_write', 0.0):.1f} "
                        f"right_write={action_averages.get('action_right_write', 0.0):.1f} "
                        f"base_write={action_averages.get('action_base_write', 0.0):.1f} "
                        f"total={action_averages.get('action_total', 0.0):.1f}",
                        flush=True,
                    )
                tracking_rows = []
                for key, command in sorted(last_sent_action.items()):
                    if not key.endswith(".pos") or key not in last_observation:
                        continue
                    measured = float(last_observation[key])
                    error = float(command) - measured
                    motor = key.removesuffix(".pos")
                    tracking_rows.append((motor, float(command), measured, error))
                max_tracking_error = max((abs(row[3]) for row in tracking_rows), default=0.0)
                print(f"[HOST TRACKING] max_error={max_tracking_error:.2f}", flush=True)
                for motor, command, measured, error in tracking_rows:
                    target = float(latest_action.get(f"{motor}.pos", command))
                    current = tracking_currents_ma.get(motor)
                    current_text = "n/a" if current is None else f"{current:+.1f}mA"
                    print(
                        f"[HOST TRACKING][{motor}] "
                        f"target={target:.2f} command={command:.2f} "
                        f"measured={measured:.2f} error={error:+.2f} "
                        f"current={current_text}",
                        flush=True,
                    )
                timing_report_start_t = loop_done_t
                timing_loop_count = 0
                timing_totals_ms.clear()
                timing_command_count = 0
                action_timing_totals_ms.clear()

            duration = time.perf_counter() - start
        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down AlohaMini Host.")
        if camera_stream is not None:
            camera_stream.stop()
        jpeg_executor.shutdown(wait=True, cancel_futures=True)
        robot.disconnect()
        host.disconnect()

    logging.info("Finished AlohaMini cleanly")


if __name__ == "__main__":
    main()
