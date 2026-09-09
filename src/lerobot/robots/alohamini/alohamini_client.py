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

# TODO(aliberts, Steven, Pepijn): use gRPC calls instead of zmq?

import base64
import json
import logging
import time
from collections import deque
from functools import cached_property

import cv2
import numpy as np

from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from .config_alohamini import AlohaMiniClientConfig
from .lift_axis import LiftAxisConfig
from .model_specs import arm_state_keys_for_robot_model

logging.basicConfig(
    #level=logging.INFO,
    format="[%(filename)s:%(lineno)d] %(message)s"
)

class AlohaMiniClient(Robot):
    config_class = AlohaMiniClientConfig
    name = "alohamini_client"

    def __init__(self, config: AlohaMiniClientConfig):
        import zmq

        self._zmq = zmq
        super().__init__(config)
        self.config = config
        self.id = config.id
        self.robot_type = config.type

        self.remote_ip = config.remote_ip
        self.port_zmq_cmd = config.port_zmq_cmd
        self.port_zmq_observations = config.port_zmq_observations

        self.teleop_keys = config.teleop_keys

        self.polling_timeout_ms = config.polling_timeout_ms
        self.connect_timeout_s = config.connect_timeout_s
        self.observation_request_window = config.observation_request_window
        if self.observation_request_window < 1:
            raise ValueError("observation_request_window must be at least 1")

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None
        self._observation_request_tokens: deque[bytes] = deque()
        self._observation_request_id = 0

        self.last_frames = {}

        self.last_remote_state = {}
        # Incremented only when a new observation message is successfully decoded.
        # Callers can use this to distinguish a fresh remote frame from ``last_frames`` fallback.
        self._observation_sequence = 0
        self.latest_host_timing: dict = {}
        self.latest_robot_metadata: dict = {}
        self._lift_target_mm = None
        self._lift_last_update_t: float | None = None
        self._lift_direction = 0

        # Define three speed levels and a current index
        self.speed_levels = [
            {"xy": 0.15, "theta": 45},  # slow
            {"xy": 0.2, "theta": 60},  # medium
            {"xy": 0.25, "theta": 75},  # fast
        ]
        self.speed_index = 0  # Start at slow

        self._is_connected = False
        self.logs = {}

        # Must match the host-side robot_model so observation/action schemas stay aligned.
        self._left_arm_state_keys, self._right_arm_state_keys = arm_state_keys_for_robot_model(
            config.robot_model
        )

    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                *self._left_arm_state_keys,
                *self._right_arm_state_keys,
                "x.vel",
                "y.vel",
                "theta.vel",
                "lift_axis.height_mm",
            ),
            float,
        )

    @cached_property
    def _state_order(self) -> tuple[str, ...]:
        return tuple(self._state_ft.keys())

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def observation_sequence(self) -> int:
        """Number of successfully received remote observations."""
        return self._observation_sequence

    @property
    def is_calibrated(self) -> bool:
        pass

    @check_if_already_connected
    def connect(self) -> None:
        """Establishes ZMQ sockets with the remote mobile robot"""

        zmq = self._zmq
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUSH)
        # Socket options that control queueing must be set before connect().
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        zmq_cmd_locator = f"tcp://{self.remote_ip}:{self.port_zmq_cmd}"
        self.zmq_cmd_socket.connect(zmq_cmd_locator)

        # Request-driven observation transport with a small bounded window. This covers
        # network round-trip latency without allowing stale frames to accumulate unboundedly.
        self.zmq_observation_socket = self.zmq_context.socket(zmq.DEALER)
        self.zmq_observation_socket.setsockopt(zmq.RCVHWM, self.observation_request_window)
        self.zmq_observation_socket.setsockopt(zmq.SNDHWM, self.observation_request_window)
        zmq_observations_locator = f"tcp://{self.remote_ip}:{self.port_zmq_observations}"
        self.zmq_observation_socket.connect(zmq_observations_locator)

        handshake_message = self._request_observation(self.connect_timeout_s * 1000)
        if handshake_message is None:
            raise DeviceNotConnectedError("Timeout waiting for AlohaMini Host to connect expired.")

        # The handshake proves that the Host is available, but it may become stale while
        # the remaining teleoperation devices connect. Discard it and fill a bounded request
        # window for frames that get_observation() will consume.
        self._fill_observation_request_window()

        self._is_connected = True

    def calibrate(self) -> None:
        pass

    def _send_observation_request(self, *, include_cameras: bool = True) -> bytes | None:
        """Send one observation request without waiting for its response."""
        zmq = self._zmq
        self._observation_request_id += 1
        response_kind = "camera" if include_cameras else "state"
        request_token = f"{self._observation_request_id}:{response_kind}".encode("ascii")

        try:
            self.zmq_observation_socket.send(request_token, flags=zmq.NOBLOCK)
        except zmq.ZMQError as e:
            logging.error(f"ZMQ observation request failed: {e}")
            return None
        return request_token

    def _receive_observation_response(
        self, request_token: bytes, timeout_ms: int
    ) -> list[bytes] | None:
        """Wait for one token-matched response and discard responses to older requests."""
        zmq = self._zmq

        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)
        deadline = time.monotonic() + timeout_ms / 1000

        while True:
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            if remaining_ms == 0:
                return None
            try:
                socks = dict(poller.poll(remaining_ms))
            except zmq.ZMQError as e:
                logging.error(f"ZMQ observation poll failed: {e}")
                return None
            if self.zmq_observation_socket not in socks:
                return None

            while True:
                try:
                    response = self.zmq_observation_socket.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                if response and response[0] == request_token:
                    return response[1:]

    def _request_observation(self, timeout_ms: int) -> list[bytes] | None:
        """Send and synchronously receive one observation request."""
        request_token = self._send_observation_request()
        if request_token is None:
            return None
        return self._receive_observation_response(request_token, timeout_ms)

    def _fill_observation_request_window(self, *, include_cameras: bool = True) -> None:
        """Keep a bounded number of requests in flight to cover transport latency."""
        while len(self._observation_request_tokens) < self.observation_request_window:
            request_token = self._send_observation_request(include_cameras=include_cameras)
            if request_token is None:
                break
            self._observation_request_tokens.append(request_token)

    def prime_observation_request_window(self, *, include_cameras: bool) -> None:
        """Drain prefetched responses and refill the window with one payload kind.

        Recording alternates state-only and camera requests. Setup and reset may
        leave a full window of camera requests in flight; returning those at the
        start of an episode creates an unintended image burst. Drain the bounded
        window before establishing the episode's initial state-only pipeline.
        """
        while self._observation_request_tokens:
            request_token = self._observation_request_tokens.popleft()
            self._receive_observation_response(request_token, self.polling_timeout_ms)
        self.latest_host_timing = {}
        self._fill_observation_request_window(include_cameras=include_cameras)

    def _poll_and_get_latest_message(self, *, include_cameras: bool = True) -> list[bytes] | None:
        """Consume the oldest response and replenish the bounded request window."""

        if not self._observation_request_tokens:
            self._fill_observation_request_window(include_cameras=include_cameras)

        message = (
            self._receive_observation_response(
                self._observation_request_tokens.popleft(), self.polling_timeout_ms
            )
            if self._observation_request_tokens
            else None
        )
        if message is None:
            logging.info("No new data available within timeout.")
            # A missing response may make the remaining ordered tokens ambiguous.
            # Drop local bookkeeping; later responses are rejected by token matching.
            self._observation_request_tokens.clear()
        else:
            # Replenish before decoding the current frame so Host work and transport overlap
            # JPEG decoding, teleoperation, action sending, and dataset I/O.
            self._fill_observation_request_window(include_cameras=include_cameras)
        return message

    def _parse_observation_json(self, obs_data: str | bytes) -> RobotObservation | None:
        """Parses the JSON observation metadata."""
        try:
            if isinstance(obs_data, bytes):
                obs_data = obs_data.decode("utf-8")
            return json.loads(obs_data)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logging.error(f"Error decoding JSON observation: {e}")
            return None

    def _decode_image_from_b64(self, image_b64: str) -> np.ndarray | None:
        """Decodes a base64 encoded image string to an OpenCV image."""
        if not image_b64:
            return None
        try:
            jpg_data = base64.b64decode(image_b64)
            np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                logging.warning("cv2.imdecode returned None for an image.")
            return frame
        except (TypeError, ValueError) as e:
            logging.error(f"Error decoding base64 image data: {e}")
            return None

    def _decode_image_from_jpeg_bytes(self, jpg_data: bytes) -> np.ndarray | None:
        """Decodes JPEG bytes from a ZMQ multipart frame to an OpenCV image."""
        if not jpg_data:
            return None
        frame = cv2.imdecode(np.frombuffer(jpg_data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            logging.warning("cv2.imdecode returned None for JPEG bytes.")
        return frame

    def _parse_observation_message(
        self, message_parts: list[bytes]
    ) -> tuple[RobotObservation, dict[str, np.ndarray]] | None:
        """Parse either the new multipart JPEG protocol or the legacy base64 JSON protocol."""
        parse_start_t = time.perf_counter()
        if not message_parts:
            return None

        observation = self._parse_observation_json(message_parts[0])
        if observation is None:
            return None
        json_done_t = time.perf_counter()

        encoded_frames: dict[str, np.ndarray] = {}
        decode_timings_ms: dict[str, float] = {}
        if len(message_parts) == 1:
            # Backward compatibility with the previous JSON/base64 protocol.
            for cam_name, image_b64 in observation.items():
                if cam_name not in self._cameras_ft:
                    continue
                decode_start_t = time.perf_counter()
                frame = self._decode_image_from_b64(image_b64)
                decode_timings_ms[f"decode_{cam_name}"] = (
                    time.perf_counter() - decode_start_t
                ) * 1e3
                if frame is not None:
                    encoded_frames[cam_name] = frame
        else:
            if (len(message_parts) - 1) % 2 != 0:
                logging.warning("Invalid multipart observation: expected camera/JPEG pairs.")
            for index in range(1, len(message_parts) - 1, 2):
                try:
                    cam_name = message_parts[index].decode("utf-8")
                except UnicodeDecodeError:
                    logging.warning("Invalid camera name in multipart observation.")
                    continue
                if cam_name not in self._cameras_ft:
                    continue
                decode_start_t = time.perf_counter()
                frame = self._decode_image_from_jpeg_bytes(message_parts[index + 1])
                decode_timings_ms[f"decode_{cam_name}"] = (
                    time.perf_counter() - decode_start_t
                ) * 1e3
                if frame is not None:
                    encoded_frames[cam_name] = frame

        parse_done_t = time.perf_counter()
        self.logs["observation_decode_timing_ms"] = {
            "obs_json": (json_done_t - parse_start_t) * 1e3,
            **decode_timings_ms,
            "obs_parse_decode": (parse_done_t - parse_start_t) * 1e3,
        }
        return observation, encoded_frames

    def _remote_state_from_obs(
        self, observation: RobotObservation, encoded_frames: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], RobotObservation]:
        """Extracts frames, and state from the parsed observation."""

        flat_state = {key: observation.get(key, 0.0) for key in self._state_order}

        state_vec = np.array([flat_state[key] for key in self._state_order], dtype=np.float32)

        obs_dict: RobotObservation = {**flat_state, OBS_STATE: state_vec}
        #lineno = frame.f_lineno
        #print(f"[{filename}:{lineno}] obs_dict:{obs_dict}")
        #print(f"[{filename}:{frame.f_lineno}] obs_dict:{obs_dict}")

        #logging.warning("obs_dict: %s", obs_dict)

        return encoded_frames, obs_dict

    def _get_data(self, *, include_cameras: bool = True) -> tuple[dict[str, np.ndarray], RobotObservation]:
        """
        Polls the video socket for the latest observation data.

        Attempts to retrieve and decode the latest message within a short timeout.
        If successful, updates and returns the new frames, speed, and arm state.
        If no new data arrives or decoding fails, returns the last known values.
        """

        observation_start_t = time.perf_counter()

        # 1. Get the latest message from the socket
        latest_message_parts = self._poll_and_get_latest_message(include_cameras=include_cameras)
        receive_done_t = time.perf_counter()

        # 2. If no message, return cached data
        if latest_message_parts is None:
            self.logs["observation_timing_ms"] = {
                "obs_wait": (receive_done_t - observation_start_t) * 1e3,
                "obs_client_total": (receive_done_t - observation_start_t) * 1e3,
            }
            return self.last_frames, self.last_remote_state

        # 3. Parse the observation message
        parsed = self._parse_observation_message(latest_message_parts)
        parse_done_t = time.perf_counter()
        if parsed is None:
            self.logs["observation_timing_ms"] = {
                "obs_wait": (receive_done_t - observation_start_t) * 1e3,
                **self.logs.get("observation_decode_timing_ms", {}),
                "obs_client_total": (parse_done_t - observation_start_t) * 1e3,
            }
            return self.last_frames, self.last_remote_state
        observation, encoded_frames = parsed
        self.latest_host_timing = dict(observation.get("_host_timing", {}))
        self.latest_robot_metadata = dict(observation.get("_robot_metadata", {}))

        # 4. Process the valid observation data
        try:
            new_frames, new_state = self._remote_state_from_obs(observation, encoded_frames)
        except Exception as e:
            logging.error(f"Error processing observation data, serving last observation: {e}")
            return self.last_frames, self.last_remote_state

        self.last_frames = {**self.last_frames, **new_frames}
        self.last_remote_state = new_state
        self._observation_sequence += 1
        observation_done_t = time.perf_counter()
        self.logs["observation_timing_ms"] = {
            "obs_wait": (receive_done_t - observation_start_t) * 1e3,
            **self.logs.get("observation_decode_timing_ms", {}),
            "obs_state": (observation_done_t - parse_done_t) * 1e3,
            "obs_client_total": (observation_done_t - observation_start_t) * 1e3,
        }

        return self.last_frames, new_state

    @check_if_not_connected
    def get_observation(self, *, include_cameras: bool = True) -> RobotObservation:
        """
        Capture observations from the remote robot: current follower arm positions,
        present wheel speeds (converted to body-frame velocities: x, y, theta),
        and a camera frame. Receives over ZMQ, translate to body-frame vel
        """
        frames, obs_dict = self._get_data(include_cameras=include_cameras)

        # Always return every configured camera key. Dataset feature construction expects a stable
        # observation schema even if a frame is dropped or a camera has not produced data yet.
        for cam_name, (height, width, channels) in self._cameras_ft.items():
            frame = frames.get(cam_name)
            if frame is None:
                logging.warning("Frame is None for %s; using zeros.", cam_name)
                frame = np.zeros((height, width, channels), dtype=np.uint8)
            obs_dict[cam_name] = frame


        return obs_dict

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray):
        # Speed control
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, 2)
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]  # e.g. 0.1, 0.25, or 0.4
        theta_speed = speed_setting["theta"]  # e.g. 30, 60, or 90

        x_cmd = 0.0  # m/s forward/backward
        y_cmd = 0.0  # m/s lateral
        theta_cmd = 0.0  # deg/s rotation

        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd += xy_speed
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd -= xy_speed
        if self.teleop_keys["left"] in pressed_keys:
            y_cmd += xy_speed
        if self.teleop_keys["right"] in pressed_keys:
            y_cmd -= xy_speed
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += theta_speed
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= theta_speed


        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
        }

    # lift_axis.vel
    # def _from_keyboard_to_lift_action(self, pressed_keys: np.ndarray):
    #     LIFT_VEL = 1000  # adjust if too slow/fast
    #     up_pressed = self.teleop_keys.get("lift_up", "u") in pressed_keys
    #     dn_pressed = self.teleop_keys.get("lift_down", "j") in pressed_keys

    #     if up_pressed and not dn_pressed:
    #         v = +LIFT_VEL
    #     elif dn_pressed and not up_pressed:
    #         v = -LIFT_VEL
    #     else:
    #         v = 0.0
    #     return {"lift_axis.vel": int(v)}


    # lift_axis.height_mm
    def _from_keyboard_to_lift_action(self, pressed_keys: np.ndarray):
        up_pressed = self.teleop_keys.get("lift_up", "u") in pressed_keys
        dn_pressed = self.teleop_keys.get("lift_down", "j") in pressed_keys
        direction = int(up_pressed) - int(dn_pressed)

        # Use physical height, not the servo's wrapping single-turn register.
        h_now = float(self.last_remote_state.get("lift_axis.height_mm", 0.0))
        now = time.monotonic()
        default_lift_config = LiftAxisConfig()
        lift_metadata = self.latest_robot_metadata.get("lift_axis", {})
        soft_min_mm = float(
            lift_metadata.get("soft_min_mm", default_lift_config.soft_min_mm)
        )
        soft_max_mm = float(
            lift_metadata.get("soft_max_mm", default_lift_config.soft_max_mm)
        )

        # Release, opposing keys, and direction changes re-latch to feedback.
        if direction == 0:
            self._lift_target_mm = h_now
        else:
            relatch = self._lift_target_mm is None or direction != self._lift_direction
            if relatch:
                self._lift_target_mm = h_now
            dt = (
                1.0 / 50.0
                if self._lift_last_update_t is None or relatch
                else min(max(now - self._lift_last_update_t, 0.0), 0.1)
            )
            self._lift_target_mm += direction * self.config.lift_target_speed_mm_s * dt
            max_lead_mm = self.config.lift_target_max_lead_mm
            self._lift_target_mm = min(
                max(self._lift_target_mm, h_now - max_lead_mm),
                h_now + max_lead_mm,
            )

        self._lift_target_mm = min(max(self._lift_target_mm, soft_min_mm), soft_max_mm)
        self._lift_last_update_t = now
        self._lift_direction = direction
        return {"lift_axis.height_mm": self._lift_target_mm}




    def configure(self):
        pass

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command AlohaMini to move to a target joint configuration. Translates to motor space + sends over ZMQ

        Args:
            action (np.ndarray): array containing the goal positions for the motors.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            np.ndarray: the action sent to the motors, potentially clipped.
        """
        self.zmq_cmd_socket.send_string(json.dumps(action))  # action is in motor space

        # TODO(Steven): Remove the np conversion when it is possible to record a non-numpy array value
        actions = np.array([action.get(k, 0.0) for k in self._state_order], dtype=np.float32)

        action_sent = {key: actions[i] for i, key in enumerate(self._state_order)}
        action_sent[ACTION] = actions
        return action_sent

    @check_if_not_connected
    def disconnect(self):
        """Cleans ZMQ comms"""

        self._observation_request_tokens.clear()
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()
        self._is_connected = False
