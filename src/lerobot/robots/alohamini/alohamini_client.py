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
import inspect
import json
import logging
import time
from functools import cached_property
import os
from typing import Any

import cv2
import numpy as np

from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from .config_alohamini import AlohaMiniClientConfig
from .model_specs import arm_state_keys_for_robot_model
from .lift_axis import LiftAxisConfig

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

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None
        self._pending_observation_message = None
        self._observation_request_id = 0

        self.last_frames = {}

        self.last_remote_state = {}
        # Incremented only when a new observation message is successfully decoded.
        # Callers can use this to distinguish a fresh remote frame from ``last_frames`` fallback.
        self._observation_sequence = 0
        self._lift_target_mm = None

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

        # Request-driven observation transport: no observations are produced while
        # this client is saving, so stale reset frames cannot accumulate in transit.
        self.zmq_observation_socket = self.zmq_context.socket(zmq.DEALER)
        self.zmq_observation_socket.setsockopt(zmq.RCVHWM, 1)
        self.zmq_observation_socket.setsockopt(zmq.SNDHWM, 1)
        zmq_observations_locator = f"tcp://{self.remote_ip}:{self.port_zmq_observations}"
        self.zmq_observation_socket.connect(zmq_observations_locator)

        self._pending_observation_message = self._request_observation(self.connect_timeout_s * 1000)
        if self._pending_observation_message is None:
            raise DeviceNotConnectedError("Timeout waiting for AlohaMini Host to connect expired.")

        self._is_connected = True

    def calibrate(self) -> None:
        pass

    def _request_observation(self, timeout_ms: int) -> list[bytes] | None:
        """Request one observation and ignore responses to older timed-out requests."""
        zmq = self._zmq
        self._observation_request_id += 1
        request_token = str(self._observation_request_id).encode("ascii")

        try:
            self.zmq_observation_socket.send(request_token, flags=zmq.NOBLOCK)
        except zmq.ZMQError as e:
            logging.error(f"ZMQ observation request failed: {e}")
            return None

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

    def _poll_and_get_latest_message(self) -> list[bytes] | None:
        """Requests and returns the newest observation available from the host."""

        if self._pending_observation_message is not None:
            message = self._pending_observation_message
            self._pending_observation_message = None
            return message

        message = self._request_observation(self.polling_timeout_ms)
        if message is None:
            logging.info("No new data available within timeout.")
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
        if not message_parts:
            return None

        observation = self._parse_observation_json(message_parts[0])
        if observation is None:
            return None

        encoded_frames: dict[str, np.ndarray] = {}
        if len(message_parts) == 1:
            # Backward compatibility with the previous JSON/base64 protocol.
            for cam_name, image_b64 in observation.items():
                if cam_name not in self._cameras_ft:
                    continue
                frame = self._decode_image_from_b64(image_b64)
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
                frame = self._decode_image_from_jpeg_bytes(message_parts[index + 1])
                if frame is not None:
                    encoded_frames[cam_name] = frame

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

    def _get_data(self) -> tuple[dict[str, np.ndarray], RobotObservation]:
        """
        Polls the video socket for the latest observation data.

        Attempts to retrieve and decode the latest message within a short timeout.
        If successful, updates and returns the new frames, speed, and arm state.
        If no new data arrives or decoding fails, returns the last known values.
        """

        # 1. Get the latest message from the socket
        latest_message_parts = self._poll_and_get_latest_message()

        # 2. If no message, return cached data
        if latest_message_parts is None:
            return self.last_frames, self.last_remote_state

        # 3. Parse the observation message
        parsed = self._parse_observation_message(latest_message_parts)
        if parsed is None:
            return self.last_frames, self.last_remote_state
        observation, encoded_frames = parsed

        # 4. Process the valid observation data
        try:
            new_frames, new_state = self._remote_state_from_obs(observation, encoded_frames)
        except Exception as e:
            logging.error(f"Error processing observation data, serving last observation: {e}")
            return self.last_frames, self.last_remote_state

        self.last_frames = {**self.last_frames, **new_frames}
        self.last_remote_state = new_state
        self._observation_sequence += 1

        return self.last_frames, new_state

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        Capture observations from the remote robot: current follower arm positions,
        present wheel speeds (converted to body-frame velocities: x, y, theta),
        and a camera frame. Receives over ZMQ, translate to body-frame vel
        """
        frames, obs_dict = self._get_data()

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
        now_pressed = up_pressed or dn_pressed

        # Read the last height (mm) reported by the Host
        h_now = float(self.last_remote_state.get("lift_axis.height_mm", 0.0))

        if not now_pressed:
            return {"lift_axis.height_mm": h_now, "lift_axis.vel": 0}

        step_mm = 50.0
        if up_pressed and not dn_pressed:
            target = h_now + step_mm
        elif dn_pressed and not up_pressed:
            target = h_now - step_mm
        else:
            target = h_now

        return {"lift_axis.height_mm": target}




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

        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()
        self._is_connected = False
