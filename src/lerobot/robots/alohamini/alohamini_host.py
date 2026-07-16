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
import time

import cv2
import zmq

from .alohamini import AlohaMini
from .config_alohamini import AlohaMiniConfig, AlohaMiniHostConfig


class AlohaMiniHost:
    def __init__(self, config: AlohaMiniHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        # Observations are request-driven: the client asks for a frame and the host
        # replies with the current one. This prevents frames from accumulating while
        # the client is busy encoding/saving a dataset episode.
        self.zmq_observation_socket = self.zmq_context.socket(zmq.ROUTER)
        self.zmq_observation_socket.setsockopt(zmq.SNDHWM, 1)
        self.zmq_observation_socket.setsockopt(zmq.RCVHWM, 1)
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


def build_observation_multipart(observation: dict, camera_keys) -> list[bytes]:
    """Encode state as JSON and camera images as binary JPEG multipart frames."""
    state_observation = {
        key: _jsonable(value) for key, value in observation.items() if key not in camera_keys
    }
    state_observation["_image_encoding"] = "jpeg"

    parts = [json.dumps(state_observation).encode("utf-8")]
    image_names = []
    for cam_key in camera_keys:
        frame = observation.get(cam_key)
        if frame is None:
            continue
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            logging.warning("Failed to JPEG encode camera frame %s.", cam_key)
            continue
        image_names.append(cam_key)
        parts.extend([cam_key.encode("utf-8"), buffer.tobytes()])

    state_observation["_images"] = image_names
    parts[0] = json.dumps(state_observation).encode("utf-8")
    return parts


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

    logging.info("Starting HostAgent")
    host_config = AlohaMiniHostConfig()
    host = AlohaMiniHost(host_config)

    last_cmd_time = time.time()
    watchdog_active = False
    logging.info("Waiting for commands...")

    try:
        # Business logic
        start = time.perf_counter()
        duration = 0

        while duration < host.connection_time_s:
            loop_start_time = time.time()
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                #print(f"Received action: {data}")   # debug 
                _action_sent = robot.send_action(data)
                
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                pass
            except Exception as e:
                logging.exception("Message fetching failed: %s", e)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping robot motion."
                )
                watchdog_active = True
                robot.stop_motion()

            
            last_observation = robot.get_observation()

            # Drain duplicate requests and answer only the latest one. Since the
            # response is built after receiving the request, it cannot come from a
            # queue accumulated while the client was saving.
            request_identity = None
            request_token = None
            try:
                while True:
                    request_parts = host.zmq_observation_socket.recv_multipart(flags=zmq.NOBLOCK)
                    request_identity = request_parts[0]
                    request_token = request_parts[-1]
            except zmq.Again:
                pass

            if request_identity is not None and request_token is not None:
                observation_parts = build_observation_multipart(last_observation, robot.cameras.keys())
                try:
                    host.zmq_observation_socket.send_multipart(
                        [request_identity, request_token, *observation_parts], flags=zmq.NOBLOCK
                    )
                except zmq.Again:
                    logging.info("Dropping observation response, client is not ready")

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.time() - loop_start_time

            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start
        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down AlohaMini Host.")
        robot.disconnect()
        host.disconnect()

    logging.info("Finished AlohaMini cleanly")
if __name__ == "__main__":
    main()
