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

import base64
import json
import logging
import time

import cv2
import zmq

from .config_lekiwi import LeKiwiConfig, LeKiwiHostConfig
from .lekiwi import LeKiwi


class LeKiwiHost:
    def __init__(self, config: LeKiwiHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()

    def _check_overcurrent_and_maybe_shutdown(self, left_curr_raw: dict, right_curr_raw: dict):
        """将原始电流换算为 mA，并在超过阈值时执行安全断开。
        注意：只做保护检查，不写 observation。
        """
        if not self.enable_overcurrent_protect:
            return  # 开关关闭则不做任何事

        for name, raw in {**left_curr_raw, **right_curr_raw}.items():
            try:
                current_ma = float(raw) * self.current_scale
            except Exception:
                current_ma = 0.0

            if current_ma > self.current_limit_ma:
                print(f"[Overcurrent] {name}: {current_ma:.1f} mA > {self.current_limit_ma:.1f} mA -> disconnecting!")
                try:
                    # 你已有的安全停机逻辑，按需选择 stop_base / disconnect
                    self.stop_base()     # 先停底盘（如果有）
                except Exception:
                    pass
                try:
                    self.disconnect()    # 断开总线
                except Exception as e:
                    print(f"[Overcurrent] disconnect error: {e}")
                sys.exit(1)    

def main():
    logging.info("Configuring LeKiwi")
    robot_config = LeKiwiConfig()
    robot_config.id = "AlohaMiniRobot"
    robot = LeKiwi(robot_config)


    logging.info("Connecting LeKiwi")
    robot.connect()

    logging.info("Starting HostAgent")
    host_config = LeKiwiHostConfig()
    host = LeKiwiHost(host_config)

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
                if not watchdog_active:
                    logging.warning("No command available")
            except Exception as e:
                logging.exception("Message fetching failed: %s", e)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping the base."
                )
                watchdog_active = True
                robot.stop_base()

            
            robot.lift.update()
            last_observation = robot.get_observation()

            # Encode ndarrays to base64 strings
            for cam_key, _ in robot.cameras.items():
                ret, buffer = cv2.imencode(
                    ".jpg", last_observation[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                )
                if ret:
                    last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
                else:
                    last_observation[cam_key] = ""

            # Send the observation to the remote agent
            try:
                host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.info("Dropping observation, no client connected")

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.time() - loop_start_time

            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start
        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down Lekiwi Host.")
        robot.disconnect()
        host.disconnect()

    logging.info("Finished LeKiwi cleanly")


if __name__ == "__main__":
    main()
