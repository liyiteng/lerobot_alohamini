#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-omniwheel (120-degree) keyboard teleop demo (Feetech, new LeRobot API style).
- Only `--port` is a CLI argument; everything else is global constants.
- Three motors are set to VELOCITY mode and written via Goal_Speed (raw 16-bit).
- Keys: W/S forward/back, A/D rotate left/right, Q/E strafe, X or ESC to quit.

Dependencies:
  pip install pynput numpy
Usage:
  python omni_teleop_feetech_new_lerobot.py --port /dev/ttyACM0
"""
from __future__ import annotations
import argparse
import time
from typing import Dict, List

import numpy as np
from pynput import keyboard

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

# ------------------------ Global constants (edit here) ------------------------ #
DEFAULT_PORT: str = "/dev/ttyACM0"
MODEL: str = "sts3215"         # Feetech model
LEFT_ID: int = 8                # Left wheel motor ID
BACK_ID: int = 9                # Back wheel motor ID
RIGHT_ID: int = 10              # Right wheel motor ID
LIN_SPEED: float = 0.2         # Linear speed (m/s)
ANG_SPEED: float = 80.0         # Angular speed (deg/s)
WHEEL_RADIUS: float = 0.05      # Wheel radius (m)
BASE_RADIUS: float = 0.125      # Wheel-to-center distance (m)
MAX_RAW: int = 3000             # Raw speed limit (scaled)


# def degps_to_raw(degps: float) -> int:
#     """Angular speed (deg/s) -> Feetech 16-bit signed encoding (bit15 is sign)."""
#     steps_per_deg = 4096.0 / 360.0
#     speed_in_steps = abs(degps) * steps_per_deg
#     speed_int = int(round(speed_in_steps))
#     if speed_int > 0x7FFF:
#         speed_int = 0x7FFF
#     return (speed_int | 0x8000) if degps < 0 else (speed_int & 0x7FFF)

def degps_to_raw(degps: float) -> int:
    """Angular speed (deg/s) -> steps/s (-32767..+32767), no sign-bit encoding."""
    steps_per_deg = 4096.0 / 360.0
    mag = int(round(abs(degps) * steps_per_deg))
    if mag > 0x7FFF:
        mag = 0x7FFF
    return -mag if degps < 0 else mag



def raw_to_degps(raw_speed: int) -> float:
    steps_per_deg = 4096.0 / 360.0
    magnitude = raw_speed & 0x7FFF
    degps = magnitude / steps_per_deg
    return -degps if (raw_speed & 0x8000) else degps


# ------------------------ Kinematics (equilateral tri-wheel, 120 deg) ------------------------ #

def body_to_wheel_raw(
    x_cmd: float,
    y_cmd: float,
    theta_cmd_degps: float,
    *,
    wheel_radius: float = WHEEL_RADIUS,
    base_radius: float = BASE_RADIUS,
    max_raw: int = MAX_RAW,
) -> Dict[str, int]:
    """Body velocity -> per-wheel raw speed commands.
    Args: x_cmd/y_cmd in m/s; theta_cmd_degps in deg/s.
    Wheel mounting angles are defined clockwise from +y: left=300°, back=180°, right=60°.
    Returns dict: left_wheel/back_wheel/right_wheel.
    """
    theta_rad = theta_cmd_degps * (np.pi / 180.0)
    vel = np.array([-x_cmd, -y_cmd, theta_rad])

    #angles = np.radians(np.array([300, 180, 60]))
    angles = np.radians(np.array([240,0,120]) - 90)
    M = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

    v_lin = M.dot(vel)                      # m/s
    w_rad = v_lin / wheel_radius            # rad/s
    w_degps = w_rad * (180.0 / np.pi)       # °/s

    # Scale to avoid exceeding raw limits.
    steps_per_deg = 4096.0 / 360.0
    raw_abs = np.abs(w_degps) * steps_per_deg
    peak = float(np.max(raw_abs)) if raw_abs.size else 0.0
    if peak > max_raw and peak > 1e-6:
        w_degps = w_degps * (max_raw / peak)

    raw = [degps_to_raw(v) for v in w_degps]
    print(f"raw:{raw}")
    return {"left_wheel": raw[0], "back_wheel": raw[1], "right_wheel": raw[2]}


def wheel_raw_to_body(
    wheel_raw: Dict[str, int], *, wheel_radius: float = WHEEL_RADIUS, base_radius: float = BASE_RADIUS
) -> tuple[float, float, float]:
    raw_list = [int(wheel_raw.get(n, 0)) for n in ("left_wheel", "back_wheel", "right_wheel")]
    w_degps = np.array([raw_to_degps(r) for r in raw_list])
    w_rad = w_degps * (np.pi / 180.0)
    v_lin = w_rad * wheel_radius

    #angles = np.radians(np.array([300, 180, 60]))
    angles = np.radians(np.array([240,0,120]) - 90)

    M = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])
    M_inv = np.linalg.inv(M)
    x_cmd, y_cmd, theta_rad = M_inv.dot(v_lin)
    x_cmd = -x_cmd
    y_cmd = -y_cmd
    theta_cmd_degps = theta_rad * (180.0 / np.pi)
    return x_cmd, y_cmd, theta_cmd_degps


# ------------------------ Keyboard teleop ------------------------ #

TELEOP_KEYS = {
    "forward": "w",
    "backward": "s",
    "left": "a",          # Strafe left
    "right": "d",         # Strafe right
    "rotate_left": "z",
    "rotate_right": "x",
    "quit": "q",
}


class OmniTeleop:
    def __init__(self, port: str):
        self.motors = {
            "left_wheel":  Motor(id=LEFT_ID,  model=MODEL, norm_mode=MotorNormMode.RANGE_0_100),
            "back_wheel":  Motor(id=BACK_ID,  model=MODEL, norm_mode=MotorNormMode.RANGE_0_100),
            "right_wheel": Motor(id=RIGHT_ID, model=MODEL, norm_mode=MotorNormMode.RANGE_0_100),
        }
        self.bus = FeetechMotorsBus(port=port, motors=self.motors)
        self.running = True
        self.pressed = {k: False for k in TELEOP_KEYS}

        self.lin_speed = float(LIN_SPEED)
        self.ang_speed = float(ANG_SPEED)

    # ---- Keyboard events ----
    def _on_press(self, key):
        try:
            ch = key.char
        except Exception:
            ch = None
        if ch is None:
            if key == keyboard.Key.esc:
                self.running = False
            return
        for action, bind in TELEOP_KEYS.items():
            if ch == bind:
                if action == "quit":
                    self.running = False
                else:
                    self.pressed[action] = True

    def _on_release(self, key):
        try:
            ch = key.char
        except Exception:
            ch = None
        if ch is None:
            return
        for action, bind in TELEOP_KEYS.items():
            if ch == bind and action in self.pressed:
                self.pressed[action] = False

    # ---- Connect and mode switch ----
    def connect(self) -> None:
        self.bus.connect(handshake=False)
        print(f"Connected on port {self.bus.port}")
        for name in self.motors:
            try:
                self.bus.write("Lock", name, 0, normalize=False)
            except Exception:
                pass
            try:
                self.bus.disable_torque(name)
            except Exception:
                pass
            self.bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value, normalize=False)
            self.bus.enable_torque(name)
        print("Motors set to VELOCITY mode.")

    def stop(self):
        try:
            names = list(self.motors.keys())
            zeros = [0, 0, 0]
            self.bus.write("Goal_Velocity", zeros, names)
        except Exception:
            pass

    def close(self):
        try:
            self.bus.disconnect(disable_torque=False)
        except Exception:
            pass

    # ---- Main loop ----
    def run(self):
        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.start()
        try:
            while self.running:
                x = self.lin_speed if self.pressed.get("forward") else (-self.lin_speed if self.pressed.get("backward") else 0.0)
                y = self.lin_speed if self.pressed.get("left")    else (-self.lin_speed if self.pressed.get("right")    else 0.0)
                th = self.ang_speed if self.pressed.get("rotate_left") else (-self.ang_speed if self.pressed.get("rotate_right") else 0.0)

                wheel_cmds = body_to_wheel_raw(x, y, th)
                names: List[str] = list(self.motors.keys())
                raw_vals = [wheel_cmds[n] for n in names]
                for name, val in zip(names, raw_vals):
                    self.bus.write("Goal_Velocity", name, val, normalize=False)

                try:
                    currents_raw = {
                        name: self.bus.read("Present_Current", name, normalize=False) for name in names
                    }
                    currents_ma = {name: currents_raw[name] * 6.5 for name in names}
                    ids = {name: self.motors[name].id for name in names}
                    print(
                        "Current(mA) left(id={left_id})={left_wheel:.1f} "
                        "back(id={back_id})={back_wheel:.1f} right(id={right_id})={right_wheel:.1f}".format(
                            left_id=ids["left_wheel"],
                            back_id=ids["back_wheel"],
                            right_id=ids["right_wheel"],
                            **currents_ma,
                        )
                    )
                except Exception as exc:
                    print(f"Current read failed: {exc}")

                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            listener.stop()
            self.stop()
            self.close()
            print("Teleop stopped.")


# ------------------------ CLI ------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Feetech 3-omni teleop (globals-only)")
    p.add_argument("--port", type=str, default=DEFAULT_PORT, help=f"Serial port (default: {DEFAULT_PORT})")
    return p.parse_args()


def main():
    args = parse_args()
    teleop = OmniTeleop(args.port)
    teleop.connect()
    teleop.run()


if __name__ == "__main__":
    main()
