#!/usr/bin/env python3
# axis.py - 带过流保护（PORT 参数化版）

import time
import argparse
from pynput import keyboard
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

# ==================== 常量配置 ==================== #
SERVO_ID = 11               # 舵机 ID
MODEL = "sts3215"           # 舵机型号
SPEED_DEGPS = 180.0         # 转速 deg/s
CURRENT_CUTOFF = 300.0      # 过流阈值 mA
SAMPLES_TO_TRIGGER = 2      # 连续多少次超过阈值才触发
# ================================================= #

STEPS_PER_DEG = 4096.0 / 360.0  # ≈11.377 ticks/deg
TELEOP_KEYS = {"up": "u", "down": "j"}
pressed = {"up": False, "down": False}


def degps_to_raw(degps: float) -> int:
    mag = int(round(abs(degps) * STEPS_PER_DEG))
    if mag > 0x7FFF:
        mag = 0x7FFF
    return -mag if degps < 0 else mag


def on_press(key):
    try:
        if key.char == TELEOP_KEYS["up"]:
            pressed["up"] = True
        elif key.char == TELEOP_KEYS["down"]:
            pressed["down"] = True
    except Exception:
        pass


def on_release(key):
    try:
        if key.char == TELEOP_KEYS["up"]:
            pressed["up"] = False
        elif key.char == TELEOP_KEYS["down"]:
            pressed["down"] = False
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Lift-axis teleop with overcurrent cutoff")
    parser.add_argument("--port", default="/dev/ttyACM0", help="USB-TTL port, e.g. /dev/ttyACM0")
    args = parser.parse_args()

    name = "lift_axis"
    motors = {name: Motor(id=SERVO_ID, model=MODEL, norm_mode=MotorNormMode.RANGE_0_100)}
    bus = FeetechMotorsBus(port=args.port, motors=motors)

    bus.connect(handshake=False)
    print(f"[INFO] Connected on {args.port}")
    try:
        bus.disable_torque(name)
    except Exception:
        pass
    bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value, normalize=False)
    try:
        bus.enable_torque(name)
    except Exception:
        pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print(f"[INFO] Ready. U ↑ / J ↓ / Ctrl-C to quit.")
    print(f"[INFO] Overcurrent cutoff: {CURRENT_CUTOFF} mA, trigger after {SAMPLES_TO_TRIGGER} samples")

    over_cnt = 0
    try:
        while True:
            # 1) 读当前电流
            try:
                raw_current_ma = bus.read("Present_Current", name, normalize=False)
                if isinstance(raw_current_ma, tuple):
                    raw_current_ma = raw_current_ma[0]
                if raw_current_ma is None:
                    raw_current_ma = 0
                else:
                    current_ma = raw_current_ma * 6.5
                    print(f"[DEBUG] Present_Current={current_ma} mA")  # debug
            except Exception:
                raw_current_ma = 0

            # 2) 判断是否过流
            if current_ma > CURRENT_CUTOFF:
                over_cnt += 1
            else:
                over_cnt = 0

            if over_cnt >= SAMPLES_TO_TRIGGER:
                print(f"\n[SAFETY] Overcurrent detected: {current_ma} mA ≥ {CURRENT_CUTOFF} mA")
                try:
                    bus.write("Goal_Velocity", name, 0, normalize=False)
                except Exception:
                    pass
                try:
                    bus.disable_torque(name)
                except Exception:
                    pass
                try:
                    bus.disconnect(disable_torque=True)
                except Exception:
                    pass
                return

            # 3) 速度控制
            if pressed["up"] and not pressed["down"]:
                raw = degps_to_raw(-SPEED_DEGPS)
            elif pressed["down"] and not pressed["up"]:
                raw = degps_to_raw(SPEED_DEGPS)
            else:
                raw = 0
            bus.write("Goal_Velocity", name, raw, normalize=False)

            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping motor...")
        try:
            bus.write("Goal_Velocity", name, 0, normalize=False)
        except Exception:
            pass
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
