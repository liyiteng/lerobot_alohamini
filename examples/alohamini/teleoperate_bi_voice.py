#!/usr/bin/env python3
# teleoperate_bi_voice.py - 带语音控制、参数化 USE_DUMMY/FPS

import argparse
import inspect
import os
import time

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.bi_so100_leader import BiSO100Leader, BiSO100LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from voice_gummy import VoiceConfig, VoiceEngine
#from voice_vosk import VoiceConfig, VoiceEngine


# ============ 参数化部分 ============ #
parser = argparse.ArgumentParser()
parser.add_argument("--use_dummy", action="store_true", help="不连接 robot，仅打印 action")
parser.add_argument("--fps", type=int, default=30, help="主循环频率 (frames per second)")
parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="LeKiwi host IP address")

args = parser.parse_args()

USE_DUMMY = args.use_dummy
FPS = args.fps
# =================================== #


class DummyLeader:
    """在没连上主从臂硬件时的空实现，确保调用安全"""
    def __init__(self, id="dummy_leader"):
        self.id = id
        self.is_connected = False
    def connect(self):
        self.is_connected = False
        print("🧪 DummyLeader.connect() called")
        return False
    def get_action(self):
        return {}
    def calibrate(self):
        pass
    def close(self):
        pass


if USE_DUMMY:
    print("🧪 USE_DUMMY 模式启动：不会连接机械臂，仅打印 action。")

# Create configs
robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_lekiwi")
bi_cfg = BiSO100LeaderConfig(
    left_arm_port="/dev/am_arm_leader_left",
    right_arm_port="/dev/am_arm_leader_right",
    id="so101_leader_bi3",
)

leader = DummyLeader() if USE_DUMMY else BiSO100Leader(bi_cfg)
keyboard = KeyboardTeleop(KeyboardTeleopConfig(id="my_laptop_keyboard"))
robot = LeKiwiClient(robot_config)

# 连接逻辑
if not USE_DUMMY:
    robot.connect()
else:
    print("🧪 robot.connect() 被跳过，仅打印 action。")

leader.connect()
keyboard.connect()
init_rerun(session_name="lekiwi_teleop")

if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:
    print("⚠️ Warning: Some devices are not connected! Still running for debug.")


def set_height_mm(mm: float):
    """命令Z轴上升到指定高度（mm）"""
    action = {"lift_axis.height_mm": float(mm)}
    if not USE_DUMMY:
        robot.send_action(action)
    print(f"tb.py Set lift height to {mm} mm")


voice = VoiceEngine(VoiceConfig())
voice.start()

VOICE_Z_EPS = 0.8          # 认为到位的误差阈值（mm）
voice_z_target_mm = None   # 语音设定的粘性 Z 目标
last_print = 0.0

while True:
    t0 = time.perf_counter()

    observation = robot.get_observation() if not USE_DUMMY else {}

    cur_h = float(observation.get("lift_axis.height_mm", 0.0)) if observation else 0.0
    voice.set_height_mm(cur_h)
    voice_act = voice.get_action_nowait()  # dict 或 {}

    now = time.monotonic()
    if now - last_print >= 1.0:
        print(f"lift_axis.height_mm = {cur_h:.2f}")
        last_print = now

    arm_actions = leader.get_action()
    keyboard_keys = keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)
    lift_action = robot._from_keyboard_to_lift_action(keyboard_keys)

    # 粘性控制逻辑
    if "lift_axis.height_mm" in voice_act:
        voice_z_target_mm = float(voice_act.pop("lift_axis.height_mm"))
    if voice_act.get("__cancel_z"):
        voice_z_target_mm = None
        voice_act.pop("__cancel_z", None)
    if voice_z_target_mm is not None:
        if abs(cur_h - voice_z_target_mm) <= VOICE_Z_EPS:
            voice_z_target_mm = None
        else:
            lift_action["lift_axis.height_mm"] = voice_z_target_mm

    action = {**arm_actions, **base_action, **lift_action, **voice_act}
    log_rerun_data(observation, action)

    if USE_DUMMY:
        print(f"[USE_DUMMY] action → {action}")
    else:
        robot.send_action(action)

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
