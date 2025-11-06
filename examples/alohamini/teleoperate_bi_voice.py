#!/usr/bin/env python3


import argparse
import time

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.bi_so100_leader import BiSO100Leader, BiSO100LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# NEW:
from voice_engine_gummy import SpeechEngineGummy, SpeechConfig
from voice_exec import VoiceExecutor, ExecConfig

parser = argparse.ArgumentParser()
parser.add_argument("--use_dummy", action="store_true", help="Do not connect robot, only print actions")
parser.add_argument("--fps", type=int, default=30, help="Main loop frequency (frames per second)")
parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="Alohamini host IP address")
args = parser.parse_args()

USE_DUMMY = args.use_dummy
FPS = args.fps

if USE_DUMMY:
    print("🧪 USE_DUMMY mode: no robot connection, printing actions only.")

# Create configs
robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini")
bi_cfg = BiSO100LeaderConfig(
    left_arm_port="/dev/am_arm_leader_left",
    right_arm_port="/dev/am_arm_leader_right",
    id="so101_leader_bi3",
)

leader = BiSO100Leader(bi_cfg)
keyboard = KeyboardTeleop(KeyboardTeleopConfig(id="my_laptop_keyboard"))
robot = LeKiwiClient(robot_config)

# NEW: build engines
speech = SpeechEngineGummy(SpeechConfig(
    model="gummy-chat-v1",
    vocabulary_prefix="gummyam",
    hotwords=[
        "上升","下降","前进","后退","左移","右移","左转","右转","停止",
        "毫米","厘米","米","秒","秒钟",
        "up","down","forward","back","turn left","turn right","rotate left","rotate right",
        "move left","move right","strafe left","strafe right",
        "millimeter","millimeters","centimeter","centimeters","meter","meters",
        "second","seconds","sec","s","for",
        "zero","oh","one","two","three","four","five","six","seven","eight","nine",
        "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
        "seventeen","eighteen","nineteen","twenty","thirty","forty","fifty",
        "sixty","seventy","eighty","ninety","hundred","half","quarter",
    ]
))
execu = VoiceExecutor(ExecConfig(xy_speed_cmd=0.20, theta_speed_cmd=500.0, emit_text_cmd=True))

# Connection logic
if not USE_DUMMY:
    robot.connect()
else:
    print("🧪 robot.connect() skipped.")

leader.connect()
keyboard.connect()
init_rerun(session_name="lekiwi_teleop")

if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:
    print("⚠️ Warning: Some devices are not connected! Still running for debug.")

# start speech
speech.start()

try:
    while True:
        t0 = time.perf_counter()

        observation = robot.get_observation() if not USE_DUMMY else {}
        cur_h = float(observation.get("lift_axis.height_mm", 0.0)) if observation else 0.0
        execu.update_height_mm(cur_h)

        # finalized utterance -> executor
        text = speech.get_text_nowait()
        if text:
            execu.handle_text(text)

        arm_actions = leader.get_action()
        keyboard_keys = keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)
        lift_action = robot._from_keyboard_to_lift_action(keyboard_keys)

        voice_action = execu.get_action_nowait()
        action = {**arm_actions, **base_action, **lift_action, **voice_action}
        #log_rerun_data(observation, action)

        

        if USE_DUMMY:
            print(f"[USE_DUMMY] action → {action}")
        else:
            robot.send_action(action)

        busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
finally:
    try:
        speech.stop()
    except Exception:
        pass