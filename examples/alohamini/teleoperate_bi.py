import argparse
import inspect
import os
import time

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.bi_so100_leader import BiSO100Leader, BiSO100LeaderConfig 
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ============ 参数化部分 ============ #
parser = argparse.ArgumentParser()
parser.add_argument("--use_dummy", action="store_true", help="不连接 robot，仅打印 action")
parser.add_argument("--fps", type=int, default=30, help="主循环频率 (frames per second)")
parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="LeKiwi host IP address")


args = parser.parse_args()

USE_DUMMY = args.use_dummy
FPS = args.fps
# =================================== #

if USE_DUMMY:
    print("🧪 USE_DUMMY 模式启动：不会连接机器人，只打印 action。")

# Create configs
robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini")
bi_cfg = BiSO100LeaderConfig(
    left_arm_port="/dev/am_arm_leader_left",
    right_arm_port="/dev/am_arm_leader_right",
    id="so101_leader_bi3",
)
leader = BiSO100Leader(bi_cfg)
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")
keyboard = KeyboardTeleop(keyboard_config)
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

# 主循环
while True:
    t0 = time.perf_counter()

    observation = robot.get_observation() if not USE_DUMMY else {}

    arm_actions = leader.get_action()
    keyboard_keys = keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)
    lift_action = robot._from_keyboard_to_lift_action(keyboard_keys)

    action = {**arm_actions, **base_action, **lift_action}
    log_rerun_data(observation, action)

    if USE_DUMMY:
        print(f"[USE_DUMMY] action → {action}")
    else:
        robot.send_action(action)

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
