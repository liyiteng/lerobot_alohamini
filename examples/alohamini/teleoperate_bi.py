import argparse
import os
import time

from lerobot.robots.alohamini import AlohaMiniClient, AlohaMiniClientConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ============ Parameter Section ============ #
parser = argparse.ArgumentParser()
parser.add_argument("--no_robot", action="store_true", help="Do not connect robot, only print actions")
parser.add_argument("--no_leader", action="store_true", help="Do not connect leader arm, only perform keyboard-controlled actions.")
parser.add_argument("--fps", type=int, default=50, help="Command/control frequency")
parser.add_argument(
    "--camera-fps",
    type=int,
    default=30,
    help="Camera request frequency; independent from command/control",
)
parser.add_argument(
    "--robot.remote_ip",
    "--remote_ip",
    dest="remote_ip",
    type=str,
    default="127.0.0.1",
    help="AlohaMini host IP address",
)
parser.add_argument(
    "--robot.id",
    "--robot_id",
    dest="robot_id",
    type=str,
    default="my_alohamini",
    help="Robot ID",
)
parser.add_argument(
    "--robot.robot_model",
    "--robot_model",
    dest="robot_model",
    type=str,
    default="alohamini1",
    choices=["alohamini1", "alohamini2", "alohamini2pro"],
    help="AlohaMini model. Must match the --robot_model used on the Pi host side.",
)
parser.add_argument(
    "--teleop.id",
    "--leader_id",
    dest="leader_id",
    type=str,
    default="so101_leader_bi",
    help="Leader arm device ID",
)
parser.add_argument(
    "--display-data",
    "--display_data",
    action="store_true",
    help="Display observations and actions in Rerun viewer (default: False)",
)
parser.add_argument(
    "--teleop.arm_profile",
    "--arm_profile",
    dest="arm_profile",
    type=str,
    default="so-arm-5dof",
    choices=["so-arm-5dof", "am-leader-6dof"],
    help="Leader arm profile selector.",
)
parser.add_argument(
    "--teleop.left_port",
    "--leader.left_port",
    "--left_port",
    dest="left_port",
    type=str,
    default=os.environ.get("ALOHAMINI_LEFT_PORT", "/dev/cu.usbmodem5B910458311"),
    help="Left leader arm serial port.",
)
parser.add_argument(
    "--teleop.right_port",
    "--leader.right_port",
    "--right_port",
    dest="right_port",
    type=str,
    default=os.environ.get("ALOHAMINI_RIGHT_PORT", "/dev/cu.usbmodem5B790175581"),
    help="Right leader arm serial port.",
)

args = parser.parse_args()

NO_ROBOT = args.no_robot
NO_LEADER = args.no_leader
DISPLAY_DATA = args.display_data
FPS = args.fps
CAMERA_FPS = args.camera_fps
if FPS <= 0 or CAMERA_FPS <= 0:
    parser.error("--fps and --camera-fps must be positive")
if CAMERA_FPS > FPS:
    parser.error("--camera-fps must not exceed --fps")
# ========================================== #

if NO_ROBOT:
    print("🧪 NO_ROBOT 模式已启用：不连接从臂机器人，仅打印动作。")

if NO_LEADER:
    print("🧪 NO_LEADER 模式已启用：不连接示教主臂，仅支持键盘控制。")

# Create configs
robot_config = AlohaMiniClientConfig(
    remote_ip=args.remote_ip,
    id=args.robot_id,
    robot_model=args.robot_model,
)
bi_cfg = BiSOLeaderConfig(
    left_arm_config=SOLeaderConfig(
        port=args.left_port,
        arm_profile=args.arm_profile,
        use_degrees=False,
    ),
    right_arm_config=SOLeaderConfig(
        port=args.right_port,
        arm_profile=args.arm_profile,
        use_degrees=False,
    ),
    id=args.leader_id,
)
leader = BiSOLeader(bi_cfg)
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")
keyboard = KeyboardTeleop(keyboard_config)
robot = AlohaMiniClient(robot_config)

# Connection logic
if not NO_ROBOT:
    print(f"📡 正在连接机器人 Host ({args.remote_ip})...")
    robot.connect()
    print("✅ 机器人从臂已连接！")
else:
    print("🧪 robot.connect() 已跳过。")

if not NO_LEADER:
    print("🎮 正在连接左右示教主臂...")
    leader.connect()
    print("✅ 左右示教主臂已连接！")
else:
    print("🧪 leader.connect() 已跳过。")

print("⌨️  正在初始化键盘控制...")
keyboard.connect()
print("✅ 键盘已就绪！")

if DISPLAY_DATA:
    print("🖥️  正在启动 Rerun 可视化窗口...")
    init_rerun(session_name="alohamini_teleop")

if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:
    print("⚠️ 警告: 部分设备未连接，但仍尝试继续运行。")

print("\n🚀【遥操作已正式就绪】")
print("👉 现在缓慢移动左/右主臂，机器人从臂将实时同步跟随！")
print("👉 键盘控制底盘: W/S 前后，A/D 旋转，Z/X 平移；升降: U/J 升降。")
print("👉 按 Ctrl + C 可安全停止退出。\n")

# Main loop: 50 Hz command/state control with an independent 30 Hz camera cadence.
next_camera_request_t = time.perf_counter()
camera_interval_s = 1.0 / CAMERA_FPS
loop_start_t = time.perf_counter()
last_status_print_t = time.perf_counter()
frame_count = 0

try:
    while True:
        t0 = time.perf_counter()

        request_cameras = t0 >= next_camera_request_t
        if request_cameras:
            while next_camera_request_t <= t0:
                next_camera_request_t += camera_interval_s
        observation = (
            robot.get_observation(include_cameras=request_cameras) if not NO_ROBOT else {}
        )
        arm_actions = leader.get_action() if not NO_LEADER else {}
        arm_actions = {f"arm_{k}": v for k, v in arm_actions.items()}
        keyboard_keys = keyboard.get_action() if keyboard.is_connected else {}
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)
        lift_action = robot._from_keyboard_to_lift_action(keyboard_keys)

        action = {**arm_actions, **base_action, **lift_action}
        if DISPLAY_DATA:
            log_rerun_data(observation, action)

        if not NO_ROBOT:
            robot.send_action(action)

        frame_count += 1
        now = time.perf_counter()
        if now - last_status_print_t >= 1.0:
            elapsed = now - loop_start_t
            rate = frame_count / elapsed
            print(f"\r[遥操作运行中] 控制频率: {rate:.1f} FPS | 累计控制帧: {frame_count}", end="", flush=True)
            last_status_print_t = now

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
except KeyboardInterrupt:
    print("\n\n🛑 接收到用户中断信号，正在断开连接...")
finally:
    if not NO_ROBOT and robot.is_connected:
        robot.disconnect()
    if not NO_LEADER and leader.is_connected:
        leader.disconnect()
    if keyboard.is_connected:
        keyboard.disconnect()
    print("👋 遥操作已安全退出！")
