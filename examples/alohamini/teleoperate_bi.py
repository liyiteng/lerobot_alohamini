import argparse
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
    "--teleop.arm_profile",
    "--arm_profile",
    dest="arm_profile",
    type=str,
    default="so-arm-5dof",
    choices=["so-arm-5dof", "am-leader-6dof"],
    help="Leader arm profile selector.",
)

args = parser.parse_args()

NO_ROBOT = args.no_robot
NO_LEADER = args.no_leader
FPS = args.fps
CAMERA_FPS = args.camera_fps
if FPS <= 0 or CAMERA_FPS <= 0:
    parser.error("--fps and --camera-fps must be positive")
if CAMERA_FPS > FPS:
    parser.error("--camera-fps must not exceed --fps")
# ========================================== #

if NO_ROBOT:
    print("🧪 NO_ROBOT mode enabled: robot will not connect, only print actions.")

if NO_LEADER:
    print("🧪 NO_LEADER mode enabled: leader arm will not connect, only print actions.")
# Create configs
robot_config = AlohaMiniClientConfig(
    remote_ip=args.remote_ip,
    id=args.robot_id,
    robot_model=args.robot_model,
)
bi_cfg = BiSOLeaderConfig(
    left_arm_config=SOLeaderConfig(
        port="/dev/am_arm_leader_left",
        arm_profile=args.arm_profile,
    ),
    right_arm_config=SOLeaderConfig(
        port="/dev/am_arm_leader_right",
        arm_profile=args.arm_profile,
    ),
    id=args.leader_id,
)
leader = BiSOLeader(bi_cfg)
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")
keyboard = KeyboardTeleop(keyboard_config)
robot = AlohaMiniClient(robot_config)

# Connection logic
if not NO_ROBOT:
    robot.connect()
else:
    print("🧪 robot.connect() skipped, only printing actions.")

if not NO_LEADER:
    leader.connect()
else:
    print("🧪 robot.connect() skipped, only printing actions.")

keyboard.connect()



init_rerun(session_name="alohamini_teleop")

if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:
    print("⚠️ Warning: Some devices are not connected! Still running for debug.")

# Main loop: 50 Hz command/state control with an independent 30 Hz camera cadence.
next_camera_request_t = time.perf_counter()
camera_interval_s = 1.0 / CAMERA_FPS
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
    keyboard_keys = keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)
    lift_action = robot._from_keyboard_to_lift_action(keyboard_keys)

    action = {**arm_actions, **base_action, **lift_action}
    log_rerun_data(observation, action)

    if not NO_ROBOT:
        robot.send_action(action)

    precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
