import time
import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.alohamini import AlohaMiniClient, AlohaMiniClientConfig
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME
from lerobot.utils.robot_utils import precise_sleep

parser = argparse.ArgumentParser(description="Replay a LeRobot dataset episode")
parser.add_argument("--dataset.repo_id", "--dataset", dest="dataset_repo_id", type=str, required=True,
                    help="Dataset repo_id, e.g. liyitenga/record_20250914225057")
parser.add_argument("--dataset.root", "--root", dest="dataset_root", type=str, default=None,
                    help="Local dataset root. Defaults to $HF_LEROBOT_HOME/<dataset.repo_id> and never downloads from Hub.")
parser.add_argument("--dataset.episode", "--episode", dest="dataset_episode", type=int, default=0,
                    help="Episode index to replay (default 0)")
parser.add_argument(
    "--replay.fps",
    "--replay_fps",
    dest="replay_fps",
    type=float,
    default=None,
    help=(
        "Action playback rate. Defaults to the dataset FPS. Set this to the actual capture rate "
        "when recording ran slower than --dataset.fps (for example, 10 for a 30 FPS dataset "
        "that was captured at about 10 Hz)."
    ),
)
parser.add_argument(
    "--replay.speed",
    "--speed",
    dest="replay_speed",
    type=float,
    default=1.0,
    help="Playback speed multiplier (default: 1.0). Use 0.333 to replay at one-third speed.",
)
parser.add_argument(
    "--verbose-actions",
    action="store_true",
    help="Print every action. Disabled by default because terminal output can disturb replay timing.",
)
parser.add_argument(
    "--robot.remote_ip",
    "--remote_ip",
    dest="remote_ip",
    type=str,
    default="127.0.0.1",
    help="AlohaMini host IP address",
)
parser.add_argument("--robot.id", "--robot_id", dest="robot_id", type=str, default="my_alohamini", help="Robot ID")
parser.add_argument(
    "--robot.robot_model",
    "--robot_model",
    dest="robot_model",
    type=str,
    default="alohamini1",
    choices=["alohamini1", "alohamini2", "alohamini2pro", "alohamini2edu"],
    help="AlohaMini model. Must match the --robot_model used on the Pi host side.",
)



args = parser.parse_args()

if args.replay_fps is not None and args.replay_fps <= 0:
    parser.error("--replay.fps must be greater than zero")
if args.replay_speed <= 0:
    parser.error("--replay.speed must be greater than zero")


robot_config = AlohaMiniClientConfig(remote_ip=args.remote_ip, id=args.robot_id,
                                     robot_model=args.robot_model)
robot = AlohaMiniClient(robot_config)


#dataset = LeRobotDataset("liyitenga/record_20250914225057", episodes=[EPISODE_IDX])
dataset_root = Path(args.dataset_root) if args.dataset_root else HF_LEROBOT_HOME / args.dataset_repo_id
info_path = dataset_root / "meta" / "info.json"
if not info_path.exists():
    raise FileNotFoundError(
        f"Local dataset metadata not found: {info_path}\n"
        "This replay script is configured to use local datasets only. "
        "Pass --root /path/to/dataset or make sure the dataset exists under "
        f"{HF_LEROBOT_HOME}."
    )

dataset = LeRobotDataset(args.dataset_repo_id, root=dataset_root, episodes=[args.dataset_episode])
actions = dataset.hf_dataset.select_columns(ACTION)
#print(f"Dataset loaded with id: {dataset.repo_id}, num_frames: {dataset.num_frames}")

robot.connect()

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

#log_say(f"Replaying episode {args.episode} from {args.dataset}")
print(f"Replaying episode {args.dataset_episode} from {args.dataset_repo_id}")
base_fps = args.replay_fps if args.replay_fps is not None else float(dataset.fps)
effective_fps = base_fps * args.replay_speed
frame_interval_s = 1.0 / effective_fps
print(
    f"Dataset FPS: {dataset.fps}; replay rate: {effective_fps:.3f} Hz; "
    f"expected duration: {dataset.num_frames / effective_fps:.2f}s"
)

next_frame_t = time.perf_counter()
try:
    for idx in range(dataset.num_frames):
        action = {
            name: float(actions[idx][ACTION][i])
            for i, name in enumerate(dataset.features[ACTION]["names"])
        }

        if args.verbose_actions:
            print(f"replay_bi.action:{action}")
        robot.send_action(action)

        # Schedule against an absolute deadline so occasional slow iterations do not
        # permanently add drift to the rest of the episode.
        next_frame_t += frame_interval_s
        precise_sleep(max(next_frame_t - time.perf_counter(), 0.0))
finally:
    if robot.is_connected:
        # Velocity commands are held by the host until another command (or the watchdog)
        # arrives. Stop the base immediately when replay finishes or is interrupted.
        robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        robot.disconnect()
