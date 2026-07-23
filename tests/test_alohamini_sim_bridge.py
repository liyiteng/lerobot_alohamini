#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the alohamini_sim -> LeRobotDataset bridge.

Builds tiny synthetic engine episodes (10 frames, 2 cameras at 96x96, 18-D qpos / 16-D
sim action) and checks that the bridge writes a valid v3.0 LeRobotDataset that reloads
with the expected feature keys, dtypes, shapes, and joint/pixel/task round-trips.
"""

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from alohamini_sim.data_engine.lerobot_bridge import (
    ROBOT_STATE_NAMES,
    SIM_ACTION_NAMES,
    SIM_STATE_NAMES,
    convert_state_action,
    write_episodes,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "alohamini-sim/test-bridge"
FPS = 20
NUM_STEPS = 10
CAMERAS = {"top": (60, 120, 180), "wrist": (200, 40, 90)}  # name -> constant RGB fill
HEIGHT, WIDTH = 96, 96

QPOS_X = SIM_STATE_NAMES.index("root_x_axis_joint")
QPOS_Y = SIM_STATE_NAMES.index("root_y_axis_joint")
QPOS_LIFT = SIM_STATE_NAMES.index("vertical_move")


def _make_qpos(t: int) -> np.ndarray:
    """18-D sim qpos: base drives +x at 0.2 m/s, yaw 0, lift 0.05 m, arms on known ramps."""
    qpos = np.zeros(len(SIM_STATE_NAMES), dtype=np.float32)
    qpos[QPOS_X] = 0.01 * t
    qpos[QPOS_Y] = 0.3
    qpos[QPOS_LIFT] = 0.05
    for i in range(1, 6):
        qpos[SIM_STATE_NAMES.index(f"left_joint{i}")] = 0.1 * i + 0.01 * t
        qpos[SIM_STATE_NAMES.index(f"right_joint{i}")] = 0.2 * i - 0.01 * t
    qpos[SIM_STATE_NAMES.index("left_finger_joint1")] = 0.0175
    qpos[SIM_STATE_NAMES.index("right_finger_joint1")] = 0.035
    return qpos


def _make_sim_action(t: int) -> np.ndarray:
    """16-D sim action: root targets 5 mm ahead in +x, lift target 0.06 m, arms on known ramps."""
    action = np.zeros(len(SIM_ACTION_NAMES), dtype=np.float32)
    action[SIM_ACTION_NAMES.index("root_x_axis_joint_target")] = 0.01 * t + 0.005
    action[SIM_ACTION_NAMES.index("root_y_axis_joint_target")] = 0.3
    action[SIM_ACTION_NAMES.index("vertical_move_target")] = 0.06
    for i in range(1, 6):
        action[SIM_ACTION_NAMES.index(f"left_joint{i}_target")] = 0.1 * i
        action[SIM_ACTION_NAMES.index(f"right_joint{i}_target")] = 0.2 * i
    action[SIM_ACTION_NAMES.index("left_gripper_target")] = 0.0
    action[SIM_ACTION_NAMES.index("right_gripper_target")] = 0.035
    return action


def _make_episode(task: str, with_images: bool = True) -> dict:
    steps = []
    for t in range(NUM_STEPS):
        rgb = None
        if with_images:
            rgb = {cam: np.full((HEIGHT, WIDTH, 3), color, dtype=np.uint8) for cam, color in CAMERAS.items()}
        steps.append({"qpos": _make_qpos(t), "action": _make_sim_action(t), "rgb": rgb})
    return {"steps": steps, "annotation": task, "seed": 0, "success": True, "command": {"verb": "move"}}


def test_convert_state_action_mapping():
    """Arm joints copy through, base becomes body-frame velocities, lift becomes millimeters."""
    qpos = np.stack([_make_qpos(t) for t in range(NUM_STEPS)])
    sim_action = np.stack([_make_sim_action(t) for t in range(NUM_STEPS)])
    state, action = convert_state_action(qpos, sim_action, FPS, gripper_range=(0.0, 0.035))

    assert state.shape == (NUM_STEPS, 16) and state.dtype == np.float32
    assert action.shape == (NUM_STEPS, 16) and action.dtype == np.float32

    t = 4
    np.testing.assert_allclose(state[t, 0], 0.1 + 0.01 * t, atol=1e-6)  # arm_left_shoulder_pan.pos
    np.testing.assert_allclose(state[t, 5], 50.0, atol=1e-4)  # arm_left_gripper.pos (mid-range)
    np.testing.assert_allclose(state[t, 11], 100.0, atol=1e-4)  # arm_right_gripper.pos (fully open)
    np.testing.assert_allclose(state[0, 12], 0.0, atol=1e-6)  # x.vel first frame
    np.testing.assert_allclose(state[t, 12], 0.01 * FPS, atol=1e-4)  # x.vel = 0.2 m/s
    np.testing.assert_allclose(state[t, 13:15], 0.0, atol=1e-4)  # y.vel, theta.vel
    np.testing.assert_allclose(state[t, 15], 50.0, atol=1e-4)  # lift_axis.height_mm
    np.testing.assert_allclose(action[t, 12], 0.005 * FPS, atol=1e-4)  # reaching velocity 0.1 m/s
    np.testing.assert_allclose(action[t, 15], 60.0, atol=1e-4)  # lift target in mm


def test_bridge_write_and_reload(tmp_path):
    """Full round-trip: write two episodes with videos, reload, check features and values."""
    root = tmp_path / "ds"
    episodes = [_make_episode("move the red cube"), _make_episode("push the mug")]
    write_episodes(episodes, repo_id=REPO_ID, root=root, fps=FPS, gripper_range=(0.0, 0.035))

    dataset = LeRobotDataset(REPO_ID, root=root)
    assert dataset.num_episodes == 2
    assert dataset.num_frames == 2 * NUM_STEPS
    assert dataset.fps == FPS

    expected_keys = {"action", "observation.state"} | {f"observation.images.{cam}" for cam in CAMERAS}
    assert expected_keys <= set(dataset.features)
    assert dataset.features["observation.state"]["names"] == ROBOT_STATE_NAMES
    assert dataset.features["action"]["names"] == ROBOT_STATE_NAMES
    assert dataset.features["observation.state"]["dtype"] == "float32"
    assert dataset.features["observation.state"]["shape"] == (16,)
    for cam in CAMERAS:
        camera_ft = dataset.features[f"observation.images.{cam}"]
        assert camera_ft["dtype"] == "video"
        assert tuple(camera_ft["shape"]) == (HEIGHT, WIDTH, 3)

    t = 4
    item = dataset[t]
    assert item["observation.state"].dtype == torch.float32
    assert item["observation.state"].shape == (16,)
    assert item["action"].dtype == torch.float32
    assert item["task"] == "move the red cube"

    # Joint value round-trip (state is written exactly, so equality is tight).
    torch.testing.assert_close(item["observation.state"][0], torch.tensor(0.1 + 0.01 * t), atol=1e-5, rtol=0)
    torch.testing.assert_close(item["observation.state"][15], torch.tensor(50.0), atol=1e-4, rtol=0)
    torch.testing.assert_close(item["action"][12], torch.tensor(0.005 * FPS), atol=1e-4, rtol=0)

    # Pixel round-trip (video encoding is lossy: compare against the constant fill color).
    for cam, color in CAMERAS.items():
        image = item[f"observation.images.{cam}"]
        assert image.dtype == torch.float32
        assert image.shape == (3, HEIGHT, WIDTH)
        expected = torch.tensor(color, dtype=torch.float32) / 255.0
        torch.testing.assert_close(image.mean(dim=(1, 2)), expected, atol=0.04, rtol=0)

    # Second episode keeps its own task.
    assert dataset[NUM_STEPS + 1]["task"] == "push the mug"


def test_bridge_state_only_episode(tmp_path):
    """Episodes recorded without rendering (rgb=None) produce a valid state-only dataset."""
    root = tmp_path / "ds_state_only"
    write_episodes([_make_episode("blind move", with_images=False)], repo_id=REPO_ID, root=root, fps=FPS)

    dataset = LeRobotDataset(REPO_ID, root=root)
    assert dataset.num_frames == NUM_STEPS
    assert not any(key.startswith("observation.images.") for key in dataset.features)
    torch.testing.assert_close(dataset[0]["observation.state"][15], torch.tensor(50.0), atol=1e-4, rtol=0)


def test_bridge_cli(tmp_path):
    """The python -m CLI converts a pickled episode list into a loadable dataset."""
    episodes_path = tmp_path / "episodes.pkl"
    with episodes_path.open("wb") as f:
        pickle.dump([_make_episode("cli move")], f)

    root = tmp_path / "ds_cli"
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alohamini_sim.data_engine.lerobot_bridge",
            "--episodes",
            str(episodes_path),
            "--repo-id",
            REPO_ID,
            "--root",
            str(root),
            "--fps",
            str(FPS),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    dataset = LeRobotDataset(REPO_ID, root=root)
    assert dataset.num_frames == NUM_STEPS
    assert dataset[0]["task"] == "cli move"
