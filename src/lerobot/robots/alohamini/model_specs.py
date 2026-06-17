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

ARM_PROFILE_JOINTS: dict[str, tuple[str, ...]] = {
    "so-arm-5dof": (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ),
    "am-leader-6dof": (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "gripper",
    ),
    "am-follower-6dof": (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "gripper",
    ),
    "am-follower-6dof-hd": (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "gripper",
    ),
}


# Per-model hardware specifications. `robot_model` is the customer-facing whole-robot SKU;
# `arm_profile` selects the follower arm hardware mounted on that robot.
ROBOT_SPECS: dict[str, dict] = {
    "alohamini1": {
        "arm_profile": "so-arm-5dof",
        "base_motor": "sts3215",
        "lift_motor": "sts3215",
        "lead_mm_per_rev": 84.0,
    },
    "alohamini2": {
        "arm_profile": "am-follower-6dof",
        "base_motor": "sts3215",
        "lift_motor": "sts3095",
        "lead_mm_per_rev": 131.0,
    },
    "alohamini2pro": {
        "arm_profile": "am-follower-6dof-hd",
        "base_motor": "sts3250",
        "lift_motor": "sts3095",
        "lead_mm_per_rev": 131.0,
    },
}


def validate_robot_model(robot_model: str) -> dict:
    if robot_model not in ROBOT_SPECS:
        raise ValueError(
            f"Unknown robot_model '{robot_model}'. "
            f"Expected one of: {list(ROBOT_SPECS.keys())}."
        )

    return ROBOT_SPECS[robot_model]


def arm_state_keys(prefix: str, arm_profile: str) -> tuple[str, ...]:
    if arm_profile not in ARM_PROFILE_JOINTS:
        raise ValueError(
            f"Unknown arm_profile '{arm_profile}'. Expected one of: {list(ARM_PROFILE_JOINTS.keys())}."
        )

    return tuple(f"{prefix}_{joint}.pos" for joint in ARM_PROFILE_JOINTS[arm_profile])


def arm_state_keys_for_robot_model(robot_model: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    arm_profile = validate_robot_model(robot_model)["arm_profile"]
    return arm_state_keys("arm_left", arm_profile), arm_state_keys("arm_right", arm_profile)
