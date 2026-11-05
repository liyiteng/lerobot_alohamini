#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
from functools import cached_property

from lerobot.types import RobotAction
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_not_connected

from ..so_leader import SOLeader, SOLeaderTeleopConfig
from ..teleoperator import Teleoperator
from .config_bi_so_leader import BiSOLeaderConfig

logger = logging.getLogger(__name__)


class BiSOLeader(BimanualMixin, Teleoperator):
    """
    [Bimanual SO Leader Arms](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    """

    config_class = BiSOLeaderConfig
    name = "bi_so_leader"

    def __init__(self, config: BiSOLeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SOLeaderTeleopConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            arm_profile=config.left_arm_config.arm_profile,
            use_degrees=config.left_arm_config.use_degrees,
        )

        right_arm_config = SOLeaderTeleopConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            arm_profile=config.right_arm_config.arm_profile,
            use_degrees=config.right_arm_config.use_degrees,
        )

        self.left_arm = SOLeader(left_arm_config)
        self.right_arm = SOLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        left_arm_features = self.left_arm.action_features
        right_arm_features = self.right_arm.action_features

        return {
            **{f"left_{k}": v for k, v in left_arm_features.items()},
            **{f"right_{k}": v for k, v in right_arm_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action_dict = {}

        # Add "left_" prefix
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        # Add "right_" prefix
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def calibrate_one_step(self) -> None:
        """Execute 1-step calibration on both left and right arms."""
        self.left_arm.calibrate_one_step()
        self.right_arm.calibrate_one_step()

    @check_if_not_connected
    def auto_home(
        self,
        duration_s: float = 1.8,
        torque_limit: int = 220,
        hold_s: float = 0.8,
        release_after: bool = True,
    ) -> None:
        """Gently move both left and right leader arms in parallel to their 90-degree midpoint (2048)."""
        import threading

        t_left = threading.Thread(
            target=self.left_arm.auto_home,
            kwargs={
                "duration_s": duration_s,
                "torque_limit": torque_limit,
                "hold_s": hold_s,
                "release_after": release_after,
            },
            daemon=True,
        )
        t_right = threading.Thread(
            target=self.right_arm.auto_home,
            kwargs={
                "duration_s": duration_s,
                "torque_limit": torque_limit,
                "hold_s": hold_s,
                "release_after": release_after,
            },
            daemon=True,
        )
        t_left.start()
        t_right.start()
        t_left.join()
        t_right.join()

    @check_if_not_connected
    def sync_to_pose(
        self,
        target_pose: dict[str, float],
        duration_s: float = 1.5,
        torque_limit: int = 220,
        release_after: bool = True,
    ) -> None:
        """Smoothly align both leader arms over USB to match normalized joint values (e.g. Follower observation)."""
        import threading

        def _norm_to_raw(arm: SOLeader, prefix: str) -> dict[str, int]:
            ids_vals: dict[int, float] = {}
            for motor, m in arm.bus.motors.items():
                for candidate_key in (
                    f"arm_{prefix}_{motor}.pos",
                    f"{prefix}_{motor}.pos",
                    f"{motor}.pos",
                ):
                    if candidate_key in target_pose:
                        ids_vals[m.id] = float(target_pose[candidate_key])
                        break
            if not ids_vals or not arm.bus.calibration:
                return {m: 2048 for m in arm.bus.motors}
            try:
                raw_by_id = arm.bus._unnormalize(ids_vals)
                return {arm.bus._id_to_name(i): int(v) for i, v in raw_by_id.items()}
            except Exception:
                return {m: 2048 for m in arm.bus.motors}

        left_raw = _norm_to_raw(self.left_arm, "left")
        right_raw = _norm_to_raw(self.right_arm, "right")

        t_left = threading.Thread(
            target=self.left_arm.smooth_move_to_raw,
            kwargs={
                "target_raw": left_raw,
                "duration_s": duration_s,
                "torque_limit": torque_limit,
                "release_after": release_after,
                "hold_s": 0.25,
            },
            daemon=True,
        )
        t_right = threading.Thread(
            target=self.right_arm.smooth_move_to_raw,
            kwargs={
                "target_raw": right_raw,
                "duration_s": duration_s,
                "torque_limit": torque_limit,
                "release_after": release_after,
                "hold_s": 0.25,
            },
            daemon=True,
        )
        t_left.start()
        t_right.start()
        t_left.join()
        t_right.join()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        left_fb = {
            k.removeprefix("left_"): v
            for k, v in feedback.items()
            if k.startswith("left_")
        }
        right_fb = {
            k.removeprefix("right_"): v
            for k, v in feedback.items()
            if k.startswith("right_")
        }
        if left_fb:
            self.left_arm.send_feedback(left_fb)
        if right_fb:
            self.right_arm.send_feedback(right_fb)
