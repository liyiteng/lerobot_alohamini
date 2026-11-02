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

"""Calibrate the two leader arms used by AlohaMini bimanual teleoperation.

Run this before ``teleoperate_bi.py`` to calibrate both arms independently of
the robot. The argument defaults intentionally match ``teleoperate_bi.py`` so
that both scripts read and write the same calibration files.
"""

import argparse

from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.utils import init_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="Leader arm profile selector",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_logging()

    leader = BiSOLeader(
        BiSOLeaderConfig(
            left_arm_config=SOLeaderConfig(
                port="/dev/cu.usbmodem5B910458311",
                arm_profile=args.arm_profile,
            ),
            right_arm_config=SOLeaderConfig(
                port="/dev/cu.usbmodem5B790175581",
                arm_profile=args.arm_profile,
            ),
            id=args.leader_id,
        )
    )

    leader.connect(calibrate=False)
    try:
        leader.calibrate()
    finally:
        leader.disconnect()


if __name__ == "__main__":
    main()
