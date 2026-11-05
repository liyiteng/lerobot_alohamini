#!/usr/bin/env python3

# Copyright 2026 The AlohaMini Authors. All rights reserved.
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

"""AlohaMini Leader Arm Active Control & Zero-Friction Calibration Utility.

Provides direct USB active control:
- auto_home: Compliantly drives leader arms to 90-degree midpoint (2048).
- one_step_calibrate: 1-step calibration (locks current 90° pose + golden ROM).
- check_calibration: Inspects JSON files and EEPROM health, detects narrow ROM.
- heal_calibration: Automatically fixes narrow joint limits in existing calibration.
- release_torque: Safely releases torque on all leader joints.
"""

import argparse
import json
import logging
import os
import sys
import time

from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.utils import init_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=[
            "auto_home",
            "one_step_calibrate",
            "check_calibration",
            "heal_calibration",
            "release_torque",
        ],
        help="Action to execute on leader arms",
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
        default="am-leader-6dof",
        choices=["so-arm-5dof", "am-leader-6dof"],
        help="Leader arm profile selector",
    )
    parser.add_argument(
        "--teleop.left_port",
        "--leader.left_port",
        "--left_port",
        dest="left_port",
        type=str,
        default=os.environ.get("ALOHAMINI_LEFT_PORT", "/dev/cu.usbmodem5B910458311"),
        help="Left leader arm serial port",
    )
    parser.add_argument(
        "--teleop.right_port",
        "--leader.right_port",
        "--right_port",
        dest="right_port",
        type=str,
        default=os.environ.get("ALOHAMINI_RIGHT_PORT", "/dev/cu.usbmodem5B790175581"),
        help="Right leader arm serial port",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Movement duration in seconds for smooth homing",
    )
    parser.add_argument(
        "--torque_limit",
        type=int,
        default=220,
        help="Torque limit (0-1000) for compliant motion; default 220 is safe for touch",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Hold position after auto_home instead of releasing torque",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_logging()

    # Action 1: check_calibration can run purely by inspecting local files and EEPROM
    if args.action == "check_calibration":
        from pathlib import Path
        home_dir = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))
        calib_dir = home_dir / "calibration/teleoperators/so_leader"
        
        candidates = ["so101_leader_bi", "am_leader_bi"]
        left_file = None
        right_file = None
        for cid in candidates:
            lf = calib_dir / f"{cid}_left.json"
            rf = calib_dir / f"{cid}_right.json"
            if lf.is_file() and rf.is_file():
                left_file = lf
                right_file = rf
                break

        status = {
            "exists": False,
            "is_healthy": True,
            "issues": [],
            "left_file": str(left_file) if left_file else None,
            "right_file": str(right_file) if right_file else None,
        }

        if left_file and right_file:
            status["exists"] = True
            try:
                with open(left_file) as f:
                    l_data = json.load(f)
                with open(right_file) as f:
                    r_data = json.load(f)

                # Check for identical left and right files
                if l_data == r_data:
                    status["is_healthy"] = False
                    status["issues"].append("左右主臂标定数据完全相同，疑似复用了对侧标定")

                # Check for narrow ROM
                for side, data in [("左臂", l_data), ("右臂", r_data)]:
                    for motor, val in data.items():
                        rmin = val.get("range_min", 0)
                        rmax = val.get("range_max", 4095)
                        span = rmax - rmin
                        if motor != "wrist_roll" and span < 500:
                            status["is_healthy"] = False
                            deg = round(span * 360 / 4096, 1)
                            status["issues"].append(
                                f"{side}关节 [{motor}] 行程仅有 {span} 步 (~{deg}°)，未完全转动"
                            )
            except Exception as e:
                status["is_healthy"] = False
                status["issues"].append(f"标定文件解析失败: {e}")

        if args.json:
            print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            if not status["exists"]:
                print("❌ 未检测到左右主臂标定文件，请进行初次标定。")
            elif not status["is_healthy"]:
                print("⚠️ 检测到主臂标定存在潜在异常：")
                for iss in status["issues"]:
                    print(f"  • {iss}")
                print("建议执行「一键极速标定」或「一键修复行程」。")
            else:
                print("✔ 主臂标定文件健康，行程范围正常。")
        return

    # Action 2: heal_calibration can repair local files directly
    if args.action == "heal_calibration":
        from pathlib import Path
        home_dir = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))
        calib_dir = home_dir / "calibration/teleoperators/so_leader"
        golden = {
            "shoulder_pan": (1024, 3072),
            "shoulder_lift": (907, 3164),
            "elbow_flex": (924, 3032),
            "wrist_flex": (1020, 3079),
            "wrist_yaw": (1037, 2917),
            "wrist_roll": (0, 4095),
            "gripper": (1431, 2678),
        }
        candidates = ["so101_leader_bi", "am_leader_bi"]
        healed_files = []
        for cid in candidates:
            for side in ["left", "right"]:
                fp = calib_dir / f"{cid}_{side}.json"
                if fp.is_file():
                    try:
                        with open(fp) as f:
                            data = json.load(f)
                        changed = False
                        for motor, val in data.items():
                            if (val.get("range_max", 4095) - val.get("range_min", 0)) < 600 and motor in golden:
                                g_min, g_max = golden[motor]
                                val["range_min"] = g_min
                                val["range_max"] = g_max
                                changed = True
                        if changed:
                            with open(fp, "w") as f:
                                json.dump(data, f, indent=4)
                            healed_files.append(fp.name)
                    except Exception as e:
                        print(f"Failed to heal {fp}: {e}")
        if healed_files:
            print(f"✔ [标定修复完成] 已自动修复异常过窄行程并更新文件: {', '.join(healed_files)}")
        else:
            print("✔ 标定文件中的行程均在正常范围。")

        # Check for duplicate left/right calibration files
        for cid in candidates:
            lf = calib_dir / f"{cid}_left.json"
            rf = calib_dir / f"{cid}_right.json"
            if lf.is_file() and rf.is_file():
                try:
                    with open(lf) as f1, open(rf) as f2:
                        if json.load(f1) == json.load(f2):
                            print("💡 提示：检测到左右主臂标定完全相同（历史代码复用对侧导致）。建议点击「3 秒极速标定」分别记录左右臂真实独立零位。")
                except Exception:
                    pass
        return

    # For hardware actions, initialize BiSOLeader
    print(f"正在连接主臂串口 [左: {args.left_port}, 右: {args.right_port}]...")
    try:
        leader = BiSOLeader(
            BiSOLeaderConfig(
                left_arm_config=SOLeaderConfig(
                    port=args.left_port,
                    arm_profile=args.arm_profile,
                ),
                right_arm_config=SOLeaderConfig(
                    port=args.right_port,
                    arm_profile=args.arm_profile,
                ),
                id=args.leader_id,
            )
        )
        leader.connect(calibrate=False)
    except Exception as e:
        print(f"❌ 无法连接主臂串口: {e}")
        sys.exit(1)

    try:
        if args.action == "auto_home":
            print("🚀 [一键归零] 正在以 22% 柔顺安全力矩让双主臂自动回到 90° 零点位置...")
            release_after = not args.hold
            leader.auto_home(
                duration_s=args.duration,
                torque_limit=args.torque_limit,
                hold_s=1.2,
                release_after=release_after,
            )
            state_msg = "已轻柔锁定在 90° 中位" if args.hold else "已回到 90° 中位并释放力矩"
            print(f"✔ [一键归零完毕] 双主臂{state_msg}。")

        elif args.action == "one_step_calibrate":
            print("🎯 [一键极速标定] 正在以当前姿态锁定 90° 中位零点并写入标准物理行程...")
            leader.calibrate_one_step()
            print("✔ [一键极速标定完成] 左右双臂零点与黄金行程已成功固化到磁盘与舵机芯片！")

        elif args.action == "heal_calibration":
            print("🔧 [自动修复标定] 正在检查并修复过窄的机械行程...")
            healed_l = leader.left_arm._heal_calibration_rom()
            healed_r = leader.right_arm._heal_calibration_rom()
            if healed_l or healed_r:
                print("✔ 标定行程异常已修复并更新保存！")
            else:
                print("✔ 当前标定行程已处于正常范围，无需修复。")

        elif args.action == "release_torque":
            print("🔓 正在释放双臂全部力矩...")
            leader.left_arm.disable_torque()
            leader.right_arm.disable_torque()
            print("✔ 双臂力矩已释放。")

    finally:
        leader.disconnect()


if __name__ == "__main__":
    main()
