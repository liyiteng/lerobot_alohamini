#!/usr/bin/env python

"""AlohaMini Follower Robot Calibration & Diagnostics CLI.

Features:
- check_calibration: Inspect follower arms calibration health (ROM width, homing offsets).
- heal_calibration: Repair corrupt/narrow ROM limits in AlohaMiniRobot.json.
- one_step_calibrate: Generate golden standard calibration for AlohaMini follower arms.
- remote_heal: Push healed calibration to remote Pi host via SSH.

Usage:
    python follower_control.py --action=check_calibration
    python follower_control.py --action=one_step_calibrate
    python follower_control.py --action=heal_calibration
    python follower_control.py --action=remote_heal --remote_ip=192.168.10.2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Standard AlohaMini 2 Pro Golden Follower Limits
FOLLOWER_GOLDEN_LIMITS = {
    "left_arm_shoulder_pan": {"id": 1, "range_min": 1024, "range_max": 3072},
    "left_arm_shoulder_lift": {"id": 2, "range_min": 900, "range_max": 3150},
    "left_arm_elbow_flex": {"id": 3, "range_min": 900, "range_max": 3050},
    "left_arm_wrist_flex": {"id": 4, "range_min": 1000, "range_max": 3100},
    "left_arm_wrist_yaw": {"id": 5, "range_min": 1000, "range_max": 3000},
    "left_arm_wrist_roll": {"id": 6, "range_min": 0, "range_max": 4095},
    "left_arm_gripper": {"id": 7, "range_min": 1400, "range_max": 2700},
    "right_arm_shoulder_pan": {"id": 1, "range_min": 1024, "range_max": 3072},
    "right_arm_shoulder_lift": {"id": 2, "range_min": 900, "range_max": 3150},
    "right_arm_elbow_flex": {"id": 3, "range_min": 900, "range_max": 3050},
    "right_arm_wrist_flex": {"id": 4, "range_min": 1000, "range_max": 3100},
    "right_arm_wrist_yaw": {"id": 5, "range_min": 1000, "range_max": 3000},
    "right_arm_wrist_roll": {"id": 6, "range_min": 0, "range_max": 4095},
    "right_arm_gripper": {"id": 7, "range_min": 1400, "range_max": 2700},
    "base_left_wheel": {"id": 8, "range_min": 0, "range_max": 4095},
    "base_right_wheel": {"id": 9, "range_min": 0, "range_max": 4095},
    "base_lift": {"id": 10, "range_min": 0, "range_max": 4095},
}


def get_default_calib_path(robot_id: str = "AlohaMiniRobot") -> Path:
    calib_dir = Path.home() / ".cache/huggingface/lerobot/calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    return calib_dir / f"{robot_id}.json"


def check_calibration(fpath: Path) -> dict:
    if not fpath.is_file():
        return {
            "exists": False,
            "is_healthy": False,
            "path": str(fpath),
            "issue_hint": "未找到从臂标定文件",
            "motors": {},
        }

    try:
        data = json.loads(fpath.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "exists": True,
            "is_healthy": False,
            "path": str(fpath),
            "issue_hint": f"标定文件损坏或格式错误: {e}",
            "motors": {},
        }

    motor_issues = []
    motors_summary = {}

    for name, limits in FOLLOWER_GOLDEN_LIMITS.items():
        if name in data:
            entry = data[name]
            rmin = entry.get("range_min", 0)
            rmax = entry.get("range_max", 4095)
            span = rmax - rmin
            motors_summary[name] = {
                "id": entry.get("id"),
                "homing_offset": entry.get("homing_offset", 0),
                "range_min": rmin,
                "range_max": rmax,
                "span": span,
            }
            # Check for abnormal narrow ROM (< 1200 ticks for arm joints)
            if "wheel" not in name and "lift" not in name and "gripper" not in name:
                if span < 1200:
                    motor_issues.append(f"{name} 行程过窄 ({span} 步)")

    is_healthy = len(motor_issues) == 0 and len(motors_summary) >= 14
    issue_hint = "; ".join(motor_issues) if motor_issues else None

    return {
        "exists": True,
        "is_healthy": is_healthy,
        "path": str(fpath),
        "issue_hint": issue_hint,
        "motors": motors_summary,
    }


def generate_golden_calibration(existing_data: dict | None = None) -> dict:
    calib = {}
    for name, golden in FOLLOWER_GOLDEN_LIMITS.items():
        homing_offset = 0
        drive_mode = 0
        if existing_data and name in existing_data:
            homing_offset = existing_data[name].get("homing_offset", 0)
            drive_mode = existing_data[name].get("drive_mode", 0)

        calib[name] = {
            "id": golden["id"],
            "drive_mode": drive_mode,
            "homing_offset": homing_offset,
            "range_min": golden["range_min"],
            "range_max": golden["range_max"],
        }
    return calib


def heal_calibration(fpath: Path) -> dict:
    existing = {}
    if fpath.is_file():
        try:
            existing = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    healed = generate_golden_calibration(existing)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(json.dumps(healed, indent=4), encoding="utf-8")
    return {
        "success": True,
        "path": str(fpath),
        "message": "从臂黄金标准标定参数已就绪并保存",
    }


def remote_heal(remote_ip: str, robot_id: str = "AlohaMiniRobot") -> dict:
    """Push golden calibration to remote Raspberry Pi host."""
    remote_ip = remote_ip.strip()
    healed = generate_golden_calibration()
    payload = json.dumps(healed, indent=2)

    remote_dir = "~/.cache/huggingface/lerobot/calibration"
    remote_file = f"{remote_dir}/{robot_id}.json"

    # Use SSH to write remote file
    cmd = f"mkdir -p {remote_dir} && cat << 'EOF' > {remote_file}\n{payload}\nEOF\n"
    res = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=4", "-o", "StrictHostKeyChecking=no", f"pi@{remote_ip}", cmd],
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        return {
            "success": True,
            "remote_ip": remote_ip,
            "message": f"成功将从臂黄金标定参数推送到树莓派 ({remote_file})",
        }
    else:
        return {
            "success": False,
            "remote_ip": remote_ip,
            "error": res.stderr or "SSH 连接或写入失败",
        }


def main():
    parser = argparse.ArgumentParser(description="AlohaMini Follower Robot Calibration CLI")
    parser.add_argument(
        "--action",
        type=str,
        default="check_calibration",
        choices=["check_calibration", "one_step_calibrate", "heal_calibration", "remote_heal"],
        help="Action to execute",
    )
    parser.add_argument("--robot_id", type=str, default="AlohaMiniRobot", help="Robot ID")
    parser.add_argument("--remote_ip", type=str, default="192.168.10.2", help="Remote Pi Host IP")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    calib_path = get_default_calib_path(args.robot_id)

    if args.action == "check_calibration":
        res = check_calibration(calib_path)
        if args.json:
            print(json.dumps(res))
        else:
            print(f"Calibration exists: {res['exists']}, is_healthy: {res['is_healthy']}")
            if res["issue_hint"]:
                print(f"Issue: {res['issue_hint']}")

    elif args.action in ("one_step_calibrate", "heal_calibration"):
        res = heal_calibration(calib_path)
        if args.json:
            print(json.dumps(res))
        else:
            print(res["message"])

    elif args.action == "remote_heal":
        res = remote_heal(args.remote_ip, args.robot_id)
        if args.json:
            print(json.dumps(res))
        else:
            print(res["message"] if res["success"] else f"Error: {res.get('error')}")


if __name__ == "__main__":
    main()
