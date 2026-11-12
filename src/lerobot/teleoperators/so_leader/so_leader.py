# !/usr/bin/env python

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

import logging
import os
import time
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_so_leader import SOLeaderTeleopConfig

logger = logging.getLogger(__name__)


_ARM_PROFILES: dict[str, tuple[tuple[str, int, str, MotorNormMode | None], ...]] = {
    "so-arm-5dof": (
        ("shoulder_pan", 1, "sts3215", None),
        ("shoulder_lift", 2, "sts3215", None),
        ("elbow_flex", 3, "sts3215", None),
        ("wrist_flex", 4, "sts3215", None),
        ("wrist_roll", 5, "sts3215", None),
        ("gripper", 6, "sts3215", MotorNormMode.RANGE_0_100),
    ),
    "am-leader-6dof": (
        ("shoulder_pan", 1, "sts3215", None),
        ("shoulder_lift", 2, "sts3215", None),
        ("elbow_flex", 3, "sts3215", None),
        ("wrist_flex", 4, "sts3215", None),
        ("wrist_yaw", 5, "sts3215", None),
        ("wrist_roll", 6, "sts3215", None),
        ("gripper", 7, "sts3215", MotorNormMode.RANGE_0_100),
    ),
}

# Standard mechanical ROM limits around the 2047 half-turn center for STS3215 leader arms.
# Used for 1-step instant calibration and automatic healing of incomplete manual sweeps.
_GOLDEN_ROM_LIMITS: dict[str, tuple[int, int]] = {
    "shoulder_pan": (1024, 3072),
    "shoulder_lift": (907, 3164),
    "elbow_flex": (924, 3032),
    "wrist_flex": (1020, 3079),
    "wrist_yaw": (1037, 2917),
    "wrist_roll": (0, 4095),
    "gripper": (1431, 2678),
}


def _make_motors(arm_profile: str, norm_mode_body: MotorNormMode) -> dict[str, Motor]:
    if arm_profile not in _ARM_PROFILES:
        raise ValueError(
            f"Unknown arm_profile '{arm_profile}'. Expected one of: {list(_ARM_PROFILES.keys())}."
        )

    return {
        joint: Motor(motor_id, model, norm_mode or norm_mode_body)
        for joint, motor_id, model, norm_mode in _ARM_PROFILES[arm_profile]
    }


class SOLeader(Teleoperator):
    """Generic SO leader base for SO-100/101/10X teleoperators."""

    config_class = SOLeaderTeleopConfig
    name = "so_leader"

    def __init__(self, config: SOLeaderTeleopConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        motors = _make_motors(config.arm_profile, norm_mode_body)
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def _save_calibration(self, fpath: Path | None = None) -> None:
        super()._save_calibration(fpath)
        try:
            curr_path = self.calibration_fpath if fpath is None else fpath
            name = curr_path.name
            alt_name = None
            if "so101_leader_bi" in name:
                alt_name = name.replace("so101_leader_bi", "am_leader_bi")
            elif "am_leader_bi" in name:
                alt_name = name.replace("am_leader_bi", "so101_leader_bi")
            if alt_name:
                alt_path = curr_path.parent / alt_name
                import shutil
                shutil.copyfile(curr_path, alt_path)
        except Exception:
            pass

    def load_calibration(self, fpath: Path | None = None) -> dict[str, MotorCalibration]:
        try:
            curr_path = self.calibration_fpath if fpath is None else fpath
            if not curr_path.is_file():
                alt_name = None
                if "so101_leader_bi" in curr_path.name:
                    alt_name = curr_path.name.replace("so101_leader_bi", "am_leader_bi")
                elif "am_leader_bi" in curr_path.name:
                    alt_name = curr_path.name.replace("am_leader_bi", "so101_leader_bi")
                if alt_name and (curr_path.parent / alt_name).is_file():
                    curr_path = curr_path.parent / alt_name
            super().load_calibration(curr_path)
        except Exception:
            pass
        return self.calibration

    def _heal_calibration_rom(self) -> bool:
        """Ensure no joint has an abnormally narrow ROM (e.g. from an incomplete manual sweep)."""
        if not self.calibration:
            return False
        healed = False
        for motor, cal in self.calibration.items():
            if (cal.range_max - cal.range_min) < 600 and motor in _GOLDEN_ROM_LIMITS:
                g_min, g_max = _GOLDEN_ROM_LIMITS[motor]
                logger.warning(
                    f"[{self.id}] Healing narrow ROM on '{motor}' ({cal.range_min}..{cal.range_max} -> {g_min}..{g_max})"
                )
                cal.range_min = g_min
                cal.range_max = g_max
                healed = True
        if healed:
            self._save_calibration()
        return healed

    def sync_calibration_from_eeprom(self) -> bool:
        """Read calibration parameters stored inside the STS3215 servos' EEPROM over USB."""
        try:
            eeprom_cal = self.bus.read_calibration()
            has_valid_offset = any(abs(c.homing_offset) > 5 for c in eeprom_cal.values())
            if not has_valid_offset:
                return False
            for motor, cal in eeprom_cal.items():
                if (cal.range_max - cal.range_min) < 600 or (
                    cal.range_min == 0 and cal.range_max == 4095 and motor != "wrist_roll"
                ):
                    if motor in _GOLDEN_ROM_LIMITS:
                        cal.range_min, cal.range_max = _GOLDEN_ROM_LIMITS[motor]
            self.calibration = eeprom_cal
            self.bus.calibration = eeprom_cal
            self._save_calibration()
            logger.info(f"[{self.id}] Recovered calibration directly from servo EEPROM -> {self.calibration_fpath}")
            return True
        except Exception as e:
            logger.debug(f"[{self.id}] EEPROM calibration recovery skipped: {e}")
            return False

    def calibrate_one_step(self) -> dict[str, MotorCalibration]:
        """Instant 1-step calibration: lock 90-degree midpoint and apply Golden ROM template."""
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        homing_offsets = self.bus.set_half_turn_homings()
        self.calibration = {}
        for motor, m in self.bus.motors.items():
            g_min, g_max = _GOLDEN_ROM_LIMITS.get(motor, (900, 3160))
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=g_min,
                range_max=g_max,
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")
        return self.calibration

    def calibrate(self) -> None:
        if self.calibration:
            self._heal_calibration_rom()
            if os.environ.get("LEROBOT_AUTO_USE_CALIBRATION", "").strip().lower() in {"1", "true", "yes"} and not getattr(self, "_force_interactive", False):
                logger.info(f"Auto-applying saved calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return
            if not getattr(self, "_force_interactive", False):
                # Calibration file exists, ask user whether to use it or run new calibration
                user_input = input(
                    f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
                )
                if user_input.strip().lower() != "c":
                    logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                    self.bus.write_calibration(self.calibration)
                    return

        # Try recovering calibration directly from the servos' internal EEPROM before prompting or failing
        if not self.calibration and not getattr(self, "_force_interactive", False):
            if self.sync_calibration_from_eeprom():
                self.bus.write_calibration(self.calibration)
                return

        # If not interactive calibration, prevent hanging on input()
        if os.environ.get("LEROBOT_AUTO_USE_CALIBRATION", "").strip().lower() in {"1", "true", "yes"} and not getattr(self, "_force_interactive", False):
            raise RuntimeError(
                f"主控臂 [{self.id}] 尚未完成校准！请先运行「步骤 2：机械臂校准」，缺少标定文件: {self.calibration_file}"
            )

        if getattr(self, "_one_step_mode", False):
            self.calibrate_one_step()
            return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self._heal_calibration_rom()
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    @check_if_not_connected
    def smooth_move_to_raw(
        self,
        target_raw: dict[str, int],
        duration_s: float = 1.8,
        torque_limit: int = 220,
        release_after: bool = True,
        hold_s: float = 0.4,
        fail_on_yield: bool = False,
    ) -> None:
        """Gently drive the leader arm over USB to target raw encoder positions with compliance protection."""
        import math

        motors = [m for m in self.bus.motors if m in target_raw]
        if not motors:
            return

        start_pos = self.bus.sync_read("Present_Position", motors, normalize=False)
        try:
            for m in motors:
                self.bus.write("Operating_Mode", m, OperatingMode.POSITION.value)
                self.bus.write("Acceleration", m, 15, normalize=False)
                self.bus.write("Goal_Velocity", m, 360, normalize=False)
                self.bus.write("Torque_Limit", m, torque_limit, normalize=False)
                self.bus.write("Goal_Position", m, int(start_pos[m]), normalize=False)
                self.bus.write("Torque_Enable", m, 1, normalize=False)

            steps = max(10, int(duration_s * 30))
            dt = duration_s / steps
            blocked_motors: set[str] = set()

            for step in range(1, steps + 1):
                alpha = 0.5 * (1.0 - math.cos(math.pi * step / steps))
                goals: dict[str, int] = {}
                for m in motors:
                    if m in blocked_motors:
                        continue
                    s = float(start_pos[m])
                    t = float(target_raw[m])
                    goals[m] = int(round(s + (t - s) * alpha))

                if goals:
                    self.bus.sync_write("Goal_Position", goals, normalize=False)

                # Check current for gentle obstacle yielding (> 420mA = ~65 raw units)
                if step % 3 == 0:
                    try:
                        currents = self.bus.sync_read("Present_Current", motors, normalize=False)
                        for m, raw_cur in currents.items():
                            if abs(int(raw_cur)) > 65 and m not in blocked_motors:
                                logger.info(f"[{self.id}] Motor '{m}' yielded to external touch/obstacle")
                                blocked_motors.add(m)
                                self.bus.write("Torque_Enable", m, 0, normalize=False)
                                if fail_on_yield:
                                    raise RuntimeError(f"关节 {m} 触发避让，姿态贴合未完成")
                    except Exception:
                        if fail_on_yield:
                            raise

                time.sleep(dt)

            if hold_s > 0:
                time.sleep(hold_s)

        finally:
            if release_after:
                self.bus.disable_torque()

    @check_if_not_connected
    def auto_home(
        self,
        duration_s: float = 1.8,
        torque_limit: int = 220,
        hold_s: float = 0.8,
        release_after: bool = True,
    ) -> None:
        """Gently return all leader arm joints to their 90-degree calibrated midpoint (2048)."""
        targets = {m: 2048 for m in self.bus.motors}
        self.smooth_move_to_raw(
            targets,
            duration_s=duration_s,
            torque_limit=torque_limit,
            hold_s=hold_s,
            release_after=release_after,
        )

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def enable_torque(self) -> None:
        self.bus.enable_torque()

    def disable_torque(self) -> None:
        self.bus.disable_torque()

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        raw_positions = self.bus.sync_read("Present_Position", normalize=False)
        ids_values = {self.bus.motors[motor].id: int(val) for motor, val in raw_positions.items()}
        try:
            norm_values = (
                self.bus._normalize(ids_values)
                if "Present_Position" in self.bus.normalized_data
                else ids_values
            )
            action = {
                f"{self.bus._id_to_name(id_)}.pos": val for id_, val in norm_values.items()
            }
        except RuntimeError:
            action = {f"{motor}.pos": float(val) for motor, val in raw_positions.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        goals = {k.removesuffix(".pos"): v for k, v in feedback.items() if k.endswith(".pos")}
        if goals:
            self.bus.sync_write("Goal_Position", goals)

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")


SO100Leader = SOLeader
SO101Leader = SOLeader
