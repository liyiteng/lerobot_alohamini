#!/usr/bin/env python

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

import logging
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_alohamini import AlohaMiniConfig
from .lift_axis import LiftAxis, LiftAxisConfig
from .model_specs import arm_state_keys_for_robot_model, validate_robot_model

logger = logging.getLogger(__name__)

_CURRENT_MA_PER_RAW_UNIT = 6.5
_JOINT_COLLISION_DURATION_S = 0.150
_JOINT_STALL_MIN_COMMAND_ERROR_DEG = 2.0
_JOINT_STALL_MIN_PROGRESS_DEG = 0.2
_SUSTAINED_OVERCURRENT_DURATION_S = 0.650
_NEAR_STALL_OVERCURRENT_DURATION_S = 0.080
_COLLISION_RATED_CURRENT_MULTIPLIER = 1.5
_SUSTAINED_RATED_CURRENT_MULTIPLIER = 2.0
_NEAR_STALL_CURRENT_FRACTION = 0.8

# Motor datasheet values: (rated current, stall current), in mA.
_MOTOR_CURRENT_RATINGS_MA: dict[str, tuple[float, float]] = {
    "sts3215": (900.0, 2700.0),
    "sts3095": (2200.0, 9800.0),
    "sts3250": (1400.0, 4200.0),
}


@dataclass(frozen=True)
class _MotorCurrentLimits:
    collision_ma: float
    sustained_ma: float
    near_stall_ma: float


@dataclass(frozen=True)
class _JointStallCandidate:
    started_at: float
    start_position: float
    command_direction: float


def _current_limits_for_motor(motor: Motor) -> _MotorCurrentLimits:
    try:
        rated_ma, stall_ma = _MOTOR_CURRENT_RATINGS_MA[motor.model]
    except KeyError as exc:
        raise ValueError(f"Missing current ratings for motor model '{motor.model}'.") from exc

    return _MotorCurrentLimits(
        collision_ma=_COLLISION_RATED_CURRENT_MULTIPLIER * rated_ma,
        sustained_ma=_SUSTAINED_RATED_CURRENT_MULTIPLIER * rated_ma,
        near_stall_ma=_NEAR_STALL_CURRENT_FRACTION * stall_ma,
    )


def _position_delta_degrees(bus: FeetechMotorsBus, motor: str, delta: float) -> float:
    """Convert a position difference to degrees without changing the command space."""
    actuator = bus.motors[motor]
    if actuator.norm_mode is MotorNormMode.DEGREES:
        return delta
    calibration = bus.calibration[motor]
    if actuator.norm_mode is MotorNormMode.RANGE_M100_100:
        norm_range = 200.0
    elif actuator.norm_mode is MotorNormMode.RANGE_0_100:
        norm_range = 100.0
    else:
        raise ValueError(f"Unsupported normalization for {motor}: {actuator.norm_mode}")
    resolution = bus.model_resolution_table[actuator.model] - 1
    return delta * (calibration.range_max - calibration.range_min) / norm_range * 360 / resolution


def _has_sustained_overcurrent(
    started_at: dict[str, float],
    motor: str,
    current_ma: float,
    limit_ma: float,
    now: float,
    duration_s: float,
) -> bool:
    if current_ma < limit_ma:
        started_at.pop(motor, None)
        return False

    if motor not in started_at:
        started_at[motor] = now
        return duration_s <= 0.0
    return now - started_at[motor] >= duration_s


# Per-arm hardware profiles. Keep the profile name about the arm itself:
# role, DOF, and motor class. Whole-robot SKUs are mapped separately below.
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
    "am-follower-6dof": (
        ("shoulder_pan", 1, "sts3095", None),
        ("shoulder_lift", 2, "sts3095", None),
        ("elbow_flex", 3, "sts3095", None),
        ("wrist_flex", 4, "sts3215", None),
        ("wrist_yaw", 5, "sts3215", None),
        ("wrist_roll", 6, "sts3215", None),
        ("gripper", 7, "sts3215", MotorNormMode.RANGE_0_100),
    ),
    "am-follower-6dof-hd": (
        ("shoulder_pan", 1, "sts3250", None),
        ("shoulder_lift", 2, "sts3095", None),
        ("elbow_flex", 3, "sts3095", None),
        ("wrist_flex", 4, "sts3250", None),
        ("wrist_yaw", 5, "sts3250", None),
        ("wrist_roll", 6, "sts3250", None),
        ("gripper", 7, "sts3250", MotorNormMode.RANGE_0_100),
    ),
}


def _make_arm_motors(prefix: str, arm_profile: str, norm_mode_body: MotorNormMode) -> dict[str, Motor]:
    if arm_profile not in _ARM_PROFILES:
        raise ValueError(
            f"Unknown arm_profile '{arm_profile}'. Expected one of: {list(_ARM_PROFILES.keys())}."
        )

    return {
        f"{prefix}_{joint}": Motor(motor_id, model, norm_mode or norm_mode_body)
        for joint, motor_id, model, norm_mode in _ARM_PROFILES[arm_profile]
    }


class AlohaMini(Robot):
    """
    The robot includes a three omniwheel mobile base and a remote follower arm.
    The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
    forwarded to the remote follower arm (after applying a safety clamp).
    In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.
    """

    config_class = AlohaMiniConfig
    name = "alohamini"

    def __init__(self, config: AlohaMiniConfig):
        super().__init__(config)
        self.config = config
        self.logs: dict[str, Any] = {}
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        specs = validate_robot_model(config.robot_model)
        arm_profile = specs["arm_profile"]
        bm = specs["base_motor"]
        lm = specs["lift_motor"]
        self.wheel_radius = specs["wheel_radius"]
        self.base_radius = specs["base_radius"]

        left_arm_motors_cfg = _make_arm_motors("arm_left", arm_profile, norm_mode_body)
        right_arm_motors_cfg = _make_arm_motors("arm_right", arm_profile, norm_mode_body)
        self._left_arm_state_keys, self._right_arm_state_keys = arm_state_keys_for_robot_model(
            config.robot_model
        )

        left_bus_motors = {
            **(left_arm_motors_cfg if not config.no_follower else {}),
            # base
            "base_left_wheel": Motor(8, bm, MotorNormMode.RANGE_M100_100),
            "base_back_wheel": Motor(9, bm, MotorNormMode.RANGE_M100_100),
            "base_right_wheel": Motor(10, bm, MotorNormMode.RANGE_M100_100),
            "lift_axis": Motor(11, lm, MotorNormMode.DEGREES),
        }
        left_bus_calibration = {
            name: calibration for name, calibration in self.calibration.items() if name in left_bus_motors
        }
        self.left_bus = FeetechMotorsBus(
            port=self.config.left_port,
            motors=left_bus_motors,
            calibration=left_bus_calibration,
        )

        if not config.no_follower:
            right_bus_calibration = {
                name: calibration
                for name, calibration in self.calibration.items()
                if name in right_arm_motors_cfg
            }
            self.right_bus = FeetechMotorsBus(
                port=self.config.right_port,
                motors=right_arm_motors_cfg,
                calibration=right_bus_calibration,
            )
        else:
            self.right_bus = None

        if config.no_follower:
            self.left_arm_motors = []
            self.right_arm_motors = []
            self._left_arm_state_keys = ()
            self._right_arm_state_keys = ()
        else:
            self.left_arm_motors = [m for m in self.left_bus.motors if m.startswith("arm_left_")]
            self.right_arm_motors = [m for m in self.right_bus.motors if m.startswith("arm_right_")]

        self.base_motors = [m for m in self.left_bus.motors if m.startswith("base_")]

        # self.arm_motors = [motor for motor in self.left_bus.motors if motor.startswith("arm")]
        # self.base_motors = [motor for motor in self.left_bus.motors if motor.startswith("base")]

        self.cameras = make_cameras_from_configs(config.cameras)

        self.lift = LiftAxis(
            LiftAxisConfig(lead_mm_per_rev=specs["lead_mm_per_rev"], motor_model=lm),
            bus_left=self.left_bus,
            bus_right=self.right_bus,
        )
        self._initialize_current_protection()
        self._gripper_current_limit_ma = 500.0
        # Nudge the held position slightly further closed than present, so the gripper
        # keeps a bit of squeeze (a small resting current) instead of fully relaxing to 0mA.
        self._gripper_hold_close_step = 3
        # How far a reversed caller goal must move past the held position before the
        # contact hold is released.
        self._gripper_release_margin = 1.0
        # Fixed by mechanical installation, not measured at runtime: +1.0 means
        # increasing raw position opens the gripper. Contact release direction is
        # inferred from the command that caused current limiting, because either the
        # closing side or the mechanical open endpoint can produce overcurrent.
        self._gripper_open_direction: dict[str, float] = {
            "arm_left_gripper": 1.0,
            "arm_right_gripper": 1.0,
        }
        self._gripper_hold_goal: dict[str, float] = {}
        self._gripper_hold_direction: dict[str, float] = {}

        self._joint_release_margin = 1.0

        # Feedback sampled by get_observation() is reused by the following send_action().
        # This avoids reading the same position/current registers repeatedly in one Host
        # control cycle. Direct callers that send without observing still use the safe
        # read-through fallback in send_action().
        self._feedback_positions: dict[str, float] = {}
        self._feedback_currents_raw: dict[str, float] = {}
        self._feedback_lift_height_mm: float | None = None

    def _initialize_current_protection(self) -> None:
        all_motors = dict(self.left_bus.motors)
        if self.right_bus is not None:
            all_motors.update(self.right_bus.motors)
        self._current_limits = {name: _current_limits_for_motor(motor) for name, motor in all_motors.items()}
        self._sustained_overcurrent_started_at: dict[str, float] = {}
        self._near_stall_overcurrent_started_at: dict[str, float] = {}
        self._joint_stall_candidates: dict[str, _JointStallCandidate] = {}
        self._joint_hold_goal: dict[str, float] = {}
        self._joint_hold_direction: dict[str, float] = {}
        self._arm_goal_positions: dict[str, float] = {}
        self._arm_sent_positions: dict[str, float] = {}
        self._last_currents_log_t = 0.0

    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                *self._left_arm_state_keys,
                *self._right_arm_state_keys,
                "x.vel",
                "y.vel",
                "theta.vel",
                "lift_axis.height_mm",  # new
                # "lift_axis.vel",         # new (optional, for debugging)
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    # @property
    # def is_connected(self) -> bool:
    #     return self.left_bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_connected(self) -> bool:
        cams_ok = all(cam.is_connected for cam in self.cameras.values())
        return (
            self.left_bus.is_connected
            and (self.right_bus.is_connected if self.right_bus else True)
            and cams_ok
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.left_bus.connect()
        if self.right_bus:
            self.right_bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

        if self.is_calibrated:
            self.lift.home()
            print("Lift axis homed to 0mm.")
        else:
            logger.info("Skipping lift homing because AlohaMini is not calibrated.")

    @property
    def is_calibrated(self) -> bool:
        return self.left_bus.is_calibrated and (self.right_bus.is_calibrated if self.right_bus else True)

    def calibrate(self) -> None:
        """
        Dual-arm calibration (left arm + chassis on self.left_bus, right arm on self.right_bus):
        - Left arm: position mode → half-turn homing → collect ROM
        - Chassis: no homing; ROM fixed to 0–4095
        - Right arm (if present): position mode → half-turn homing → collect ROM
        - Merge into a single self.calibration, split by bus, write back to both buses, and save
        """
        # If a calibration file already exists: load it and write back, filtering for each bus separately
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                f"or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info("Writing existing calibration to both buses (trim per-bus caches)")

                calib_left = {k: v for k, v in self.calibration.items() if k in self.left_bus.motors}
                self.left_bus.write_calibration(calib_left, cache=False)
                self.left_bus.calibration = calib_left

                if getattr(self, "right_bus", None):
                    calib_right = {k: v for k, v in self.calibration.items() if k in self.right_bus.motors}
                    self.right_bus.write_calibration(calib_right, cache=False)
                    self.right_bus.calibration = calib_right

                return

        logger.info(f"\nRunning calibration of {self} (dual-bus if right_bus present)")

        if self.config.no_follower:
            logger.info("no_follower mode: writing default base/lift calibration.")
            self.calibration = {}
            for name, motor in self.left_bus.motors.items():
                self.calibration[name] = MotorCalibration(
                    id=motor.id,
                    drive_mode=0,
                    homing_offset=0,
                    range_min=0,
                    range_max=4095,
                )

            calib_left = {k: v for k, v in self.calibration.items() if k in self.left_bus.motors}
            self.left_bus.write_calibration(calib_left, cache=False)
            self.left_bus.calibration = calib_left
            self._save_calibration()
            print("Calibration saved to", self.calibration_fpath)
            return

        if not getattr(self, "left_arm_motors", None):
            raise RuntimeError("left_arm_motors is empty; expected names starting with 'left_arm_'")

        self.left_bus.disable_torque(self.left_arm_motors)
        for name in self.left_arm_motors:
            self.left_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move LEFT arm to the middle of its range of motion, then press ENTER...")
        left_homing = self.left_bus.set_half_turn_homings(self.left_arm_motors)  # left arm only

        for wheel in self.base_motors:
            left_homing[wheel] = 0

        motors_left_all = self.left_arm_motors + self.base_motors
        left_full_turn_motor = "arm_left_wrist_roll"
        full_turn_left = [m for m in motors_left_all if m.startswith("base_")]  # three base wheels
        if left_full_turn_motor in motors_left_all:
            full_turn_left.append(left_full_turn_motor)
        unknown_left = [m for m in motors_left_all if m not in full_turn_left]

        print(
            f"Move LEFT arm joints sequentially through full ROM (except '{left_full_turn_motor}'). "
            "Press ENTER to stop..."
        )
        l_mins, l_maxs = self.left_bus.record_ranges_of_motion(unknown_left)
        for m in full_turn_left:
            l_mins[m] = 0
            l_maxs[m] = 4095

        right_homing = {}
        r_mins, r_maxs = {}, {}

        if getattr(self, "right_bus", None) and getattr(self, "right_arm_motors", None):
            self.right_bus.disable_torque(self.right_arm_motors)
            for name in self.right_arm_motors:
                self.right_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

            input("Move RIGHT arm to the middle of its range of motion, then press ENTER...")
            right_homing = self.right_bus.set_half_turn_homings(self.right_arm_motors)

            right_full_turn_motor = "arm_right_wrist_roll"
            full_turn_right = (
                [right_full_turn_motor] if right_full_turn_motor in self.right_arm_motors else []
            )
            unknown_right = [m for m in self.right_arm_motors if m not in full_turn_right]

            print(
                f"Move RIGHT arm joints sequentially through full ROM (except '{right_full_turn_motor}'). "
                "Press ENTER to stop..."
            )
            r_mins, r_maxs = self.right_bus.record_ranges_of_motion(unknown_right)
            for m in full_turn_right:
                r_mins[m] = 0
                r_maxs[m] = 4095

        # Merge → filter by bus and write back → save as a single file
        self.calibration = {}

        for name, motor in self.left_bus.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=left_homing.get(name, 0),
                range_min=l_mins.get(name, 0),
                range_max=l_maxs.get(name, 4095),
            )

        if getattr(self, "right_bus", None):
            for name, motor in self.right_bus.motors.items():
                self.calibration[name] = MotorCalibration(
                    id=motor.id,
                    drive_mode=0,
                    homing_offset=right_homing.get(name, 0),
                    range_min=r_mins.get(name, 0),
                    range_max=r_maxs.get(name, 4095),
                )

        # Write back: each bus only writes its own entries to avoid KeyError
        calib_left = {k: v for k, v in self.calibration.items() if k in self.left_bus.motors}
        self.left_bus.write_calibration(calib_left, cache=False)
        self.left_bus.calibration = calib_left

        if getattr(self, "right_bus", None):
            calib_right = {k: v for k, v in self.calibration.items() if k in self.right_bus.motors}
            self.right_bus.write_calibration(calib_right, cache=False)
            self.right_bus.calibration = calib_right

        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self):
        # Set-up arm actuators (position mode)
        # We assume that at connection time, arm is in a rest position,
        # and torque can be safely disabled to run calibration.
        self.left_bus.disable_torque()
        self.left_bus.configure_motors()
        for name in self.left_arm_motors:
            self.left_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            self.left_bus.write(
                "Goal_Velocity",
                name,
                self.config.arm_goal_velocity,
                normalize=False,
            )
            self.left_bus.write(
                "Acceleration",
                name,
                self.config.arm_acceleration,
                normalize=False,
            )
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.left_bus.write("P_Coefficient", name, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.left_bus.write("I_Coefficient", name, 0)
            self.left_bus.write("D_Coefficient", name, 32)

        for name in self.base_motors:
            self.left_bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)

        # self.left_bus.enable_torque()

        if self.right_bus:
            self.right_bus.disable_torque()
            self.right_bus.configure_motors()
            for name in self.right_arm_motors:
                self.right_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
                self.right_bus.write(
                    "Goal_Velocity",
                    name,
                    self.config.arm_goal_velocity,
                    normalize=False,
                )
                self.right_bus.write(
                    "Acceleration",
                    name,
                    self.config.arm_acceleration,
                    normalize=False,
                )
                self.right_bus.write("P_Coefficient", name, 16)
                self.right_bus.write("I_Coefficient", name, 0)
                self.right_bus.write("D_Coefficient", name, 32)
            # self.right_bus.enable_torque()

        # self.lift.configure()

    def setup_motors(self) -> None:
        for motor in chain(reversed(self.arm_motors), reversed(self.base_motors)):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.left_bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.left_bus.motors[motor].id}")

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # Cap the value to fit within signed 16-bit range (-32768 to 32767)
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF  # 32767 -> maximum positive value
        elif speed_int < -0x8000:
            speed_int = -0x8000  # -32768 -> minimum negative value
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed
        degps = magnitude / steps_per_deg
        return degps

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float | None = None,
        base_radius: float | None = None,
        max_raw: int = 3000,
    ) -> dict:
        """
        Convert desired body-frame velocities into wheel raw commands.

        Parameters:
          x_cmd      : Linear velocity in x (m/s).
          y_cmd      : Linear velocity in y (m/s).
          theta_cmd  : Rotational velocity (deg/s).
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the center of rotation to each wheel (meters).
          max_raw    : Maximum allowed raw command (ticks) per wheel.

        Returns:
          A dictionary with wheel raw commands:
             {"base_left_wheel": value, "base_back_wheel": value, "base_right_wheel": value}.

        Notes:
          - Internally, the method converts theta_cmd to rad/s for the kinematics.
          - The raw command is computed from the wheels angular speed in deg/s
            using _degps_to_raw(). If any command exceeds max_raw, all commands
            are scaled down proportionally.
        """
        wheel_radius = self.wheel_radius if wheel_radius is None else wheel_radius
        base_radius = self.base_radius if base_radius is None else base_radius

        # Convert rotational velocity from deg/s to rad/s.
        theta_rad = theta * (np.pi / 180.0)
        # Create the body velocity vector [x, y, theta_rad].
        velocity_vector = np.array([-x, -y, theta_rad])

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 0, 120]) - 90)
        # Build the kinematic matrix: each row maps body velocities to a wheel’s linear speed.
        # The third column (base_radius) accounts for the effect of rotation.
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Compute each wheel’s linear speed (m/s) and then its angular speed (rad/s).
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # Convert wheel angular speeds from rad/s to deg/s.
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # Scaling
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # Convert each wheel’s angular speed (deg/s) to a raw integer.
        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]

        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
        }

    def _wheel_raw_to_body(
        self,
        left_wheel_speed,
        back_wheel_speed,
        right_wheel_speed,
        wheel_radius: float | None = None,
        base_radius: float | None = None,
    ) -> dict[str, Any]:
        """
        Convert wheel raw command feedback back into body-frame velocities.

        Parameters:
          wheel_raw   : Vector with raw wheel commands ("base_left_wheel", "base_back_wheel", "base_right_wheel").
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the robot center to each wheel (meters).

        Returns:
          A dict (x.vel, y.vel, theta.vel) all in m/s
        """
        wheel_radius = self.wheel_radius if wheel_radius is None else wheel_radius
        base_radius = self.base_radius if base_radius is None else base_radius

        # Convert each raw command back to an angular speed in deg/s.
        wheel_degps = np.array(
            [
                self._raw_to_degps(left_wheel_speed),
                self._raw_to_degps(back_wheel_speed),
                self._raw_to_degps(right_wheel_speed),
            ]
        )

        # Convert from deg/s to rad/s.
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # Compute each wheel’s linear speed (m/s) from its angular speed.
        wheel_linear_speeds = wheel_radps * wheel_radius

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Solve the inverse kinematics: body_velocity = M⁻¹ · wheel_linear_speeds.
        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector

        theta = theta_rad * (180.0 / np.pi)
        return {
            "x.vel": -x,
            "y.vel": -y,
            "theta.vel": theta,
        }  # m/s and deg/s

    @check_if_not_connected
    def get_observation(self, *, include_cameras: bool = True) -> RobotObservation:
        # Read actuators position for arm and vel for base
        observation_start_t = time.perf_counter()
        # arm_pos = self.left_bus.sync_read("Present_Position", self.arm_motors)

        # print(f"Left arm motors: {self.left_arm_motors}, Right arm motors: {self.right_arm_motors}")  # debug
        left_pos = (
            self.left_bus.sync_read("Present_Position", self.left_arm_motors) if self.left_arm_motors else {}
        )
        left_arm_done_t = time.perf_counter()

        base_wheel_vel = self.left_bus.sync_read("Present_Velocity", self.base_motors)

        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_left_wheel"],
            base_wheel_vel["base_back_wheel"],
            base_wheel_vel["base_right_wheel"],
        )
        base_done_t = time.perf_counter()

        right_pos = (
            self.right_bus.sync_read("Present_Position", self.right_arm_motors)
            if self.right_bus and self.right_arm_motors
            else {}
        )
        right_arm_done_t = time.perf_counter()

        left_arm_state = {f"{k}.pos": v for k, v in left_pos.items()}
        right_arm_state = {f"{k}.pos": v for k, v in right_pos.items()}

        obs_dict = {**left_arm_state, **right_arm_state, **base_vel}
        self.lift.contribute_observation(obs_dict)
        self._feedback_positions = {**left_pos, **right_pos}
        self._feedback_lift_height_mm = obs_dict.get("lift_axis.height_mm")
        lift_done_t = time.perf_counter()
        # print(f"Observation dict so far: {obs_dict}")  # debug

        dt_ms = (lift_done_t - observation_start_t) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # currents protection
        self._feedback_currents_raw = self.read_and_check_currents(print_currents=True, raw=True)
        currents_done_t = time.perf_counter()

        # Camera retrieval is outside the state-only control path. Camera drivers keep
        # their own async capture running; a full observation only fetches the latest
        # frames when a camera client has actually requested them.
        camera_timings_ms = {}
        camera_capture_monotonic_s = {}
        if include_cameras:
            for cam_key, cam in self.cameras.items():
                camera_start_t = time.perf_counter()
                # Capture runs in one background thread per camera. Host control only
                # peeks the bounded-age cache; the recorder, not this 50 Hz thread,
                # decides whether all camera timestamps have advanced.
                frame = cam.read_latest(max_age_ms=500)
                capture_timestamp = getattr(cam, "latest_timestamp", None)
                # Keep the pixel buffer and its capture timestamp from the same
                # camera-thread update. A frame may arrive between read_latest()
                # returning and metadata serialization.
                frame_lock = getattr(cam, "frame_lock", None)
                if frame_lock is not None:
                    with frame_lock:
                        latest_frame = getattr(cam, "latest_frame", None)
                        latest_timestamp = getattr(cam, "latest_timestamp", None)
                    if latest_frame is not None and latest_timestamp is not None:
                        frame = latest_frame
                        capture_timestamp = latest_timestamp
                obs_dict[cam_key] = frame
                if capture_timestamp is not None:
                    camera_capture_monotonic_s[cam_key] = float(capture_timestamp)
                camera_done_t = time.perf_counter()
                dt_ms = (camera_done_t - camera_start_t) * 1e3
                camera_timings_ms[f"camera_{cam_key}"] = dt_ms
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        clock_reference_monotonic_s = time.perf_counter()
        clock_reference_unix_ns = time.time_ns()
        state_sample_midpoint_monotonic_s = (observation_start_t + lift_done_t) / 2.0
        obs_dict["_host_timing"] = {
            "state_sample_started_monotonic_s": observation_start_t,
            "state_sample_finished_monotonic_s": lift_done_t,
            "state_sample_unix_ns": clock_reference_unix_ns
            - round((clock_reference_monotonic_s - state_sample_midpoint_monotonic_s) * 1e9),
            "host_clock_reference": {
                "monotonic_s": clock_reference_monotonic_s,
                "unix_ns": clock_reference_unix_ns,
            },
            "camera_capture_monotonic_s": camera_capture_monotonic_s,
        }

        observation_done_t = time.perf_counter()
        self.logs["observation_timing_ms"] = {
            "left_arm": (left_arm_done_t - observation_start_t) * 1e3,
            "base": (base_done_t - left_arm_done_t) * 1e3,
            "right_arm": (right_arm_done_t - base_done_t) * 1e3,
            "lift": (lift_done_t - right_arm_done_t) * 1e3,
            "currents": (currents_done_t - lift_done_t) * 1e3,
            **camera_timings_ms,
            "robot_observation_total": (observation_done_t - observation_start_t) * 1e3,
        }

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command AlohaMini to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            np.ndarray: the action sent to the motors, potentially clipped.
        """
        action_start_t = time.perf_counter()
        arm_action = {**self._arm_goal_positions, **action}
        # arm_goal_pos = {k: v for k, v in action.items() if k.endswith(".pos")}
        left_pos = {
            k: v
            for k, v in arm_action.items()
            if k.endswith(".pos")
            and k.startswith("arm_left_")
            and k.replace(".pos", "") in self.left_bus.motors
        }
        right_pos = {
            k: v
            for k, v in arm_action.items()
            if k.endswith(".pos")
            and k.startswith("arm_right_")
            and self.right_bus is not None
            and k.replace(".pos", "") in self.right_bus.motors
        }

        base_goal_vel = {key: float(action.get(key, 0.0)) for key in ("x.vel", "y.vel", "theta.vel")}

        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel["x.vel"], base_goal_vel["y.vel"], base_goal_vel["theta.vel"]
        )
        prepare_done_t = time.perf_counter()

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        # if self.config.max_relative_target is not None:
        #     present_pos = self.left_bus.sync_read("Present_Position", self.arm_motors)
        #     goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in arm_goal_pos.items()}
        #     arm_safe_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
        #     arm_goal_pos = arm_safe_goal_pos

        lift_sent = self.lift.apply_action(action, current_height_mm=self._feedback_lift_height_mm)
        lift_action_done_t = time.perf_counter()

        if left_pos and self.config.max_relative_target is not None:
            present_left = {
                motor: self._feedback_positions[motor]
                for motor in self.left_arm_motors
                if motor in self._feedback_positions
            }
            if not all(key.replace(".pos", "") in present_left for key in left_pos):
                present_left = self.left_bus.sync_read("Present_Position", self.left_arm_motors)
            gp_left = {k: (v, present_left[k.replace(".pos", "")]) for k, v in left_pos.items()}
            left_pos = ensure_safe_goal_position(gp_left, self.config.max_relative_target)

        if self.right_bus and right_pos and self.config.max_relative_target is not None:
            present_right = {
                motor: self._feedback_positions[motor]
                for motor in self.right_arm_motors
                if motor in self._feedback_positions
            }
            if not all(key.replace(".pos", "") in present_right for key in right_pos):
                present_right = self.right_bus.sync_read("Present_Position", self.right_arm_motors)
            gp_right = {k: (v, present_right[k.replace(".pos", "")]) for k, v in right_pos.items()}
            right_pos = ensure_safe_goal_position(gp_right, self.config.max_relative_target)
        relative_limit_done_t = time.perf_counter()

        self._arm_goal_positions.update(left_pos)
        self._arm_goal_positions.update(right_pos)
        left_pos = self._limit_gripper_goal_by_current(self.left_bus, left_pos)
        left_gripper_limit_done_t = time.perf_counter()
        left_pos = self._limit_joint_goal_by_current(self.left_bus, left_pos)
        left_joint_limit_done_t = time.perf_counter()
        if self.right_bus and right_pos:
            right_pos = self._limit_gripper_goal_by_current(self.right_bus, right_pos)
        right_gripper_limit_done_t = time.perf_counter()
        if self.right_bus and right_pos:
            right_pos = self._limit_joint_goal_by_current(self.right_bus, right_pos)
        right_joint_limit_done_t = time.perf_counter()

        # Send goal position to the actuators
        # arm_goal_pos_raw = {k.replace(".pos", ""): v for k, v in arm_goal_pos.items()}
        # self.left_bus.sync_write("Goal_Position", arm_goal_pos_raw)
        # self.left_bus.sync_write("Goal_Velocity", base_wheel_goal_vel)

        # return {**arm_goal_pos, **base_goal_vel}

        # print(f"[{filename}:{lineno}]Sending left_pos:{left_pos}, right_pos:{right_pos}, base_wheel_goal_vel:{base_wheel_goal_vel}")  # debug

        if left_pos:
            self.left_bus.sync_write("Goal_Position", {k.replace(".pos", ""): v for k, v in left_pos.items()})
            self._arm_sent_positions.update(left_pos)
        left_write_done_t = time.perf_counter()
        if self.right_bus and right_pos:
            self.right_bus.sync_write(
                "Goal_Position", {k.replace(".pos", ""): v for k, v in right_pos.items()}
            )
            self._arm_sent_positions.update(right_pos)
        right_write_done_t = time.perf_counter()
        self.left_bus.sync_write("Goal_Velocity", base_wheel_goal_vel)
        base_write_done_t = time.perf_counter()

        self.logs["action_timing_ms"] = {
            "action_prepare": (prepare_done_t - action_start_t) * 1e3,
            "action_lift": (lift_action_done_t - prepare_done_t) * 1e3,
            "action_relative_limit": (relative_limit_done_t - lift_action_done_t) * 1e3,
            "action_left_gripper_limit": (left_gripper_limit_done_t - relative_limit_done_t) * 1e3,
            "action_left_joint_limit": (left_joint_limit_done_t - left_gripper_limit_done_t) * 1e3,
            "action_right_gripper_limit": (right_gripper_limit_done_t - left_joint_limit_done_t) * 1e3,
            "action_right_joint_limit": (right_joint_limit_done_t - right_gripper_limit_done_t) * 1e3,
            "action_left_write": (left_write_done_t - right_joint_limit_done_t) * 1e3,
            "action_right_write": (right_write_done_t - left_write_done_t) * 1e3,
            "action_base_write": (base_write_done_t - right_write_done_t) * 1e3,
            "action_total": (base_write_done_t - action_start_t) * 1e3,
        }

        # A feedback snapshot belongs to exactly one observe -> act cycle.
        self._feedback_positions.clear()
        self._feedback_currents_raw.clear()
        self._feedback_lift_height_mm = None

        return {**left_pos, **right_pos, **base_goal_vel, **lift_sent}

    def supervise_arm_motion(self) -> RobotAction:
        """Check active arm targets using fresh feedback; write only safety corrections."""
        corrections = {}
        for bus in (self.left_bus, self.right_bus):
            if bus is None:
                continue
            goals = {
                key: value
                for key, value in self._arm_goal_positions.items()
                if key.removesuffix(".pos") in bus.motors
            }
            limited = self._limit_gripper_goal_by_current(bus, goals)
            limited = self._limit_joint_goal_by_current(bus, limited)
            changed = {
                key: value for key, value in limited.items() if value != self._arm_sent_positions.get(key)
            }
            if changed:
                bus.sync_write(
                    "Goal_Position", {key.removesuffix(".pos"): value for key, value in changed.items()}
                )
                self._arm_sent_positions.update(changed)
                corrections.update(changed)
        return corrections

    def _limit_gripper_goal_by_current(
        self, bus: FeetechMotorsBus, goal_pos: dict[str, float]
    ) -> dict[str, float]:
        """Stop pushing a gripper harder once its measured current exceeds the force limit."""
        target_keys = [
            key for key in goal_pos if key.endswith("_gripper.pos") and key.replace(".pos", "") in bus.motors
        ]
        if not target_keys:
            return goal_pos

        target_motors = [key.replace(".pos", "") for key in target_keys]
        feedback = self._read_force_feedback(bus, target_motors, "GripperCurrentLimit")
        if feedback is None:
            return {
                key: self._gripper_hold_goal.get(key.removesuffix(".pos"), value)
                for key, value in goal_pos.items()
            }
        currents_raw, present_pos = feedback

        limited_goal_pos = dict(goal_pos)
        for goal_key in target_keys:
            motor = goal_key.replace(".pos", "")
            goal = float(limited_goal_pos[goal_key])
            present = float(present_pos[motor])

            if motor not in self._gripper_hold_goal:
                current_ma = abs(float(currents_raw[motor]) * _CURRENT_MA_PER_RAW_UNIT)
                if current_ma < self._gripper_current_limit_ma:
                    continue

                command_delta = goal - present
                open_direction = self._gripper_open_direction.get(motor, 1.0)
                if command_delta > 0.0:
                    release_direction = -1.0
                elif command_delta < 0.0:
                    release_direction = 1.0
                else:
                    release_direction = open_direction
                if command_delta * open_direction < 0.0:
                    hold_goal = min(
                        100.0,
                        max(
                            0.0,
                            present - open_direction * self._gripper_hold_close_step,
                        ),
                    )
                else:
                    hold_goal = present
                self._gripper_hold_goal[motor] = hold_goal
                self._gripper_hold_direction[motor] = release_direction
                logger.warning(
                    "Gripper contact hold: %s, %.1f mA, position=%.2f",
                    motor,
                    current_ma,
                    hold_goal,
                )

            hold_goal = self._gripper_hold_goal[motor]
            release_direction = self._gripper_hold_direction[motor]
            if (goal - hold_goal) * release_direction >= self._gripper_release_margin:
                self._gripper_hold_goal.pop(motor, None)
                self._gripper_hold_direction.pop(motor, None)
                continue

            limited_goal_pos[goal_key] = hold_goal

        return limited_goal_pos

    def _limit_joint_goal_by_current(
        self, bus: FeetechMotorsBus, goal_pos: dict[str, float]
    ) -> dict[str, float]:
        """Hold joints that draw high current without moving toward their target."""
        target_keys = [
            key
            for key in goal_pos
            if not key.endswith("_gripper.pos") and key.replace(".pos", "") in bus.motors
        ]
        if not target_keys:
            return goal_pos

        target_motors = [key.replace(".pos", "") for key in target_keys]
        feedback = self._read_force_feedback(bus, target_motors, "JointStallLimit")
        if feedback is None:
            return {
                key: self._joint_hold_goal.get(key.removesuffix(".pos"), value)
                for key, value in goal_pos.items()
            }
        currents_raw, present_pos = feedback

        limited_goal_pos = dict(goal_pos)
        now = time.monotonic()
        for goal_key in target_keys:
            motor = goal_key.replace(".pos", "")
            goal = float(limited_goal_pos[goal_key])
            present = float(present_pos[motor])

            if motor not in self._joint_hold_goal:
                current_ma = abs(float(currents_raw[motor]) * _CURRENT_MA_PER_RAW_UNIT)
                command_error = _position_delta_degrees(bus, motor, goal - present)
                current_limit_ma = self._current_limits[motor].collision_ma
                if current_ma < current_limit_ma or abs(command_error) < _JOINT_STALL_MIN_COMMAND_ERROR_DEG:
                    self._joint_stall_candidates.pop(motor, None)
                    continue

                command_direction = 1.0 if command_error > 0.0 else -1.0
                candidate = self._joint_stall_candidates.get(motor)
                if candidate is None or candidate.command_direction != command_direction:
                    self._joint_stall_candidates[motor] = _JointStallCandidate(
                        now, present, command_direction
                    )
                    continue

                progress = (
                    _position_delta_degrees(bus, motor, present - candidate.start_position)
                    * command_direction
                )
                if progress >= _JOINT_STALL_MIN_PROGRESS_DEG:
                    self._joint_stall_candidates[motor] = _JointStallCandidate(
                        now, present, command_direction
                    )
                    continue
                if now - candidate.started_at < _JOINT_COLLISION_DURATION_S:
                    continue

                self._joint_hold_goal[motor] = present
                self._joint_hold_direction[motor] = -command_direction
                self._joint_stall_candidates.pop(motor, None)
                logger.warning(
                    "Joint stall hold: %s, %.1f mA, error=%.2f deg, progress=%.2f deg, position=%.2f",
                    motor,
                    current_ma,
                    command_error,
                    progress,
                    present,
                )

            hold_goal = self._joint_hold_goal[motor]
            release_direction = self._joint_hold_direction[motor]
            if (goal - hold_goal) * release_direction >= self._joint_release_margin:
                self._joint_hold_goal.pop(motor, None)
                self._joint_hold_direction.pop(motor, None)
                continue

            limited_goal_pos[goal_key] = hold_goal

        return limited_goal_pos

    def _read_force_feedback(
        self, bus: FeetechMotorsBus, motors: list[str], log_tag: str
    ) -> tuple[dict[str, float], dict[str, float]] | None:
        currents_raw = {
            motor: self._feedback_currents_raw[motor]
            for motor in motors
            if motor in self._feedback_currents_raw
        }
        present_pos = {
            motor: self._feedback_positions[motor] for motor in motors if motor in self._feedback_positions
        }
        if len(currents_raw) == len(motors) and len(present_pos) == len(motors):
            return currents_raw, present_pos

        try:
            return (
                bus.sync_read("Present_Current", motors),
                bus.sync_read("Present_Position", motors),
            )
        except Exception as e:
            for motor in motors:
                self._joint_stall_candidates.pop(motor, None)
            logger.warning("Failed to read %s current/position: %s", log_tag, e)
            return None

    def stop_base(self):
        self.left_bus.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=0)
        logger.info("Base motors stopped")

    def stop_lift(self):
        self.lift.stop()
        logger.info("Lift motor stopped")

    def stop_motion(self):
        self.stop_base()
        self.stop_lift()

    def read_and_check_currents(
        self,
        print_currents: bool = False,
        *,
        raw: bool = False,
    ) -> dict[str, float]:
        """Read motor currents and enforce model-aware, frequency-independent protection."""
        left_curr_raw = self.left_bus.sync_read("Present_Current", list(self.left_bus.motors.keys()))
        right_curr_raw = {}
        if getattr(self, "right_bus", None):
            right_curr_raw = self.right_bus.sync_read("Present_Current", list(self.right_bus.motors.keys()))

        now = time.monotonic()
        if print_currents and (now - self._last_currents_log_t >= 1.0):
            left_arr = [int(float(value) * _CURRENT_MA_PER_RAW_UNIT) for value in left_curr_raw.values()]
            logger.info("[Currents][left_bus] %s", left_arr)
            if right_curr_raw:
                right_arr = [
                    int(float(value) * _CURRENT_MA_PER_RAW_UNIT) for value in right_curr_raw.values()
                ]
                logger.info("[Currents][right_bus] %s", right_arr)
            self._last_currents_log_t = now

        tripped = None
        combined_raw = {**left_curr_raw, **right_curr_raw}
        for name, value in combined_raw.items():
            current_ma = abs(float(value) * _CURRENT_MA_PER_RAW_UNIT)
            limits = self._current_limits[name]

            if current_ma < limits.collision_ma:
                self._joint_stall_candidates.pop(name, None)

            for limit_ma, duration_s, timer, cause in (
                (
                    limits.near_stall_ma,
                    _NEAR_STALL_OVERCURRENT_DURATION_S,
                    self._near_stall_overcurrent_started_at,
                    "near-stall current",
                ),
                (
                    limits.sustained_ma,
                    _SUSTAINED_OVERCURRENT_DURATION_S,
                    self._sustained_overcurrent_started_at,
                    "sustained overload",
                ),
            ):
                if _has_sustained_overcurrent(timer, name, current_ma, limit_ma, now, duration_s):
                    tripped = (name, current_ma, limit_ma, duration_s, cause)
                    break
            if tripped is not None:
                break

        if tripped is not None:
            name, current_ma, current_limit_ma, duration_s, cause = tripped
            logger.error(
                "Overcurrent: %s, %s, %.1f mA >= %.1f mA for %.0f ms; disconnecting",
                name,
                cause,
                current_ma,
                current_limit_ma,
                duration_s * 1000,
            )
            with suppress(Exception):
                self.stop_motion()
            try:
                self.disconnect()
            except Exception as e:
                logger.error("Overcurrent disconnect failed: %s", e)
            sys.exit(1)

        if raw:
            return combined_raw
        return {k: round(v * _CURRENT_MA_PER_RAW_UNIT, 1) for k, v in combined_raw.items()}

    @check_if_not_connected
    def disconnect(self):
        self.stop_motion()
        self.left_bus.disconnect(self.config.disable_torque_on_disconnect)
        if self.right_bus:
            self.right_bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
