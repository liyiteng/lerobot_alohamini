"""
Base Agent for AlohaMini Robot variants.

This module provides the common base functionality shared between
AlohaMiniVirtual and AlohaMiniSO100V2 agents, following DRY principle.
"""

import os
from abc import abstractmethod
from pathlib import Path

import sapien.physx as physx
import torch
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link
from scipy.spatial.transform import Rotation as R

# Collision bits for AlohaMini robots
ALOHA_MINI_BASE_COLLISION_BIT = 29
ALOHA_MINI_WHEELS_COLLISION_BIT = 30

# Robot-description search path (vendored copy). Upstream loaded the URDFs from
# ~/.maniskill/data/robots/aloha_mini (populated by AlohaMini/maniskill/install.py);
# the vendored package ships them in assets/ next to this file so a fresh clone of
# this repo needs no out-of-repo install step.
_VENDORED_ASSET_DIR = Path(__file__).resolve().parent / "assets"
_LEGACY_ASSET_DIR = Path.home() / ".maniskill" / "data" / "robots" / "aloha_mini"


def resolve_urdf(filename: str) -> str:
    """Resolve a robot URDF path independent of the current working directory.

    Search order:
      1. ``$ALOHAMINI_URDF_DIR`` (env-var override, e.g. a locally patched robot),
      2. the ``assets/`` directory vendored next to this package,
      3. the legacy ``~/.maniskill/data/robots/aloha_mini`` install location.
    """
    candidates = []
    override = os.environ.get("ALOHAMINI_URDF_DIR")
    if override:
        candidates.append(Path(override) / filename)
    candidates += [_VENDORED_ASSET_DIR / filename, _LEGACY_ASSET_DIR / filename]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    # Nothing found: return the vendored path so the loader error names the
    # location that is supposed to contain the file.
    return str(_VENDORED_ASSET_DIR / filename)


def euler_to_quat_xyz(rx, ry, rz):
    """
    Convert XYZ euler angles to quaternion [w, x, y, z].

    Args:
        rx: Rotation around X axis in radians
        ry: Rotation around Y axis in radians
        rz: Rotation around Z axis in radians

    Returns:
        Quaternion as [w, x, y, z]
    """
    r = R.from_euler("xyz", [rx, ry, rz])
    q = r.as_quat()
    return [q[3], q[0], q[1], q[2]]


class AlohaMiniBaseAgent(BaseAgent):
    """
    Base agent for AlohaMini robot variants.

    Provides common functionality:
    - Virtual mobile base (prismatic X/Y + rotation joints)
    - Vertical lift mechanism
    - Dual arm structure (left/right)
    - Contact-based grasping detection
    - Static state detection

    Subclasses must define:
    - uid: Robot identifier
    - urdf_path: Path to URDF file
    - urdf_config: Material configurations
    - keyframes: Robot keyframe poses
    - _sensor_configs: Camera configurations
    - left_arm_joint_names, right_arm_joint_names: Arm joint names
    - _controller_configs: Controller configurations
    - _after_init(): Link initialization (call super()._after_init_base())
    - get_left_ee_pose(): Left end-effector pose
    - get_right_ee_pose(): Right end-effector pose
    - _check_single_arm_grasping(): Arm-specific grasp check
    """

    # Base joint names (virtual mobile base) - common to all variants
    BASE_JOINT_NAMES = [
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_rotation_joint",
    ]

    # Lift joint name - common to all variants
    LIFT_JOINT_NAMES = ["vertical_move"]

    def __init__(self, *args, **kwargs):
        # Set joint names as instance attributes
        self.base_joint_names = self.BASE_JOINT_NAMES.copy()
        self.lift_joint_names = self.LIFT_JOINT_NAMES.copy()

        # Default controller parameters for lift
        self.lift_stiffness = 2e3
        self.lift_damping = 2e2
        self.lift_force_limit = 150

        # Default controller parameters for base
        self.base_damping = 1000
        self.base_force_limit = 500
        # Position controller gains for the fixed-base mode (subclasses with a heavier
        # arm, e.g. the 6-DOF Pro, raise these so the virtual base does not drift/rotate
        # under the arm's reaction forces).
        self.base_pos_stiffness = 1e4
        self.base_pos_damping = 1e3
        self.base_pos_force_limit = 2000

        super().__init__(*args, **kwargs)

    def _create_base_controller(self):
        """Create PDBaseVelController config for the virtual mobile base."""
        return PDBaseVelControllerConfig(
            self.base_joint_names,
            lower=[-1, -1, -3.14],
            upper=[1, 1, 3.14],
            damping=self.base_damping,
            force_limit=self.base_force_limit,
        )

    def _create_base_pos_controller(self):
        """Position controller for the virtual base (holds it rigid when commanded).

        Useful for stationary tabletop manipulation / scripted data generation,
        where the velocity-controlled mobile base would otherwise drift under the
        arm's reaction forces.
        """
        return PDJointPosControllerConfig(
            self.base_joint_names,
            lower=None,
            upper=None,
            stiffness=self.base_pos_stiffness,
            damping=self.base_pos_damping,
            force_limit=self.base_pos_force_limit,
            normalize_action=False,
        )

    # Physical travel of the vertical_move (lift) joint, from the upstream
    # liyiteng/alohamini alohamini2pro.urdf: [-0.3, +0.3] (0.6 m total travel,
    # 0.3 m below the home carriage position — enables near-floor reach).
    LIFT_LOWER = -0.3
    LIFT_UPPER = 0.3

    def _create_lift_pos_controller(self):
        """Create position controller for the lift mechanism.

        The action is clamped to the joint's physical travel so a commanded lift
        target can never drive the column past its limit.
        """
        return PDJointPosControllerConfig(
            self.lift_joint_names,
            lower=self.LIFT_LOWER,
            upper=self.LIFT_UPPER,
            stiffness=self.lift_stiffness,
            damping=self.lift_damping,
            force_limit=self.lift_force_limit,
            normalize_action=False,
        )

    def _create_lift_delta_pos_controller(self, delta_limit=0.05):
        """Create delta position controller for the lift mechanism."""
        return PDJointPosControllerConfig(
            self.lift_joint_names,
            lower=-delta_limit,
            upper=delta_limit,
            stiffness=self.lift_stiffness,
            damping=self.lift_damping,
            force_limit=self.lift_force_limit,
            use_delta=True,
        )

    def _after_init_base(self):
        """
        Initialize common base links and collision settings.

        Call this from subclass _after_init() method.
        """
        self.base_link: Link = sapien_utils.get_obj_by_name(self.robot.get_links(), "base_link")
        self.vertical_link: Link = sapien_utils.get_obj_by_name(self.robot.get_links(), "vertical_link")

        # Set collision group for base link
        self.base_link.set_collision_group_bit(group=2, bit_idx=ALOHA_MINI_BASE_COLLISION_BIT, bit=1)

        # Initialize contact query dictionary
        self.queries: dict[str, tuple[physx.PhysxGpuContactPairImpulseQuery, tuple[int]]] = dict()

    @abstractmethod
    def get_left_ee_pose(self):
        """Get left end-effector pose. Must be implemented by subclass."""
        pass

    @abstractmethod
    def get_right_ee_pose(self):
        """Get right end-effector pose. Must be implemented by subclass."""
        pass

    def get_ee_poses(self):
        """Get end-effector poses for both arms."""
        return self.get_left_ee_pose(), self.get_right_ee_pose()

    @abstractmethod
    def _check_single_arm_grasping(self, object: Actor, min_force=0.5, max_angle=110, arm_id=1):
        """
        Check if a single arm is grasping an object.
        Must be implemented by subclass due to different gripper structures.
        """
        pass

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110, arm_id=None):
        """
        Check if the robot is grasping an object.

        Args:
            object: The actor to check grasping against
            min_force: Minimum contact force threshold
            max_angle: Maximum angle between force and gripper direction
            arm_id: 1 for left, 2 for right, None for either

        Returns:
            Boolean tensor indicating grasping state per environment.
        """
        if arm_id is None:
            arm1_grasping = self._check_single_arm_grasping(object, min_force, max_angle, arm_id=1)
            arm2_grasping = self._check_single_arm_grasping(object, min_force, max_angle, arm_id=2)
            return torch.logical_or(arm1_grasping, arm2_grasping)
        else:
            return self._check_single_arm_grasping(object, min_force, max_angle, arm_id)

    def is_static(self, threshold=0.2):
        """
        Check if the robot is static (not moving).

        Args:
            threshold: Maximum velocity threshold for static state.

        Returns:
            Boolean tensor indicating static state per environment.
        """
        qvel = self.robot.get_qvel()[:, 3:]  # exclude the base joints
        return torch.max(torch.abs(qvel), 1)[0] <= threshold
