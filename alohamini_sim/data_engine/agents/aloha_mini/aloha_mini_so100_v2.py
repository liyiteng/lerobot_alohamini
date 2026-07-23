"""
AlohaMini Robot Agent with ManiSkill SO100 Arms V2 (Parallel Gripper)

This variant uses the maniskill_so100_version.urdf which is based on the
official ManiSkill SO100 arm structure (no rotation in base_joint, rotation in
shoulder_pan).

Gripper: the original SO-100 single rotating jaw has been replaced with a
2-finger **parallel** gripper that reproduces the roboninecom SO-ARM100/101
Parallel Gripper kinematics (1-DOF, two jaws translating symmetrically). Each
gripper has two prismatic finger joints (`*_finger_joint1`, `*_finger_joint2`)
driven as a SINGLE action via a PDJointPosMimicController (finger2 mimics
finger1). From the policy/action point of view this is still 1 gripper DOF per
arm, exactly like the old design — only the internal joint structure changed.

Finger joint convention: qpos 0 = closed (jaws meet at center),
qpos 0.037 = fully open (~74 mm aperture).
"""

import math
from copy import deepcopy

import numpy as np
import sapien
import torch
from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link

from .base_agent import AlohaMiniBaseAgent, euler_to_quat_xyz, resolve_urdf

# Collision bits so the two fingers of each gripper do not collide with each
# other when closed (they are siblings in the URDF tree, so SAPIEN does not
# auto-disable their pairwise collision). Separate bits per arm so that the
# left and right fingers can still collide with one another. Bits 29/30 are
# used by the base/wheels in base_agent.py.
ALOHA_MINI_LEFT_GRIPPER_COLLISION_BIT = 27
ALOHA_MINI_RIGHT_GRIPPER_COLLISION_BIT = 28


@register_agent()
class AlohaMiniSO100V2(AlohaMiniBaseAgent):
    """
    AlohaMini with official ManiSkill SO100 arms + parallel grippers (V2).

    This robot uses virtual base joints and SO100 arm structure:
    - root_x_axis_joint: prismatic joint for X movement
    - root_y_axis_joint: prismatic joint for Y movement
    - root_z_rotation_joint: continuous joint for rotation
    - vertical_move: prismatic joint for lift
    - Left/Right arm: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
      wrist_roll (5 revolute) + finger_joint1/finger_joint2 (2 prismatic)

    Active joint (qpos) order, 18 DOF. NOTE: SAPIEN interleaves the two arms'
    joints (verified by tools/smoke_test_gripper.py), so the order is:
      0-2   base:  root_x, root_y, root_z
      3     lift:  vertical_move
      4-13  arms (interleaved): L_pan, R_pan, L_lift, R_lift, L_elbow, R_elbow,
            L_wrist_flex, R_wrist_flex, L_wrist_roll, R_wrist_roll
      14-15 left fingers:  left_finger_joint1, left_finger_joint2
      16-17 right fingers: right_finger_joint1, right_finger_joint2
    Controllers map joints by name, so the action layout is unaffected by this
    interleaving; only hardcoded keyframe qpos arrays must follow this order.
    """

    uid = "aloha_mini_so100_v2"
    urdf_path = resolve_urdf("maniskill_so100_version.urdf")

    urdf_config = dict(
        _materials=dict(
            gripper=dict(
                static_friction=2.0,
                dynamic_friction=2.0,
                restitution=0.0,
            ),
        ),
        link=dict(
            left_finger1=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            left_finger2=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            right_finger1=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            right_finger2=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    # Gripper finger position limits (meters). 0 = closed, GRIPPER_OPEN = open.
    GRIPPER_OPEN = 0.037  # roboninecom clamp stroke (so_101.urdf.xacro right_clamp upper)
    GRIPPER_CLOSED = 0.0

    # Keyframe qpos arrays follow SAPIEN's interleaved active-joint order (see the
    # class docstring): base(3), lift(1), then arms interleaved L,R,L,R,..., then
    # left fingers (2), then right fingers (2).
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    0.0,
                    0.0,  # base: x, y, rot
                    0.0,  # lift
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,  # arms interleaved (all zero)
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,  # arms interleaved (all zero)
                    GRIPPER_CLOSED,
                    GRIPPER_CLOSED,  # left fingers (closed)
                    GRIPPER_CLOSED,
                    GRIPPER_CLOSED,  # right fingers (closed)
                ]
            ),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
        ready=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    0.0,
                    0.0,  # base
                    0.05,  # lift
                    # arms interleaved: L_pan,R_pan, L_lift,R_lift, L_elbow,R_elbow,
                    #                   L_wflex,R_wflex, L_wroll,R_wroll
                    0.0,
                    0.0,
                    0.3,
                    0.3,
                    -0.3,
                    -0.3,
                    0.0,
                    0.0,
                    0.3,
                    0.3,
                    GRIPPER_OPEN,
                    GRIPPER_OPEN,  # left fingers (open)
                    GRIPPER_OPEN,
                    GRIPPER_OPEN,  # right fingers (open)
                ]
            ),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
        zero=Keyframe(
            qpos=np.array([0.0] * 18),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
    )

    @property
    def _sensor_configs(self):
        # Main camera: overhead view
        q_main = euler_to_quat_xyz(math.radians(90), math.radians(90), math.radians(0))

        # Wrist cameras (on Fixed_Jaw / gripper palm, looking down at the fingers)
        q_wrist = euler_to_quat_xyz(math.radians(0), math.radians(45), math.radians(0))

        return [
            CameraConfig(
                uid="cam_main",
                pose=Pose.create_from_pq(
                    p=[0.0, 0.0, 1.45],
                    q=q_main,
                ),
                width=320,
                height=240,
                fov=1.5,
                near=0.01,
                far=100,
                entity_uid="base_link",
            ),
            CameraConfig(
                uid="cam_left_wrist",
                pose=Pose.create_from_pq(
                    p=[0.0, 0.0, 0.06],
                    q=q_wrist,
                ),
                width=128,
                height=128,
                fov=1.5,
                near=0.01,
                far=100,
                entity_uid="left_Fixed_Jaw",
            ),
            CameraConfig(
                uid="cam_right_wrist",
                pose=Pose.create_from_pq(
                    p=[0.0, 0.0, 0.06],
                    q=q_wrist,
                ),
                width=128,
                height=128,
                fov=1.5,
                near=0.01,
                far=100,
                entity_uid="right_Fixed_Jaw",
            ),
        ]

    def __init__(self, *args, **kwargs):
        # SO100 arm joints (5 revolute joints, gripper handled separately)
        self.left_arm_joint_names = [
            "left_shoulder_pan",
            "left_shoulder_lift",
            "left_elbow_flex",
            "left_wrist_flex",
            "left_wrist_roll",
        ]
        self.right_arm_joint_names = [
            "right_shoulder_pan",
            "right_shoulder_lift",
            "right_elbow_flex",
            "right_wrist_flex",
            "right_wrist_roll",
        ]
        self.arm_joint_names = self.left_arm_joint_names + self.right_arm_joint_names

        # Parallel gripper joints: 2 prismatic finger joints per arm.
        self.left_gripper_joint_names = ["left_finger_joint1", "left_finger_joint2"]
        self.right_gripper_joint_names = ["right_finger_joint1", "right_finger_joint2"]

        # Controller parameters for arm joints (5 revolute joints)
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100

        # Controller parameters for gripper (prismatic, position controlled)
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 15.0

        super().__init__(*args, **kwargs)

    def _create_gripper_controller(self, gripper_joint_names):
        """Single-action parallel gripper controller (finger2 mimics finger1)."""
        finger1, finger2 = gripper_joint_names
        return PDJointPosMimicControllerConfig(
            gripper_joint_names,
            lower=self.GRIPPER_CLOSED,
            upper=self.GRIPPER_OPEN,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=False,
            mimic={finger2: {"joint": finger1}},
        )

    @property
    def _controller_configs(self):
        # Base controllers (from base class): velocity (mobile) and position (fixed)
        base_pd_joint_vel = self._create_base_controller()
        base_pd_joint_pos = self._create_base_pos_controller()

        # Lift controllers (from base class)
        lift_pos = self._create_lift_pos_controller()
        lift_delta_pos = self._create_lift_delta_pos_controller()

        # Per-joint parameters for the arm joints (DOF-agnostic: 5 for Std, 6 for Pro)
        n_arm = len(self.left_arm_joint_names)
        arm_stiffness_list = [self.arm_stiffness] * n_arm
        arm_damping_list = [self.arm_damping] * n_arm
        arm_force_limit_list = [self.arm_force_limit] * n_arm

        # Left arm controllers (5 revolute joints)
        left_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.left_arm_joint_names,
            lower=None,
            upper=None,
            stiffness=arm_stiffness_list,
            damping=arm_damping_list,
            force_limit=arm_force_limit_list,
            normalize_action=False,
        )
        left_arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.left_arm_joint_names,
            lower=[-0.05] * n_arm,
            upper=[0.05] * n_arm,
            stiffness=arm_stiffness_list,
            damping=arm_damping_list,
            force_limit=arm_force_limit_list,
            use_delta=True,
        )

        # Right arm controllers (5 revolute joints)
        right_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.right_arm_joint_names,
            lower=None,
            upper=None,
            stiffness=arm_stiffness_list,
            damping=arm_damping_list,
            force_limit=arm_force_limit_list,
            normalize_action=False,
        )
        right_arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.right_arm_joint_names,
            lower=[-0.05] * n_arm,
            upper=[0.05] * n_arm,
            stiffness=arm_stiffness_list,
            damping=arm_damping_list,
            force_limit=arm_force_limit_list,
            use_delta=True,
        )

        # Parallel gripper controllers (1 action each, finger2 mimics finger1)
        left_gripper_pd = self._create_gripper_controller(self.left_gripper_joint_names)
        right_gripper_pd = self._create_gripper_controller(self.right_gripper_joint_names)

        controller_configs = dict(
            pd_joint_pos=dict(
                base=base_pd_joint_vel,
                lift=lift_pos,
                left_arm=left_arm_pd_joint_pos,
                left_gripper=left_gripper_pd,
                right_arm=right_arm_pd_joint_pos,
                right_gripper=right_gripper_pd,
            ),
            pd_joint_delta_pos=dict(
                base=base_pd_joint_vel,
                lift=lift_delta_pos,
                left_arm=left_arm_pd_joint_delta_pos,
                left_gripper=left_gripper_pd,
                right_arm=right_arm_pd_joint_delta_pos,
                right_gripper=right_gripper_pd,
            ),
            # Stationary-base variant for scripted manipulation / data generation:
            # the base is position-controlled (commanding [0,0,0] holds it rigid).
            pd_joint_pos_fixed_base=dict(
                base=base_pd_joint_pos,
                lift=lift_pos,
                left_arm=left_arm_pd_joint_pos,
                left_gripper=left_gripper_pd,
                right_arm=right_arm_pd_joint_pos,
                right_gripper=right_gripper_pd,
            ),
        )

        return deepcopy(controller_configs)

    def _after_init(self):
        # Initialize base links and collision settings
        self._after_init_base()

        # Gripper palm links (hold the wrist cameras)
        self.left_palm_link: Link = sapien_utils.get_obj_by_name(self.robot.get_links(), "left_Fixed_Jaw")
        self.right_palm_link: Link = sapien_utils.get_obj_by_name(self.robot.get_links(), "right_Fixed_Jaw")

        # Left arm finger links + tips
        self.left_finger1_link: Link = sapien_utils.get_obj_by_name(self.robot.get_links(), "left_finger1")
        self.left_finger2_link: Link = sapien_utils.get_obj_by_name(self.robot.get_links(), "left_finger2")
        self.left_finger1_tip: Link = sapien_utils.get_obj_by_name(self.robot.get_links(), "left_finger1_tip")
        self.left_finger2_tip: Link = sapien_utils.get_obj_by_name(self.robot.get_links(), "left_finger2_tip")

        # Right arm finger links + tips
        self.right_finger1_link: Link = sapien_utils.get_obj_by_name(self.robot.get_links(), "right_finger1")
        self.right_finger2_link: Link = sapien_utils.get_obj_by_name(self.robot.get_links(), "right_finger2")
        self.right_finger1_tip: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_finger1_tip"
        )
        self.right_finger2_tip: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_finger2_tip"
        )

        # Disable mutual collision between the two fingers of each gripper
        # (per-arm bit, so left vs right fingers still collide normally).
        for finger in (self.left_finger1_link, self.left_finger2_link):
            finger.set_collision_group_bit(group=2, bit_idx=ALOHA_MINI_LEFT_GRIPPER_COLLISION_BIT, bit=1)
        for finger in (self.right_finger1_link, self.right_finger2_link):
            finger.set_collision_group_bit(group=2, bit_idx=ALOHA_MINI_RIGHT_GRIPPER_COLLISION_BIT, bit=1)

    @property
    def tcp_pos(self):
        """Left arm TCP position (midpoint between finger tips)."""
        return (self.left_finger1_tip.pose.p + self.left_finger2_tip.pose.p) / 2

    @property
    def tcp_pose(self):
        """Left arm TCP pose (orientation from the gripper palm)."""
        return Pose.create_from_pq(self.tcp_pos, self.left_palm_link.pose.q)

    @property
    def tcp_pos_2(self):
        """Right arm TCP position."""
        return (self.right_finger1_tip.pose.p + self.right_finger2_tip.pose.p) / 2

    @property
    def tcp_pose_2(self):
        """Right arm TCP pose."""
        return Pose.create_from_pq(self.tcp_pos_2, self.right_palm_link.pose.q)

    def get_left_ee_pose(self):
        return self.tcp_pose

    def get_right_ee_pose(self):
        return self.tcp_pose_2

    def _check_single_arm_grasping(self, object: Actor, min_force=0.5, max_angle=110, arm_id=1):
        """
        Check if a single arm is grasping (parallel gripper, two fingers).

        Uses the frame-independent finger-separation axis so the check is robust
        to the exact finger mesh orientation: when an object is grasped, it pushes
        finger1 outward (along -sep) and finger2 outward (along +sep), where
        sep is the unit vector from finger1 toward finger2.
        """
        if arm_id == 1:
            finger1_link, finger2_link = self.left_finger1_link, self.left_finger2_link
            finger1_tip, finger2_tip = self.left_finger1_tip, self.left_finger2_tip
        elif arm_id == 2:
            finger1_link, finger2_link = self.right_finger1_link, self.right_finger2_link
            finger1_tip, finger2_tip = self.right_finger1_tip, self.right_finger2_tip
        else:
            raise ValueError(f"Invalid arm_id: {arm_id}. Must be 1 or 2.")

        l_contact_forces = self.scene.get_pairwise_contact_forces(finger1_link, object)
        r_contact_forces = self.scene.get_pairwise_contact_forces(finger2_link, object)
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # Unit jaw-separation axis from the finger TIPS, not the finger link
        # origins: with the real clamp meshes the link origins are joint frames
        # offset from the jaws, so link-based separation can point the wrong way.
        sep = finger2_tip.pose.p - finger1_tip.pose.p
        sep = sep / (torch.linalg.norm(sep, axis=1, keepdim=True) + 1e-9)

        # Reaction force on each finger points outward (away from the object center)
        langle = common.compute_angle_between(-sep, l_contact_forces)
        rangle = common.compute_angle_between(sep, r_contact_forces)
        lflag = torch.logical_and(lforce >= min_force, torch.rad2deg(langle) <= max_angle)
        rflag = torch.logical_and(rforce >= min_force, torch.rad2deg(rangle) <= max_angle)
        return torch.logical_and(lflag, rflag)
