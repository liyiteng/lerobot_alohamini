"""AlohaMini Pro **v3**: true 6-joint arms + parallel gripper mounted distally.

v2 deleted the donor's link5/link6 (old jaw) and drove the gripper palm from
joint5, leaving 5 positioning joints per arm. v3 (tools/build_pro_urdf_v3.py)
re-inserts the donor chain as bare frame links, so each arm gains a 2-DOF
wrist: joint5 = wrist ROLL (axis Y along the forearm, +-180deg) and joint6 =
wrist PITCH (the donor's old jaw hinge axis X, repurposed, +-180deg).
Full-pose (6-DOF) grasp targets become satisfiable — needed by CuRobo in the
InternDataEngine port.

Active-joint (qpos) order, 20 DOF — SAPIEN interleaves the two arms:
  0-2   base:  root_x, root_y, root_z
  3     lift:  vertical_move
  4-15  arms (interleaved): L1,R1 ... L6,R6
  16-17 left fingers, 18-19 right fingers
"""

import numpy as np
import sapien
from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.registration import register_agent

from .aloha_mini_pro_v2 import AlohaMiniProV2
from .base_agent import AlohaMiniBaseAgent, resolve_urdf


@register_agent()
class AlohaMiniProV3(AlohaMiniProV2):
    """AlohaMini Pro v3: 6-joint arms (roll+pitch wrist) + parallel grippers."""

    uid = "aloha_mini_pro_v3"
    urdf_path = resolve_urdf("aloha_mini_pro_v3.urdf")

    keyframes = dict(
        rest=Keyframe(
            qpos=np.zeros(20, dtype=np.float64),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
        ready=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.05,  # lift
                    # interleaved: L1,R1, L2,R2, L3,R3, L4,R4, L5,R5, L6,R6
                    0.0,
                    0.0,
                    0.5,
                    0.5,
                    -0.6,
                    -0.6,
                    0.3,
                    0.3,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.037,
                    0.037,  # left fingers (open)
                    0.037,
                    0.037,  # right fingers (open)
                ]
            ),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
    )

    def __init__(self, *args, **kwargs):
        # 6 arm joints per side; gains identical to v2 (same masses/motors).
        self.left_arm_joint_names = [
            "left_joint1",
            "left_joint2",
            "left_joint3",
            "left_joint4",
            "left_joint5",
            "left_joint6",
        ]
        self.right_arm_joint_names = [
            "right_joint1",
            "right_joint2",
            "right_joint3",
            "right_joint4",
            "right_joint5",
            "right_joint6",
        ]
        self.arm_joint_names = self.left_arm_joint_names + self.right_arm_joint_names
        self.left_gripper_joint_names = ["left_finger_joint1", "left_finger_joint2"]
        self.right_gripper_joint_names = ["right_finger_joint1", "right_finger_joint2"]

        self.arm_stiffness = 2e3
        self.arm_damping = 4e2
        self.arm_force_limit = 300
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 15.0
        self.base_pos_stiffness = 2e4
        self.base_pos_damping = 2e3
        self.base_pos_force_limit = 4000
        self.lift_stiffness = 4e3
        self.lift_damping = 6e2
        self.lift_force_limit = 400

        # skip both 5-joint __init__s (SO100V2 and ProV2 hardcode 5 names)
        AlohaMiniBaseAgent.__init__(self, *args, **kwargs)
