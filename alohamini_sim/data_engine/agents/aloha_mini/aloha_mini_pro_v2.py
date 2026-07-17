"""
AlohaMini **Pro** agent: the longer Pro arm with the roboninecom parallel gripper
REPLACING the original jaw gripper.

The source "6-DOF" Pro arm is really 5 arm joints + a jaw: link5 was the original
gripper body (serrated static jaw) and joint6/link6 the moving hook jaw (verified
by rendering the meshes). tools/build_pro_urdf.py deletes that original gripper
and lets joint5 (the wrist ROLL, axis Y along the forearm — roboninecom's
`link4_to_link5`) drive the parallel-gripper palm `*_Fixed_Jaw` directly, exactly
mirroring roboninecom's SO-101 kit and our Std SO-100 conversion.

Active-joint (qpos) order, 18 DOF — SAPIEN interleaves the two arms:
  0-2   base:  root_x, root_y, root_z
  3     lift:  vertical_move
  4-13  arms (interleaved): L1,R1, L2,R2, L3,R3, L4,R4, L5,R5
  14-15 left fingers:  left_finger_joint1, left_finger_joint2
  16-17 right fingers: right_finger_joint1, right_finger_joint2
"""

import numpy as np
import sapien
from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.registration import register_agent

from .aloha_mini_so100_v2 import AlohaMiniSO100V2
from .base_agent import AlohaMiniBaseAgent, resolve_urdf


@register_agent()
class AlohaMiniProV2(AlohaMiniSO100V2):
    """AlohaMini Pro: 6-DOF arms + parallel grippers (inherits Std gripper stack)."""

    uid = "aloha_mini_pro_v2"
    urdf_path = resolve_urdf("aloha_mini_pro_v2.urdf")

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    0.0,
                    0.0,  # base: root_x, root_y, root_z
                    0.0,  # lift
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,  # arms interleaved L1,R1,L2,R2,L3,R3
                    0.0,
                    0.0,
                    0.0,
                    0.0,  # arms interleaved L4,R4,L5,R5
                    0.0,
                    0.0,  # left fingers (closed)
                    0.0,
                    0.0,  # right fingers (closed)
                ]
            ),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
        ready=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.05,  # lift
                    # interleaved: L1,R1, L2,R2, L3,R3, L4,R4, L5,R5
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
        # 5 arm joints per side (joint5 = wrist roll driving the palm). Everything else
        # (gripper joints, controller gains, mimic gripper, grasp check, cameras) is
        # inherited from the Std agent.
        self.left_arm_joint_names = [
            "left_joint1",
            "left_joint2",
            "left_joint3",
            "left_joint4",
            "left_joint5",
        ]
        self.right_arm_joint_names = [
            "right_joint1",
            "right_joint2",
            "right_joint3",
            "right_joint4",
            "right_joint5",
        ]
        self.arm_joint_names = self.left_arm_joint_names + self.right_arm_joint_names

        self.left_gripper_joint_names = ["left_finger_joint1", "left_finger_joint2"]
        self.right_gripper_joint_names = ["right_finger_joint1", "right_finger_joint2"]

        # Moderate, STABLE PD gains. The gripper masses are trimmed to realistic values
        # (see tools/build_pro_urdf.py) so the arm no longer sags much at these gains,
        # and 100 Hz-sim numerical stability requires keeping stiffness well below the
        # ~1e4 range where the coupled stiff arm+base rings/blows up.
        self.arm_stiffness = 2e3
        self.arm_damping = 4e2
        self.arm_force_limit = 300

        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 15.0

        # Base + lift: modestly firmer than the Std defaults (heavier arm) but still in
        # the numerically-stable range. Set before base-agent init; the create_* override
        # below makes the base gains timing-independent.
        self.base_pos_stiffness = 2e4
        self.base_pos_damping = 2e3
        self.base_pos_force_limit = 4000
        self.lift_stiffness = 4e3
        self.lift_damping = 6e2
        self.lift_force_limit = 400

        # Skip AlohaMiniSO100V2.__init__ (it hardcodes the 5-DOF names); go straight
        # to the shared base-agent initializer.
        AlohaMiniBaseAgent.__init__(self, *args, **kwargs)

    def _create_base_pos_controller(self):
        from mani_skill.agents.controllers import PDJointPosControllerConfig

        return PDJointPosControllerConfig(
            self.base_joint_names,
            lower=None,
            upper=None,
            stiffness=2e4,
            damping=2e3,
            force_limit=4000,
            normalize_action=False,
        )

    def _create_lift_pos_controller(self):
        from mani_skill.agents.controllers import PDJointPosControllerConfig

        return PDJointPosControllerConfig(
            self.lift_joint_names,
            lower=self.LIFT_LOWER,
            upper=self.LIFT_UPPER,
            stiffness=4e3,
            damping=6e2,
            force_limit=400,
            normalize_action=False,
        )

    # per-side bits used to disable gripper<->arm self-collision (see _after_init)
    _LEFT_ARM_GRIP_BIT = 25
    _RIGHT_ARM_GRIP_BIT = 26
    # NOTE: an earlier build shared a bit-24 with the table so the base could sit inside
    # the table footprint. NAV/MANIP separation drives the base to a physically-valid
    # station instead, so the base now collides with the table normally.

    def _after_init(self):
        # Sets up palm/finger links + finger-finger collision bits (Std logic).
        super()._after_init()
        # The parallel gripper is grafted onto link6; its palm/clamp meshes overlap
        # the wrist links, so disable collision between the whole gripper subtree and
        # the arm on each side (the gripper is rigidly attached -> such contacts are
        # spurious and blow up the sim). Fingers still collide with objects.
        robot_links = {l.name: l for l in self.robot.get_links()}
        for side, bit in (("left", self._LEFT_ARM_GRIP_BIT), ("right", self._RIGHT_ARM_GRIP_BIT)):
            # include the arm's mounting bracket ({side}_base): IK configs that pitch the
            # shoulder far (e.g. q2~-1.1 for low tabletop grasps) press link2/link5 into
            # it — the convex-hull contact then shoves the arm ~0.7 rad off the commanded
            # config (TCP ~150 mm high) and every grasp closes on air.
            # vertical_link (the lift column both arms mount on) likewise: its convex
            # hull grazes link2 in low-reach configs (impulse ~28 -> ~17 mm TCP error).
            # links 1-4 only: link5/link6 (the original jaw gripper) are deleted.
            group = ["vertical_link", f"{side}_base"] + [f"{side}_link{i}" for i in range(1, 5)]
            group += [
                f"{side}_Fixed_Jaw",
                f"{side}_finger1",
                f"{side}_finger2",
                f"{side}_finger1_tip",
                f"{side}_finger2_tip",
            ]
            for name in group:
                link = robot_links.get(name)
                if link is not None:
                    link.set_collision_group_bit(group=2, bit_idx=bit, bit=1)
