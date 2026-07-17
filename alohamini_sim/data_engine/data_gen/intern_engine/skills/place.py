"""Place skill for moving a held AlohaMini object to a target pose."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseSkill, register_skill
from .ik import actor_position, apply_horizontal_jaw_wrist_roll, resolve_actor, solve_arm_ik_position


@register_skill("place")
class PlaceSkill(BaseSkill):
    """Move a held object to target xy/z, open the gripper, and retreat."""

    def plan(self, env: Any, params: dict[str, Any]) -> list[np.ndarray]:
        self.reset_trace()
        params = dict(params or {})
        arm = params.get("arm", "left")
        target = _target_position(env, params)

        lift_height = float(params.get("lift_height", 0.16))
        open_gripper = float(params.get("open_gripper", self.open_gripper))
        closed_gripper = float(params.get("closed_gripper", self.closed_gripper))
        pre_place_height = float(params.get("pre_place_height", 0.10))
        place_height_offset = float(params.get("place_height_offset", 0.0))
        retreat_height = float(params.get("retreat_height", pre_place_height))
        move_steps = int(params.get("move_steps", 60))
        descend_steps = int(params.get("descend_steps", 45))
        open_steps = int(params.get("open_steps", 35))
        retreat_steps = int(params.get("retreat_steps", 40))
        shoulder_lift_seed = float(params.get("shoulder_lift_seed", 1.0))

        pre_target = target + np.array([0.0, 0.0, pre_place_height], dtype=np.float32)
        place_target = target + np.array([0.0, 0.0, place_height_offset], dtype=np.float32)
        retreat_target = target + np.array([0.0, 0.0, retreat_height], dtype=np.float32)

        pre_place = solve_arm_ik_position(
            env,
            pre_target,
            arm=arm,
            lift_position=lift_height,
            shoulder_lift_seed=shoulder_lift_seed,
        )
        pre_place = apply_horizontal_jaw_wrist_roll(env, pre_place)

        place = solve_arm_ik_position(
            env,
            place_target,
            arm=arm,
            seed=pre_place.arm_qpos,
            lift_position=lift_height,
            shoulder_lift_seed=shoulder_lift_seed,
        )
        place = apply_horizontal_jaw_wrist_roll(env, place)

        retreat = solve_arm_ik_position(
            env,
            retreat_target,
            arm=arm,
            seed=place.arm_qpos,
            lift_position=lift_height,
            shoulder_lift_seed=shoulder_lift_seed,
        )
        retreat = apply_horizontal_jaw_wrist_roll(env, retreat)

        template = self.current_action_template(env)
        template[:3] = 0.0
        if arm == "left":
            template[15] = open_gripper
        else:
            template[9] = open_gripper

        actions: list[np.ndarray] = []
        phases: list[str] = []

        def append_phase(
            phase: str,
            count: int,
            arm_qpos: np.ndarray,
            gripper: float,
            lift: float,
        ) -> None:
            for _ in range(max(0, count)):
                action = template.copy()
                action[3] = float(lift)
                self.set_arm_action(action, arm, arm_qpos, gripper)
                actions.append(action.astype(np.float32))
                phases.append(phase)

        append_phase("place/pre_place", move_steps, pre_place.arm_qpos, closed_gripper, lift_height)
        append_phase("place/descend", descend_steps, place.arm_qpos, closed_gripper, lift_height)
        append_phase("place/open", open_steps, place.arm_qpos, open_gripper, lift_height)
        append_phase("place/retreat", retreat_steps, retreat.arm_qpos, open_gripper, lift_height)

        self.metadata = {
            "skill": "place",
            "arm": arm,
            "target_position": target.tolist(),
            "ik": {
                "pre_place_success": pre_place.success,
                "pre_place_error": pre_place.error,
                "pre_place_wrist_roll": pre_place.wrist_roll,
                "place_success": place.success,
                "place_error": place.error,
                "place_wrist_roll": place.wrist_roll,
                "retreat_success": retreat.success,
                "retreat_error": retreat.error,
                "retreat_wrist_roll": retreat.wrist_roll,
            },
            "phase_counts": {
                "move": move_steps,
                "descend": descend_steps,
                "open": open_steps,
                "retreat": retreat_steps,
            },
        }
        self.set_trace(phases)
        return self.validate_actions(actions)


def _target_position(env: Any, params: dict[str, Any]) -> np.ndarray:
    if "target_xyz" in params:
        return np.asarray(params["target_xyz"], dtype=np.float32).reshape(3)

    if "target_actor" in params:
        target = actor_position(resolve_actor(env, str(params["target_actor"])))
    elif "target_xy" in params:
        target_xy = np.asarray(params["target_xy"], dtype=np.float32).reshape(2)
        target_z = float(params.get("target_z", 0.72))
        target = np.asarray([target_xy[0], target_xy[1], target_z], dtype=np.float32)
    else:
        target = np.asarray([0.0, -0.35, 0.72], dtype=np.float32)

    offset = np.asarray(params.get("target_offset", [0.0, 0.0, 0.0]), dtype=np.float32)
    return target + offset.reshape(3)
