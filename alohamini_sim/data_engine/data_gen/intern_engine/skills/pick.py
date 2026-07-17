"""Pick skill for the AlohaMini SO100 5-DOF arm."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseSkill, register_skill
from .ik import actor_position, apply_horizontal_jaw_wrist_roll, resolve_actor, solve_arm_ik_position


@register_skill("pick")
class PickSkill(BaseSkill):
    """Open above an object, descend, close, and lift with the vertical column."""

    def plan(self, env: Any, params: dict[str, Any]) -> list[np.ndarray]:
        self.reset_trace()
        params = dict(params or {})
        arm = params.get("arm", "left")
        object_name = _object_name(params)
        target = actor_position(resolve_actor(env, object_name))

        approach_height = float(params.get("approach_height", 0.10))
        descend_offset = float(params.get("descend_offset", -0.01))
        lift_start = float(params.get("lift_start", 0.0))
        lift_height = float(params.get("lift_height", 0.16))
        open_gripper = float(params.get("open_gripper", self.open_gripper))
        closed_gripper = float(params.get("closed_gripper", self.closed_gripper))
        approach_steps = int(params.get("approach_steps", 30))
        descend_steps = int(params.get("descend_steps", 35))
        close_steps = int(params.get("close_steps", 45))
        lift_steps = int(params.get("lift_steps", 120))
        hold_steps = int(params.get("hold_steps", 40))
        shoulder_lift_seed = float(params.get("shoulder_lift_seed", 1.0))

        approach = solve_arm_ik_position(
            env,
            target + np.array([0.0, 0.0, approach_height], dtype=np.float32),
            arm=arm,
            lift_position=lift_start,
            shoulder_lift_seed=shoulder_lift_seed,
        )
        approach = apply_horizontal_jaw_wrist_roll(env, approach)

        descend = solve_arm_ik_position(
            env,
            target + np.array([0.0, 0.0, descend_offset], dtype=np.float32),
            arm=arm,
            seed=approach.arm_qpos,
            lift_position=lift_start,
            shoulder_lift_seed=shoulder_lift_seed,
        )
        descend = apply_horizontal_jaw_wrist_roll(env, descend)

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

        append_phase("pick/approach", approach_steps, approach.arm_qpos, open_gripper, lift_start)
        append_phase("pick/descend", descend_steps, descend.arm_qpos, open_gripper, lift_start)
        append_phase("pick/close", close_steps, descend.arm_qpos, closed_gripper, lift_start)
        for lift in np.linspace(lift_start, lift_height, max(1, lift_steps)):
            action = template.copy()
            action[3] = float(lift)
            self.set_arm_action(action, arm, descend.arm_qpos, closed_gripper)
            actions.append(action.astype(np.float32))
            phases.append("pick/lift")
        append_phase("pick/hold_lift", hold_steps, descend.arm_qpos, closed_gripper, lift_height)

        self.metadata = {
            "skill": "pick",
            "arm": arm,
            "object_actor": object_name,
            "object_position": target.tolist(),
            "ik": {
                "approach_success": approach.success,
                "approach_error": approach.error,
                "approach_iterations": approach.iterations,
                "approach_wrist_roll": approach.wrist_roll,
                "approach_wrist_roll_score": approach.wrist_roll_score,
                "descend_success": descend.success,
                "descend_error": descend.error,
                "descend_iterations": descend.iterations,
                "descend_wrist_roll": descend.wrist_roll,
                "descend_wrist_roll_score": descend.wrist_roll_score,
            },
            "phase_counts": {
                "approach": approach_steps,
                "descend": descend_steps,
                "close": close_steps,
                "lift": lift_steps,
                "hold_lift": hold_steps,
            },
        }
        self.set_trace(phases)
        return self.validate_actions(actions)


def _object_name(params: dict[str, Any]) -> str:
    if "object_actor" in params:
        return str(params["object_actor"])
    if "object" in params:
        return str(params["object"])
    objects = params.get("objects")
    if objects:
        return str(objects[0])
    return "cube"
