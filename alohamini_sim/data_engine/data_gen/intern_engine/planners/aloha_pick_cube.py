"""Deterministic SO100 pick-cube script for AlohaMini."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import PlannerConfig
from ..registry import register_planner
from ..types import ActionStep, Scene, Sequence


@register_planner("so100_scripted_pick_cube")
@register_planner("SO100ScriptedPickPlanner")
class SO100ScriptedPickPlanner:
    """Open above cube, lower, close, then lift.

    The waypoints are intentionally conservative joint-space targets. This is a
    data-engine skeleton; grasp reliability is expected to be tuned against the
    live simulator.
    """

    def __init__(self, cfg: PlannerConfig) -> None:
        self.cfg = cfg
        if cfg.arm_dof < 1:
            raise ValueError("arm_dof must be positive.")

    def plan(self, scene: Scene) -> Sequence:
        env = scene.env
        base, lift, left_arm, right_arm = self._current_targets(env)
        del lift
        right_hold = self._fit_arm(
            self.cfg.right_arm_hold if self.cfg.right_arm_hold is not None else right_arm
        )
        approach = self._fit_arm(self.cfg.left_arm_approach)
        lower = self._fit_arm(self.cfg.left_arm_lower)
        lift_arm = self._fit_arm(self.cfg.left_arm_lift)
        cube_p = self._cube_position(env)

        steps: list[ActionStep] = []
        steps.extend(
            self._repeat_phase(
                "open_above_cube",
                self.cfg.approach_steps,
                base,
                self.cfg.approach_lift,
                approach,
                self.cfg.open_gripper,
                right_hold,
                self.cfg.open_gripper,
                cube_p,
            )
        )
        steps.extend(
            self._repeat_phase(
                "lower_to_cube",
                self.cfg.lower_steps,
                base,
                self.cfg.lower_lift,
                lower,
                self.cfg.open_gripper,
                right_hold,
                self.cfg.open_gripper,
                cube_p,
            )
        )
        for i in range(max(1, self.cfg.close_steps)):
            alpha = i / max(1, self.cfg.close_steps - 1)
            grip = (1.0 - alpha) * self.cfg.open_gripper + alpha * self.cfg.close_gripper
            steps.append(
                self._step(
                    "close_left_gripper",
                    base,
                    self.cfg.lower_lift,
                    lower,
                    grip,
                    right_hold,
                    self.cfg.open_gripper,
                    cube_p,
                )
            )
        steps.extend(
            self._repeat_phase(
                "lift_cube",
                self.cfg.lift_steps,
                base,
                self.cfg.lifted_lift,
                lift_arm,
                self.cfg.close_gripper,
                right_hold,
                self.cfg.open_gripper,
                cube_p,
            )
        )
        steps.extend(
            self._repeat_phase(
                "hold_lift",
                self.cfg.hold_steps,
                base,
                self.cfg.lifted_lift,
                lift_arm,
                self.cfg.close_gripper,
                right_hold,
                self.cfg.open_gripper,
                cube_p,
            )
        )
        steps = steps[: self.cfg.max_steps]
        return Sequence(
            episode_index=scene.episode_index,
            task=scene.task,
            steps=steps,
            metadata={
                "planner": self.__class__.__name__,
                "cube_position": cube_p.tolist() if cube_p is not None else None,
                "action_layout": "base3,lift1,left_arm5,left_gripper1,right_arm5,right_gripper1",
                "arm_dof": self.cfg.arm_dof,
            },
        )

    def _repeat_phase(
        self,
        phase: str,
        count: int,
        base: np.ndarray,
        lift: float,
        left_arm: np.ndarray,
        left_grip: float,
        right_arm: np.ndarray,
        right_grip: float,
        cube_p: np.ndarray | None,
    ) -> list[ActionStep]:
        return [
            self._step(phase, base, lift, left_arm, left_grip, right_arm, right_grip, cube_p)
            for _ in range(max(0, count))
        ]

    def _step(
        self,
        phase: str,
        base: np.ndarray,
        lift: float,
        left_arm: np.ndarray,
        left_grip: float,
        right_arm: np.ndarray,
        right_grip: float,
        cube_p: np.ndarray | None,
    ) -> ActionStep:
        left_grip = float(np.clip(left_grip, 0.0, self.cfg.open_gripper))
        right_grip = float(np.clip(right_grip, 0.0, self.cfg.open_gripper))
        action = np.concatenate(
            [
                np.asarray(base, dtype=np.float32).reshape(3),
                np.asarray([lift], dtype=np.float32),
                self._fit_arm(left_arm),
                np.asarray([left_grip], dtype=np.float32),
                self._fit_arm(right_arm),
                np.asarray([right_grip], dtype=np.float32),
            ]
        ).astype(np.float32)
        return ActionStep(
            action=action,
            phase=phase,
            info={"cube_position": cube_p.tolist() if cube_p is not None else None},
        )

    def _fit_arm(self, values: Any) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size < self.cfg.arm_dof:
            arr = np.pad(arr, (0, self.cfg.arm_dof - arr.size))
        return arr[: self.cfg.arm_dof].astype(np.float32)

    def _current_targets(self, env: Any) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        robot = env.unwrapped.agent.robot
        qpos = self._to_numpy(robot.get_qpos())
        if qpos.ndim == 2:
            qpos = qpos[0]
        names = [joint.name for joint in robot.active_joints]
        index = {name: i for i, name in enumerate(names)}
        agent = env.unwrapped.agent
        base = self._named(qpos, index, getattr(agent, "base_joint_names", []), 3)
        lift = self._named(qpos, index, getattr(agent, "lift_joint_names", []), 1)[0]
        left = self._named(qpos, index, getattr(agent, "left_arm_joint_names", []), self.cfg.arm_dof)
        right = self._named(
            qpos, index, getattr(agent, "right_arm_joint_names", []), self.cfg.arm_dof
        )
        # In pd_joint_pos mode the base controller is velocity-based; [0, 0, 0]
        # keeps the generated script stationary. The fixed-base mode interprets
        # the same values as position targets.
        base = np.zeros(3, dtype=np.float32)
        return base, float(lift), left, right

    def _named(
        self, qpos: np.ndarray, index: dict[str, int], names: list[str], count: int
    ) -> np.ndarray:
        values: list[float] = []
        for name in names[:count]:
            values.append(float(qpos[index[name]]) if name in index else 0.0)
        while len(values) < count:
            values.append(0.0)
        return np.asarray(values, dtype=np.float32)

    def _cube_position(self, env: Any) -> np.ndarray | None:
        cube = getattr(env.unwrapped, "cube", None)
        if cube is None:
            return None
        pose = getattr(cube, "pose", None)
        if pose is None or not hasattr(pose, "p"):
            return None
        pos = self._to_numpy(pose.p)
        if pos.ndim == 2:
            pos = pos[0]
        return np.asarray(pos, dtype=np.float32).reshape(-1)[:3]

    def _to_numpy(self, value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)
