"""Config-driven skill sequence planner."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import PlannerConfig
from ..registry import register_planner
from ..skills import build_skill
from ..types import ActionStep, Scene, Sequence


@register_planner("skill_sequence")
@register_planner("SkillSequencePlanner")
class SkillSequencePlanner:
    """Concatenate ordered skill plans into one episode Sequence."""

    def __init__(self, cfg: PlannerConfig) -> None:
        self.cfg = cfg
        if cfg.arm_dof != 5:
            raise ValueError(
                "AlohaMini SO100 skill_sequence currently supports the native "
                "5-DOF arm layout only."
            )
        if not cfg.skills:
            raise ValueError("planner.skills must contain at least one ordered skill.")

    def plan(self, scene: Scene) -> Sequence:
        steps: list[ActionStep] = []
        skill_metadata: list[dict[str, Any]] = []

        for skill_index, raw_skill_cfg in enumerate(self.cfg.skills):
            skill_cfg = dict(raw_skill_cfg)
            skill_name = str(skill_cfg.pop("name"))
            params = dict(skill_cfg.pop("params", {}))
            params.update(skill_cfg)
            skill = build_skill(skill_name)
            actions = skill.plan(scene.env, params)
            phases = skill.last_phases or [skill_name] * len(actions)
            infos = skill.last_info or [{} for _ in actions]

            for local_index, action in enumerate(actions):
                phase = phases[local_index] if local_index < len(phases) else skill_name
                info = dict(infos[local_index]) if local_index < len(infos) else {}
                info.update({"skill": skill_name, "skill_index": skill_index})
                steps.append(
                    ActionStep(
                        action=np.asarray(action, dtype=np.float32).reshape(16),
                        phase=phase,
                        info=info,
                    )
                )

            skill_metadata.append(
                {
                    "name": skill_name,
                    "params": _serializable(params),
                    "num_steps": len(actions),
                    "metadata": _serializable(skill.metadata),
                }
            )

        truncated = False
        if self.cfg.max_steps > 0 and len(steps) > self.cfg.max_steps:
            steps = steps[: self.cfg.max_steps]
            truncated = True

        return Sequence(
            episode_index=scene.episode_index,
            task=scene.task,
            steps=steps,
            metadata={
                "planner": self.__class__.__name__,
                "action_layout": "base3,lift1,left_arm5,left_gripper1,right_arm5,right_gripper1",
                "arm_dof": self.cfg.arm_dof,
                "skill_interface": "BaseSkill.plan(env, params) -> list[np.ndarray] of 16-D actions",
                "skills": skill_metadata,
                "objects": _serializable(self.cfg.objects),
                "regions": _serializable(self.cfg.regions),
                "truncated_to_max_steps": truncated,
            },
        )


def _serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
