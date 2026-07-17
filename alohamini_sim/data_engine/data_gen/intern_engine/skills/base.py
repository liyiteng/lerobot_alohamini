"""InternDataEngine-style skill interface for the ManiSkill data engine.

InternDataEngine registers small skill classes from config, asks each skill to
plan a sub-trajectory, and sequences those sub-trajectories into an episode. The
local mirror is intentionally simpler: a skill implements
``plan(env, params) -> list[np.ndarray]`` where every array is one 16-D
AlohaMini controller action in layout
``base3,lift1,left_arm5,left_gripper1,right_arm5,right_gripper1``. Skills keep
per-plan metadata/phases on the instance so the sequence planner can annotate
the generated episode while preserving this compact planning interface.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

import numpy as np

T = TypeVar("T", bound=type["BaseSkill"])

SKILL_REGISTRY: dict[str, type["BaseSkill"]] = {}


def normalize_skill_name(name: str) -> str:
    """Normalize config-facing skill names and class names to registry keys."""

    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip())
    text = text.replace("-", "_").replace(" ", "_").lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if text.endswith("_skill"):
        text = text[: -len("_skill")]
    return text


def register_skill(name: str | T | None = None) -> Callable[[T], T] | T:
    """Register a skill class.

    Can be used as ``@register_skill`` or ``@register_skill("pick")``.
    """

    if isinstance(name, type):
        return _register_skill_class(name, None)

    def _wrap(cls: T) -> T:
        return _register_skill_class(cls, name)

    return _wrap


def _register_skill_class(cls: T, explicit_name: str | None) -> T:
    aliases = {
        normalize_skill_name(cls.__name__),
        normalize_skill_name(cls.__name__.removesuffix("Skill")),
    }
    if explicit_name is not None:
        aliases.add(normalize_skill_name(explicit_name))

    for alias in aliases:
        if not alias:
            continue
        previous = SKILL_REGISTRY.get(alias)
        if previous is not None and previous is not cls:
            raise KeyError(f"Skill {alias!r} is already registered.")
        SKILL_REGISTRY[alias] = cls
    return cls


def get_skill_cls(name: str) -> type["BaseSkill"]:
    key = normalize_skill_name(name)
    try:
        return SKILL_REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted(SKILL_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown skill {name!r}. Registered: {known}") from exc


def build_skill(name: str) -> "BaseSkill":
    return get_skill_cls(name)()


class BaseSkill(ABC):
    """Base class for config-instantiated AlohaMini skills."""

    action_dim = 16
    action_layout = "base3,lift1,left_arm5,left_gripper1,right_arm5,right_gripper1"
    open_gripper = 0.037
    closed_gripper = 0.0

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.last_phases: list[str] = []
        self.last_info: list[dict[str, Any]] = []

    @abstractmethod
    def plan(self, env: Any, params: dict[str, Any]) -> list[np.ndarray]:
        """Return one episode sub-trajectory as 16-D controller actions."""

    def reset_trace(self) -> None:
        self.metadata = {}
        self.last_phases = []
        self.last_info = []

    def set_trace(
        self,
        phases: list[str],
        info: list[dict[str, Any]] | None = None,
    ) -> None:
        self.last_phases = list(phases)
        self.last_info = list(info) if info is not None else [{} for _ in phases]

    def validate_actions(self, actions: list[np.ndarray]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for action in actions:
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            if arr.size != self.action_dim:
                raise ValueError(
                    f"{self.__class__.__name__} emitted {arr.size}-D action; "
                    f"expected {self.action_dim}."
                )
            out.append(arr)
        return out

    def arm_layout(self, agent: Any) -> dict[str, Any]:
        """Controller action layout, derived from the agent's arm DOF (5=Std, 6=Pro).

        Flat action = base(3), lift(1), left_arm(n), left_grip(1), right_arm(n),
        right_grip(1)  =>  total dim 6 + 2n.
        """
        n = len(getattr(agent, "left_arm_joint_names", ["a"] * 5))
        return {
            "n": n,
            "dim": 6 + 2 * n,
            "left_arm": slice(4, 4 + n),
            "left_grip": 4 + n,
            "right_arm": slice(5 + n, 5 + 2 * n),
            "right_grip": 5 + 2 * n,
        }

    def current_action_template(self, env: Any, arm_dof: int | None = None) -> np.ndarray:
        """Build a hold-current action in the AlohaMini controller layout (DOF-agnostic)."""

        env_unwrapped = getattr(env, "unwrapped", env)
        agent = env_unwrapped.agent
        robot = agent.robot
        qpos = _to_numpy(robot.get_qpos())
        if qpos.ndim == 2:
            qpos = qpos[0]
        names = [joint.name for joint in robot.active_joints]
        index = {name: i for i, name in enumerate(names)}

        lay = self.arm_layout(agent)
        n = lay["n"]
        action = np.zeros(lay["dim"], dtype=np.float32)
        action[3] = _named(qpos, index, getattr(agent, "lift_joint_names", []), 1)[0]
        action[lay["left_arm"]] = _named(qpos, index, getattr(agent, "left_arm_joint_names", []), n)
        action[lay["left_grip"]] = _named(qpos, index, getattr(agent, "left_gripper_joint_names", []), 1)[0]
        action[lay["right_arm"]] = _named(qpos, index, getattr(agent, "right_arm_joint_names", []), n)
        action[lay["right_grip"]] = _named(qpos, index, getattr(agent, "right_gripper_joint_names", []), 1)[0]
        # In pd_joint_pos_fixed_base, [0, 0, 0] holds the virtual base rigid.
        action[:3] = 0.0
        self._layout = lay
        return action

    def set_arm_action(
        self,
        action: np.ndarray,
        arm: str,
        arm_qpos: np.ndarray,
        gripper: float | None = None,
    ) -> np.ndarray:
        lay = getattr(self, "_layout", None)
        if lay is None or lay.get("dim") != action.shape[-1]:
            n = (action.shape[-1] - 6) // 2
            lay = {"n": n, "left_arm": slice(4, 4 + n), "left_grip": 4 + n,
                   "right_arm": slice(5 + n, 5 + 2 * n), "right_grip": 5 + 2 * n}
        arm_qpos = np.asarray(arm_qpos, dtype=np.float32).reshape(lay["n"])
        if arm == "left":
            action[lay["left_arm"]] = arm_qpos
            if gripper is not None:
                action[lay["left_grip"]] = float(gripper)
        elif arm == "right":
            action[lay["right_arm"]] = arm_qpos
            if gripper is not None:
                action[lay["right_grip"]] = float(gripper)
        else:
            raise ValueError(f"Unsupported arm {arm!r}; expected 'left' or 'right'.")
        return action


def _named(
    qpos: np.ndarray,
    index: dict[str, int],
    names: list[str],
    count: int,
) -> np.ndarray:
    values: list[float] = []
    for name in names[:count]:
        values.append(float(qpos[index[name]]) if name in index else 0.0)
    while len(values) < count:
        values.append(0.0)
    return np.asarray(values, dtype=np.float32)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)
