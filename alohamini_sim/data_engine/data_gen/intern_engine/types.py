"""Shared dataclasses passed between InternDataEngine stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Scene:
    """A reset ManiSkill environment and per-episode metadata."""

    env: Any
    episode_index: int
    task: str
    seed: int | None = None
    reset_obs: Any = None
    reset_info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionStep:
    """One controller action in AlohaMini controller order."""

    action: np.ndarray
    phase: str
    duration: int = 1
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sequence:
    """A scripted action sequence for one episode."""

    episode_index: int
    task: str
    steps: list[ActionStep]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.steps)


@dataclass
class Observations:
    """Rollout observations aligned one-to-one with Sequence.steps."""

    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    images: dict[str, list[np.ndarray]] = field(default_factory=dict)
    timestamps: list[float] = field(default_factory=list)
    step_infos: list[dict[str, Any]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    missing_cameras: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
