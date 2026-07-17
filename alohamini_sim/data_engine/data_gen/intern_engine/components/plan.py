"""Planner wrapper that converts a planner object into a Sequence."""

from __future__ import annotations

from ..config import PlannerConfig
from ..registry import build_planner
from ..types import Scene, Sequence


class PlannerSequenceBuilder:
    def __init__(self, cfg: PlannerConfig) -> None:
        self.planner = build_planner(cfg.name, cfg)

    def plan(self, scene: Scene) -> Sequence:
        sequence = self.planner.plan(scene)
        if not isinstance(sequence, Sequence):
            raise TypeError(
                f"Planner {self.planner.__class__.__name__} returned "
                f"{type(sequence)!r}, expected Sequence."
            )
        return sequence
