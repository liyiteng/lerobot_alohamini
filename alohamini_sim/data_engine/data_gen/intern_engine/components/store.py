"""Storage component wrapper."""

from __future__ import annotations

from ..config import StoreStageConfig
from ..registry import build_writer
from ..types import Observations, Scene, Sequence


class StoreComponent:
    def __init__(self, cfg: StoreStageConfig) -> None:
        self.writer = build_writer(cfg.name, cfg)

    def write_episode(
        self, scene: Scene, sequence: Sequence, observations: Observations
    ) -> None:
        self.writer.write_episode(scene, sequence, observations)

    def close(self) -> dict:
        return self.writer.close()
