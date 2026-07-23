"""Pipeline orchestration for AlohaMini ManiSkill data generation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .config import DataEngineConfig, ensure_config
from .types import Observations, Scene, Sequence

# Import modules for registry side effects.
from .components import load as _load_components  # noqa: F401
from .components import render as _render_components  # noqa: F401
from .components import store as _store_components  # noqa: F401
from .planners import aloha_pick_cube as _aloha_pick_cube  # noqa: F401
from .workflows import skill_sequence as _skill_sequence  # noqa: F401
from .writers import lerobot_v21 as _lerobot_v21  # noqa: F401
from .registry import build_loader, build_renderer


class LoadStage:
    def __init__(self, cfg: DataEngineConfig) -> None:
        self.cfg = cfg
        self.loader = build_loader(cfg.load.name, cfg.load)
        set_render_config = getattr(self.loader, "set_render_config", None)
        if callable(set_render_config):
            set_render_config(cfg.render)

    def setup(self) -> Any:
        return self.loader.load()

    def run(self, episode_index: int) -> Scene:
        seed = None
        if self.cfg.seed is not None:
            seed = int(self.cfg.seed) + episode_index
        return self.loader.reset(
            episode_index=episode_index,
            seed=seed,
            task=self.cfg.planner.task,
        )

    def close(self) -> None:
        self.loader.close()


class PlanStage:
    def __init__(self, cfg: DataEngineConfig) -> None:
        from .components.plan import PlannerSequenceBuilder

        self.planner = PlannerSequenceBuilder(cfg.planner)

    def run(self, scene: Scene) -> Sequence:
        return self.planner.plan(scene)


class RenderStage:
    def __init__(self, cfg: DataEngineConfig) -> None:
        self.renderer = build_renderer(cfg.render.name, cfg.render)

    def run(self, scene: Scene, sequence: Sequence) -> Observations:
        return self.renderer.render(scene, sequence)


class StoreStage:
    def __init__(self, cfg: DataEngineConfig) -> None:
        from .components.store import StoreComponent

        self.store = StoreComponent(cfg.store)

    def run(self, scene: Scene, sequence: Sequence, observations: Observations) -> None:
        self.store.write_episode(scene, sequence, observations)

    def close(self) -> dict[str, Any]:
        return self.store.close()


class ManiSkillDataEngine:
    """Run Load -> Plan -> Render -> Store for N episodes."""

    def __init__(self, config: DataEngineConfig | str) -> None:
        self.cfg = ensure_config(config)

    def run(self) -> dict[str, Any]:
        load_stage = LoadStage(self.cfg)
        plan_stage = PlanStage(self.cfg)
        render_stage = RenderStage(self.cfg)
        store_stage = StoreStage(self.cfg)

        summary: dict[str, Any] = {
            "config": asdict(self.cfg),
            "episodes": self.cfg.num_episodes,
        }
        try:
            load_stage.setup()
            for episode_index in range(self.cfg.num_episodes):
                scene = load_stage.run(episode_index)
                sequence = plan_stage.run(scene)
                observations = render_stage.run(scene, sequence)
                store_stage.run(scene, sequence, observations)
            summary.update(store_stage.close())
            return summary
        finally:
            load_stage.close()
