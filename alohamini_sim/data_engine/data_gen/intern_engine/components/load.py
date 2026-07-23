"""ManiSkill environment loading and lightweight domain randomization."""

from __future__ import annotations

from typing import Any

from ..config import DomainRandomizerConfig, LoadStageConfig, RenderStageConfig
from ..registry import build_randomizer, register_loader, register_randomizer
from ..types import Scene


@register_randomizer("aloha_mini_domain_randomizer")
class AlohaMiniDomainRandomizer:
    """Small hook for task-level domain randomization.

    The current tabletop task exposes cube_xy_noise in its constructor. This
    hook also tries to update the unwrapped env attribute before reset so the
    same object can be reused across episodes.
    """

    def __init__(self, cfg: DomainRandomizerConfig) -> None:
        self.cfg = cfg

    def before_reset(self, env: Any, episode_index: int, seed: int | None) -> None:
        del episode_index, seed
        if not self.cfg.enabled:
            return
        target = getattr(env, "unwrapped", env)
        if hasattr(target, "cube_xy_noise"):
            setattr(target, "cube_xy_noise", float(self.cfg.cube_xy_noise))


@register_loader("mani_skill_env_loader")
class ManiSkillEnvLoader:
    def __init__(self, cfg: LoadStageConfig) -> None:
        self.cfg = cfg
        self.render_cfg: RenderStageConfig | None = None
        self.env: Any = None
        self.randomizer = None
        if cfg.domain_randomizer.enabled:
            self.randomizer = build_randomizer(
                cfg.domain_randomizer.name, cfg.domain_randomizer
            )

    def set_render_config(self, cfg: RenderStageConfig) -> None:
        self.render_cfg = cfg

    def load(self) -> Any:
        if self.env is not None:
            return self.env
        try:
            import gymnasium as gym
        except Exception as exc:  # pragma: no cover - import depends on runtime env
            raise RuntimeError("gymnasium is required to create ManiSkill envs.") from exc

        # These imports register the custom robot and task with ManiSkill/gym.
        try:
            import agents.aloha_mini  # noqa: F401
            import data_gen.tasks  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Failed to import AlohaMini robot/task registration modules."
            ) from exc

        kwargs = dict(self.cfg.env_kwargs)
        if self.cfg.domain_randomizer.enabled and "cube_xy_noise" not in kwargs:
            kwargs["cube_xy_noise"] = self.cfg.domain_randomizer.cube_xy_noise

        sensor_configs = dict(self.cfg.sensor_configs)
        if "sensor_configs" in kwargs:
            sensor_configs.update(dict(kwargs.pop("sensor_configs")))
        shader_pack = self._requested_shader_pack()
        if shader_pack is not None:
            sensor_configs["shader_pack"] = shader_pack
            human_render_camera_configs = dict(
                kwargs.pop("human_render_camera_configs", {})
            )
            human_render_camera_configs["shader_pack"] = shader_pack
            kwargs["human_render_camera_configs"] = human_render_camera_configs
        if sensor_configs:
            kwargs["sensor_configs"] = sensor_configs

        self.env = gym.make(
            self.cfg.env_id,
            num_envs=self.cfg.num_envs,
            obs_mode=self.cfg.obs_mode,
            control_mode=self.cfg.control_mode,
            render_mode=self.cfg.render_mode,
            sim_backend=self.cfg.sim_backend,
            render_backend=self.cfg.render_backend,
            **kwargs,
        )
        if self.cfg.num_envs != 1:
            raise ValueError(
                "The first self-contained writer supports num_envs=1. "
                "Use multiple launcher processes for parallel collection."
            )
        return self.env

    def reset(self, episode_index: int, seed: int | None, task: str) -> Scene:
        if self.env is None:
            self.load()
        if self.randomizer is not None:
            self.randomizer.before_reset(self.env, episode_index, seed)
        reset_result = self.env.reset(seed=seed, options={"episode_index": episode_index})
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs, info = reset_result, {}
        metadata = {
            "env_id": self.cfg.env_id,
            "control_mode": self.cfg.control_mode,
        }
        episode_metadata = getattr(
            getattr(self.env, "unwrapped", self.env), "get_episode_metadata", None
        )
        if callable(episode_metadata):
            metadata.update(episode_metadata())
        return Scene(
            env=self.env,
            episode_index=episode_index,
            task=task,
            seed=seed,
            reset_obs=obs,
            reset_info=info or {},
            metadata=metadata,
        )

    def close(self) -> None:
        if self.env is not None:
            close = getattr(self.env, "close", None)
            if close is not None:
                close()
            self.env = None

    def _requested_shader_pack(self) -> str | None:
        if self.render_cfg is None:
            return None
        shader_pack = self.render_cfg.shader_pack
        if shader_pack == "minimal":
            return None
        return shader_pack
