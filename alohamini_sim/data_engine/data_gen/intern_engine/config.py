"""Typed YAML config loading for the ManiSkill data engine."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

VALID_SHADER_PACKS = {"minimal", "rt", "rt-fast"}


@dataclass
class DomainRandomizerConfig:
    name: str = "aloha_mini_domain_randomizer"
    enabled: bool = False
    cube_xy_noise: float = 0.0
    seed_offset: int = 0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadStageConfig:
    name: str = "mani_skill_env_loader"
    env_id: str = "AlohaMiniTablePick-v1"
    obs_mode: str = "rgb"
    control_mode: str = "pd_joint_pos_fixed_base"
    render_mode: str | None = "sensors"
    num_envs: int = 1
    sim_backend: str = "auto"
    render_backend: str = "gpu"
    sensor_configs: dict[str, Any] = field(default_factory=dict)
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    domain_randomizer: DomainRandomizerConfig = field(
        default_factory=DomainRandomizerConfig
    )


@dataclass
class PlannerConfig:
    name: str = "so100_scripted_pick_cube"
    task: str = "Pick the cube from the pedestal with the left gripper."
    arm_dof: int = 5
    max_steps: int = 120
    open_gripper: float = 0.037
    close_gripper: float = 0.0
    approach_lift: float = 0.12
    lower_lift: float = 0.04
    lifted_lift: float = 0.18
    approach_steps: int = 18
    lower_steps: int = 16
    close_steps: int = 12
    lift_steps: int = 18
    hold_steps: int = 8
    left_arm_approach: list[float] = field(
        default_factory=lambda: [-0.25, 0.55, -0.75, 0.35, 0.0]
    )
    left_arm_lower: list[float] = field(
        default_factory=lambda: [-0.25, 0.72, -1.05, 0.43, 0.0]
    )
    left_arm_lift: list[float] = field(
        default_factory=lambda: [-0.25, 0.35, -0.65, 0.25, 0.0]
    )
    right_arm_hold: list[float] | None = None
    objects: list[dict[str, Any]] = field(default_factory=list)
    regions: list[dict[str, Any]] = field(default_factory=list)
    skills: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RenderStageConfig:
    name: str = "mani_skill_renderer"
    fps: int = 30
    shader_pack: str = "minimal"
    cameras: list[str] = field(
        default_factory=lambda: ["cam_main", "cam_left_wrist", "cam_right_wrist"]
    )
    camera_map: dict[str, str] = field(
        default_factory=lambda: {
            "cam_main": "head",
            "cam_left_wrist": "hand_left",
            "cam_right_wrist": "hand_right",
        }
    )
    missing_camera: str = "black"
    default_image_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "cam_main": [240, 320, 3],
            "cam_left_wrist": [128, 128, 3],
            "cam_right_wrist": [128, 128, 3],
        }
    )

    def __post_init__(self) -> None:
        if self.shader_pack not in VALID_SHADER_PACKS:
            known = ", ".join(sorted(VALID_SHADER_PACKS))
            raise ValueError(
                f"render.shader_pack must be one of {known}; "
                f"got {self.shader_pack!r}."
            )


@dataclass
class StoreStageConfig:
    name: str = "lerobot_v21"
    output_dir: str = "data_gen/output/aloha_mini_lerobot"
    dataset_name: str = "aloha_mini_table_pick"
    robot_type: str = "aloha_mini_so100_v2"
    fps: int = 30
    overwrite: bool = False
    chunk_size: int = 1000
    video_codec: str = "libx264"


@dataclass
class DataEngineConfig:
    num_episodes: int = 1
    seed: int | None = 0
    load: LoadStageConfig = field(default_factory=LoadStageConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    render: RenderStageConfig = field(default_factory=RenderStageConfig)
    store: StoreStageConfig = field(default_factory=StoreStageConfig)


def load_config(path: str | Path) -> DataEngineConfig:
    """Load a YAML file through OmegaConf and return typed dataclasses."""

    path = Path(path)
    try:
        from omegaconf import OmegaConf
    except Exception as exc:  # pragma: no cover - exercised only without omegaconf
        raise RuntimeError(
            "OmegaConf is required to load data engine YAML configs."
        ) from exc

    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"Config at {path} must contain a mapping at the top level.")
    if "plan" in raw and "planner" not in raw:
        raw["planner"] = raw.pop("plan")
    merged = OmegaConf.merge(OmegaConf.structured(DataEngineConfig), OmegaConf.create(raw))
    obj = OmegaConf.to_object(merged)
    if isinstance(obj, DataEngineConfig):
        return obj
    return _coerce_dataclass(DataEngineConfig, obj)


def ensure_config(config_or_path: DataEngineConfig | str | Path) -> DataEngineConfig:
    if isinstance(config_or_path, DataEngineConfig):
        return config_or_path
    return load_config(config_or_path)


def _coerce_dataclass(cls: type[Any], data: Any) -> Any:
    if not is_dataclass(cls):
        return data
    if data is None:
        return cls()
    if is_dataclass(data):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"Cannot coerce {type(data)!r} into {cls.__name__}.")

    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name in data:
            kwargs[item.name] = _coerce_value(item.type, data[item.name])
        elif item.default is not MISSING:
            kwargs[item.name] = item.default
        elif item.default_factory is not MISSING:  # type: ignore[attr-defined]
            kwargs[item.name] = item.default_factory()  # type: ignore[misc]
    return cls(**kwargs)


def _coerce_value(type_hint: Any, value: Any) -> Any:
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    if is_dataclass(type_hint):
        return _coerce_dataclass(type_hint, value)
    if origin is list and args and value is not None:
        return list(value)
    if origin is dict and value is not None:
        return dict(value)
    if origin is None:
        return value
    return value
