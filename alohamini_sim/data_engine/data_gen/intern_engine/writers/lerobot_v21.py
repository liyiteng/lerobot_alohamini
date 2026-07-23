"""Self-contained LeRobot v2.1-style dataset writer.

Feature dictionary emitted to meta/info.json:

- observation.images.head: video, [H, W, 3], from cam_main
- observation.images.hand_left: video, [H, W, 3], from cam_left_wrist
- observation.images.hand_right: video, [H, W, 3], from cam_right_wrist
- observation.state: float32[12] =
  [left_arm5, left_gripper1, right_arm5, right_gripper1]
- action: float32[12] =
  [left_arm5, left_gripper1, right_arm5, right_gripper1]
- episode_index: int64
- frame_index: int64
- timestamp: float32
- task_index: int64
- index: int64

The writer does not import lerobot. It writes parquet with pyarrow, json/jsonl
metadata, and one mp4 per episode/camera with imageio under videos/chunk-000/.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..config import StoreStageConfig
from ..registry import register_writer
from ..types import Observations, Scene, Sequence

try:  # Runtime dependency in /home/perelman/Basic_RL/.venv.
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional import guard
    pa = None
    pq = None

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - optional import guard
    imageio = None


@register_writer("lerobot_v21")
class LeRobotV21Writer:
    def __init__(self, cfg: StoreStageConfig) -> None:
        self.cfg = cfg
        self.root = Path(cfg.output_dir)
        self.data_dir = self.root / "data" / "chunk-000"
        self.video_dir = self.root / "videos" / "chunk-000"
        self.meta_dir = self.root / "meta"
        self.episodes: list[dict[str, Any]] = []
        self.tasks: dict[str, int] = {}
        self.total_frames = 0
        self.total_videos = 0
        self.global_index = 0
        self.image_shapes: dict[str, list[int]] = {}
        self.vector_stats: dict[str, dict[str, Any]] = {}
        self._prepared = False

    def write_episode(
        self, scene: Scene, sequence: Sequence, observations: Observations
    ) -> None:
        self._prepare()
        self._check_deps()
        task_index = self._task_index(scene.task)
        states = np.asarray(observations.states, dtype=np.float32)
        actions = np.asarray(observations.actions, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 12:
            raise ValueError(f"observation.state must be [T,12], got {states.shape}.")
        if actions.ndim != 2 or actions.shape[1] != 12:
            raise ValueError(f"action must be [T,12], got {actions.shape}.")
        n = min(len(observations.timestamps), states.shape[0], actions.shape[0])
        if n == 0:
            raise ValueError("Cannot write an empty episode.")
        states = states[:n]
        actions = actions[:n]
        timestamps = np.asarray(observations.timestamps[:n], dtype=np.float32)
        episode_index = int(scene.episode_index)

        video_paths = self._write_videos(episode_index, observations.images, n)
        parquet_path = self.data_dir / f"episode_{episode_index:06d}.parquet"
        self._write_parquet(
            parquet_path=parquet_path,
            episode_index=episode_index,
            task_index=task_index,
            states=states,
            actions=actions,
            timestamps=timestamps,
            video_paths=video_paths,
        )

        success = bool(
            sequence.metadata.get(
                "success", observations.metadata.get("success", False)
            )
        )
        episode_metadata = {
            "seed": scene.seed,
            "success": success,
            "scene": scene.metadata,
            "evaluate": observations.metadata.get("evaluate", {}),
            "sequence": sequence.metadata,
            "observations": observations.metadata,
            "missing_cameras": observations.missing_cameras,
        }
        if "target_object_id" in scene.metadata:
            episode_metadata["object_id"] = scene.metadata["target_object_id"]

        self.episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [scene.task],
                "length": int(n),
                "metadata": episode_metadata,
            }
        )
        self.total_frames += int(n)
        self._update_vector_stats("observation.state", states)
        self._update_vector_stats("action", actions)

    def close(self) -> dict[str, Any]:
        self._prepare()
        self._write_meta()
        return {
            "output_dir": str(self.root),
            "total_episodes": len(self.episodes),
            "total_frames": self.total_frames,
            "total_tasks": len(self.tasks),
            "total_videos": self.total_videos,
        }

    def _prepare(self) -> None:
        if self._prepared:
            return
        if self.root.exists() and any(self.root.iterdir()):
            if not self.cfg.overwrite:
                raise FileExistsError(
                    f"{self.root} already exists and is not empty. "
                    "Set store.overwrite=true to replace it."
                )
            shutil.rmtree(self.root)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self._prepared = True

    def _check_deps(self) -> None:
        if pa is None or pq is None:
            raise RuntimeError("pyarrow is required for LeRobotV21Writer parquet output.")
        if imageio is None:
            raise RuntimeError("imageio is required for LeRobotV21Writer mp4 output.")

    def _task_index(self, task: str) -> int:
        if task not in self.tasks:
            self.tasks[task] = len(self.tasks)
        return self.tasks[task]

    def _write_videos(
        self, episode_index: int, images: dict[str, list[np.ndarray]], n: int
    ) -> dict[str, str]:
        video_paths: dict[str, str] = {}
        for short_name, frames in images.items():
            key = f"observation.images.{short_name}"
            if not frames:
                continue
            frames_np = [self._as_rgb(frame) for frame in frames[:n]]
            self.image_shapes[key] = list(frames_np[0].shape)
            out_dir = self.video_dir / key
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"episode_{episode_index:06d}.mp4"
            imageio.mimsave(
                out_path,
                frames_np,
                fps=self.cfg.fps,
                codec=self.cfg.video_codec,
                macro_block_size=1,
            )
            video_paths[key] = str(out_path.relative_to(self.root))
            self.total_videos += 1
        return video_paths

    def _write_parquet(
        self,
        parquet_path: Path,
        episode_index: int,
        task_index: int,
        states: np.ndarray,
        actions: np.ndarray,
        timestamps: np.ndarray,
        video_paths: dict[str, str],
    ) -> None:
        n = states.shape[0]
        columns: dict[str, Any] = {
            "observation.state": pa.array(
                states.tolist(), type=pa.list_(pa.float32(), states.shape[1])
            ),
            "action": pa.array(
                actions.tolist(), type=pa.list_(pa.float32(), actions.shape[1])
            ),
            "episode_index": pa.array([episode_index] * n, type=pa.int64()),
            "frame_index": pa.array(list(range(n)), type=pa.int64()),
            "timestamp": pa.array(timestamps.tolist(), type=pa.float32()),
            "task_index": pa.array([task_index] * n, type=pa.int64()),
            "index": pa.array(
                list(range(self.global_index, self.global_index + n)), type=pa.int64()
            ),
        }
        for key, rel_path in sorted(video_paths.items()):
            columns[key] = pa.StructArray.from_arrays(
                [
                    pa.array([rel_path] * n, type=pa.string()),
                    pa.array(timestamps.tolist(), type=pa.float32()),
                ],
                fields=[
                    pa.field("path", pa.string()),
                    pa.field("timestamp", pa.float32()),
                ],
            )
        pq.write_table(pa.table(columns), parquet_path)
        self.global_index += n

    def _write_meta(self) -> None:
        self._write_json(self.meta_dir / "info.json", self._info_json())
        with (self.meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
            for episode in sorted(self.episodes, key=lambda item: item["episode_index"]):
                f.write(json.dumps(episode, sort_keys=True) + "\n")
        with (self.meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
            for task, task_index in sorted(self.tasks.items(), key=lambda item: item[1]):
                f.write(
                    json.dumps(
                        {"task_index": task_index, "task": task}, sort_keys=True
                    )
                    + "\n"
                )
        self._write_json(self.meta_dir / "stats.json", self._stats_json())

    def _info_json(self) -> dict[str, Any]:
        return {
            "codebase_version": "v2.1",
            "dataset_name": self.cfg.dataset_name,
            "robot_type": self.cfg.robot_type,
            "total_episodes": len(self.episodes),
            "total_frames": self.total_frames,
            "total_tasks": len(self.tasks),
            "total_videos": self.total_videos,
            "total_chunks": 1,
            "chunks_size": self.cfg.chunk_size,
            "fps": self.cfg.fps,
            "splits": {"train": f"0:{len(self.episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": self._features(),
        }

    def _features(self) -> dict[str, Any]:
        features: dict[str, Any] = {
            "observation.state": {
                "dtype": "float32",
                "shape": [12],
                "names": [
                    "left_arm_0",
                    "left_arm_1",
                    "left_arm_2",
                    "left_arm_3",
                    "left_arm_4",
                    "left_gripper",
                    "right_arm_0",
                    "right_arm_1",
                    "right_arm_2",
                    "right_arm_3",
                    "right_arm_4",
                    "right_gripper",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": [12],
                "names": [
                    "left_arm_0",
                    "left_arm_1",
                    "left_arm_2",
                    "left_arm_3",
                    "left_arm_4",
                    "left_gripper",
                    "right_arm_0",
                    "right_arm_1",
                    "right_arm_2",
                    "right_arm_3",
                    "right_arm_4",
                    "right_gripper",
                ],
            },
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
        }
        for key, shape in sorted(self.image_shapes.items()):
            features[key] = {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": self.cfg.fps,
                    "video.codec": self.cfg.video_codec,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
        return features

    def _update_vector_stats(self, name: str, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        stat = self.vector_stats.setdefault(
            name,
            {
                "count": 0,
                "sum": np.zeros(values.shape[1], dtype=np.float64),
                "sumsq": np.zeros(values.shape[1], dtype=np.float64),
                "min": np.full(values.shape[1], np.inf, dtype=np.float64),
                "max": np.full(values.shape[1], -np.inf, dtype=np.float64),
            },
        )
        stat["count"] += values.shape[0]
        stat["sum"] += values.sum(axis=0)
        stat["sumsq"] += np.square(values).sum(axis=0)
        stat["min"] = np.minimum(stat["min"], values.min(axis=0))
        stat["max"] = np.maximum(stat["max"], values.max(axis=0))

    def _stats_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, stat in self.vector_stats.items():
            count = max(int(stat["count"]), 1)
            mean = stat["sum"] / count
            var = np.maximum(stat["sumsq"] / count - np.square(mean), 0.0)
            std = np.sqrt(var)
            out[name] = {
                "count": int(stat["count"]),
                "min": self._finite_list(stat["min"]),
                "max": self._finite_list(stat["max"]),
                "mean": self._finite_list(mean),
                "std": self._finite_list(std),
            }
        return out

    def _finite_list(self, values: np.ndarray) -> list[float | None]:
        result: list[float | None] = []
        for value in values.tolist():
            result.append(float(value) if math.isfinite(float(value)) else None)
        return result

    def _as_rgb(self, frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise ValueError(f"Expected HxWx3 RGB frame, got {arr.shape}.")
        arr = arr[..., :3]
        if arr.dtype.kind == "f":
            arr = np.clip(arr * 255.0, 0, 255)
        return np.asarray(arr, dtype=np.uint8)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
