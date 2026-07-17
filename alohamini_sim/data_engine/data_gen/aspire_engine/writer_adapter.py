"""State-only LeRobot v2.1-style adapter for ASPIRE episodes.

The existing writer in intern_engine/writers/lerobot_v21.py is intentionally fixed
to a 12-D arm/gripper schema. AlohaMini Pro execution traces here use the full
18-D qpos state and 16-D controller action, so this adapter mirrors the writer's
directory and metadata shape without modifying the validated writer.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pa = None
    pq = None


STATE_NAMES = [
    "root_x_axis_joint",
    "root_y_axis_joint",
    "root_z_rotation_joint",
    "vertical_move",
    "left_joint1",
    "right_joint1",
    "left_joint2",
    "right_joint2",
    "left_joint3",
    "right_joint3",
    "left_joint4",
    "right_joint4",
    "left_joint5",
    "right_joint5",
    "left_finger_joint1",
    "left_finger_joint2",
    "right_finger_joint1",
    "right_finger_joint2",
]

ACTION_NAMES = [
    "root_x_axis_joint_target",
    "root_y_axis_joint_target",
    "root_z_rotation_joint_target",
    "vertical_move_target",
    "left_joint1_target",
    "left_joint2_target",
    "left_joint3_target",
    "left_joint4_target",
    "left_joint5_target",
    "left_gripper_target",
    "right_joint1_target",
    "right_joint2_target",
    "right_joint3_target",
    "right_joint4_target",
    "right_joint5_target",
    "right_gripper_target",
]


class AspireStateWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        fps: int = 20,
        dataset_name: str = "aloha_mini_pro_aspire_state",
        robot_type: str = "aloha_mini_pro_v2",
        overwrite: bool = True,
    ) -> None:
        self.root = Path(output_dir)
        self.data_dir = self.root / "data" / "chunk-000"
        self.meta_dir = self.root / "meta"
        self.fps = int(fps)
        self.dataset_name = dataset_name
        self.robot_type = robot_type
        self.overwrite = bool(overwrite)
        self.episodes: list[dict[str, Any]] = []
        self.tasks: dict[str, int] = {}
        self.total_frames = 0
        self.global_index = 0
        self.vector_stats: dict[str, dict[str, Any]] = {}
        self._prepared = False

    def write_episodes(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        self._prepare()
        self._check_deps()
        for episode_index, episode in enumerate(episodes):
            self.write_episode(episode, episode_index)
        return self.close()

    def write_episode(self, episode: dict[str, Any], episode_index: int) -> None:
        steps = list(episode.get("steps", []))
        states = np.asarray([step["qpos"] for step in steps], dtype=np.float32)
        actions = np.asarray([step["action"] for step in steps], dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 18:
            raise ValueError(f"observation.state must be [T,18], got {states.shape}")
        if actions.ndim != 2 or actions.shape[1] != 16:
            raise ValueError(f"action must be [T,16], got {actions.shape}")
        n = states.shape[0]
        if n == 0:
            raise ValueError("cannot write empty episode")

        task = str(episode.get("annotation") or episode.get("command", {}).get("verb") or "aspire_task")
        task_index = self._task_index(task)
        timestamps = np.arange(n, dtype=np.float32) / float(self.fps)
        parquet_path = self.data_dir / f"episode_{episode_index:06d}.parquet"
        self._write_parquet(parquet_path, episode_index, task_index, states, actions, timestamps)

        metadata = {
            "seed": episode.get("seed"),
            "success": bool(episode.get("success", False)),
            "command": episode.get("command", {}),
            "traces": episode.get("traces", []),
            "final": episode.get("final", {}),
        }
        self.episodes.append(
            {
                "episode_index": int(episode_index),
                "tasks": [task],
                "length": int(n),
                "metadata": metadata,
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
            "total_videos": 0,
        }

    def _prepare(self) -> None:
        if self._prepared:
            return
        if self.root.exists() and any(self.root.iterdir()):
            if not self.overwrite:
                raise FileExistsError(f"{self.root} already exists and is not empty")
            shutil.rmtree(self.root)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self._prepared = True

    def _check_deps(self) -> None:
        if pa is None or pq is None:
            raise RuntimeError("pyarrow is required for state-only parquet output")

    def _task_index(self, task: str) -> int:
        if task not in self.tasks:
            self.tasks[task] = len(self.tasks)
        return self.tasks[task]

    def _write_parquet(
        self,
        parquet_path: Path,
        episode_index: int,
        task_index: int,
        states: np.ndarray,
        actions: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        n = states.shape[0]
        columns: dict[str, Any] = {
            "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 18)),
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 16)),
            "episode_index": pa.array([episode_index] * n, type=pa.int64()),
            "frame_index": pa.array(list(range(n)), type=pa.int64()),
            "timestamp": pa.array(timestamps.tolist(), type=pa.float32()),
            "task_index": pa.array([task_index] * n, type=pa.int64()),
            "index": pa.array(list(range(self.global_index, self.global_index + n)), type=pa.int64()),
        }
        pq.write_table(pa.table(columns), parquet_path)
        self.global_index += n

    def _write_meta(self) -> None:
        self._write_json(self.meta_dir / "info.json", self._info_json())
        with (self.meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
            for episode in sorted(self.episodes, key=lambda item: item["episode_index"]):
                f.write(json.dumps(episode, sort_keys=True) + "\n")
        with (self.meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
            for task, task_index in sorted(self.tasks.items(), key=lambda item: item[1]):
                f.write(json.dumps({"task_index": task_index, "task": task}, sort_keys=True) + "\n")
        self._write_json(self.meta_dir / "stats.json", self._stats_json())

    def _info_json(self) -> dict[str, Any]:
        return {
            "codebase_version": "v2.1",
            "dataset_name": self.dataset_name,
            "robot_type": self.robot_type,
            "total_episodes": len(self.episodes),
            "total_frames": self.total_frames,
            "total_tasks": len(self.tasks),
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self.episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [18],
                    "names": STATE_NAMES,
                },
                "action": {
                    "dtype": "float32",
                    "shape": [16],
                    "names": ACTION_NAMES,
                },
                "episode_index": {"dtype": "int64", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
            },
        }

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

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
