from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.utils.utils import flatten_dict
from scripts.check_lerobot_dataset_integrity import DatasetRepairer, IntegrityChecker

FEATURES = {
    "action": {"dtype": "float32", "shape": [1], "names": ["joint"]},
    "observation.state": {"dtype": "float32", "shape": [1], "names": ["joint"]},
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


def _stats(values: dict[str, np.ndarray]) -> dict[str, object]:
    episode_stats = {key: get_feature_stats(value, axis=0, keepdims=False) for key, value in values.items()}
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in flatten_dict({"stats": episode_stats}).items()
    }


def _make_gapped_dataset(root: Path, *, metadata_ids: tuple[int, int] = (0, 2)) -> None:
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)

    old_episode_ids = np.array([0, 0, 2, 2], dtype=np.int64)
    frame_indices = np.array([0, 1, 0, 1], dtype=np.int64)
    global_indices = np.array([0, 1, 4, 5], dtype=np.int64)
    timestamps = np.array([0.0, 0.04, 0.0, 0.04], dtype=np.float32)
    action = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    state = action + 10
    task_indices = np.zeros(4, dtype=np.int64)

    data_table = pa.table(
        {
            "action": action.tolist(),
            "observation.state": state.tolist(),
            "timestamp": timestamps,
            "frame_index": frame_indices,
            "episode_index": old_episode_ids,
            "index": global_indices,
            "task_index": task_indices,
        }
    )
    pq.write_table(data_table, root / "data/chunk-000/file-000.parquet")

    rows = []
    for row_number, old_id in enumerate(metadata_ids):
        mask = old_episode_ids == (0 if row_number == 0 else 2)
        values = {
            "action": action[mask],
            "observation.state": state[mask],
            "timestamp": timestamps[mask].reshape(-1, 1),
            "frame_index": frame_indices[mask].reshape(-1, 1),
            "episode_index": old_episode_ids[mask].reshape(-1, 1),
            "index": global_indices[mask].reshape(-1, 1),
            "task_index": task_indices[mask].reshape(-1, 1),
        }
        row = {
            "episode_index": old_id,
            "tasks": ["test"],
            "length": 2,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 0 if row_number == 0 else 2,
            "dataset_to_index": 2 if row_number == 0 else 4,
        }
        row.update(_stats(values))
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), root / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.table({"task_index": [0], "task": ["test"]}), root / "meta/tasks.parquet")

    info = {
        "codebase_version": "v3.0",
        "fps": 25,
        "features": FEATURES,
        "total_episodes": 3,
        "total_frames": 6,
        "total_tasks": 1,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "robot_type": "test",
        "splits": {"train": "0:3"},
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/stats.json").write_text("{}", encoding="utf-8")


def test_repair_normalizes_episode_and_global_indices(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "repaired"
    _make_gapped_dataset(source)

    checker = IntegrityChecker(source, decode_videos=True, timestamp_tolerance_s=1e-4)
    source_report = checker.run()
    assert not source_report["valid"]

    manifest, repaired_report = DatasetRepairer(checker, output).run(
        source_report, timestamp_tolerance_s=1e-4
    )

    assert repaired_report["valid"]
    assert manifest["episode_mapping"] == {"0": 0, "2": 1}
    assert manifest["video_repair"]["output"] == str(output)
    repaired_info = json.loads((output / "meta/info.json").read_text())
    assert repaired_info["total_episodes"] == 2
    assert repaired_info["total_frames"] == 4
    assert repaired_info["splits"] == {"train": "0:2"}

    data = pq.read_table(output / "data/chunk-000/file-000.parquet")
    assert data["episode_index"].to_pylist() == [0, 0, 1, 1]
    assert data["index"].to_pylist() == [0, 1, 2, 3]
    stats = json.loads((output / "meta/stats.json").read_text())
    assert {"action", "observation.state", "episode_index", "index"} <= stats.keys()


def test_repair_refuses_data_metadata_episode_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "repaired"
    _make_gapped_dataset(source, metadata_ids=(0, 1))

    checker = IntegrityChecker(source, decode_videos=True, timestamp_tolerance_s=1e-4)
    report = checker.run()

    with pytest.raises(ValueError, match="Repair refused"):
        DatasetRepairer(checker, output).run(report, timestamp_tolerance_s=1e-4)
    assert not output.exists()


def test_repair_refuses_in_place_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_gapped_dataset(source)
    checker = IntegrityChecker(source, decode_videos=True, timestamp_tolerance_s=1e-4)

    with pytest.raises(ValueError, match="in-place repair is forbidden"):
        DatasetRepairer(checker, source).run(checker.run(), timestamp_tolerance_s=1e-4)
