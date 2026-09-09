#!/usr/bin/env python3
r"""Offline integrity checker for LeRobot v3 datasets.

The checker treats the per-frame data parquet files as the authoritative sample
records, then verifies that ``meta/info.json``, episode metadata parquet files,
and video files describe the same episodes and frames.  It never contacts the
Hugging Face Hub and never modifies the source dataset.

Examples
--------
Fast structural and video-header check::

    python scripts/check_lerobot_dataset_integrity.py \
      --dataset.root /path/to/dataset

Fully decode every referenced video (slower, catches damaged packets)::

    python scripts/check_lerobot_dataset_integrity.py \
      --dataset.root /path/to/dataset \
      --decode-videos \
      --output-json /tmp/dataset_integrity.json

Safely normalize recoverable metadata/index gaps into a new dataset::

    python scripts/check_lerobot_dataset_integrity.py \
      --dataset.root /path/to/dataset \
      --repair-output /path/to/dataset_repaired \
      --output-json /tmp/dataset_repair.json

Repair mode always fully decodes referenced videos, never edits the source,
refuses ambiguous damage, and runs two explicit stages: normalize metadata and
indices first, then invoke ``repair_lerobot_video_gaps.py`` to compact referenced
video ranges. It validates the completed output before publishing it.

Exit status is 0 when no errors are found and 1 when the dataset is invalid.
Warnings do not change the exit status unless ``--fail-on-warnings`` is used.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.io_utils import write_stats
from lerobot.utils.utils import flatten_dict, unflatten_dict

if __package__:
    from .repair_lerobot_video_gaps import repair_video_gaps
else:
    from repair_lerobot_video_gaps import repair_video_gaps


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    message: str


@dataclass
class DataEpisode:
    count: int = 0
    indices: list[int] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    files: set[Path] = field(default_factory=set)


@dataclass(frozen=True)
class VideoRange:
    episode_index: int
    start_frame: int
    end_frame: int
    episode_length: int
    video_key: str


REPAIRABLE_ERROR_CODES = {
    "DATA_EPISODE_IDS_NONCONTIGUOUS",
    "EPISODE_DATASET_RANGE_MISMATCH",
    "GLOBAL_INDEX_NONCONTIGUOUS",
    "METADATA_EPISODE_IDS_NONCONTIGUOUS",
    "METADATA_LENGTH_SUM_MISMATCH",
    "TOTAL_EPISODES_MISMATCH",
    "TOTAL_FRAMES_MISMATCH",
    "VIDEO_EPISODE_LENGTH_MISMATCH",
    "VIDEO_FRAME_GAP",
    "VIDEO_LEADING_GAP",
    "VIDEO_TOTAL_FRAMES_MISMATCH",
}


class IntegrityChecker:
    def __init__(self, root: Path, *, decode_videos: bool, timestamp_tolerance_s: float) -> None:
        self.root = root.expanduser().resolve()
        self.decode_videos = decode_videos
        self.timestamp_tolerance_s = timestamp_tolerance_s
        self.issues: list[Issue] = []
        self.info: dict[str, Any] = {}
        self.data_episodes: dict[int, DataEpisode] = defaultdict(DataEpisode)
        self.data_files: set[Path] = set()
        self.episode_rows: list[dict[str, Any]] = []
        self.referenced_videos: set[Path] = set()
        self.task_indices: set[int] = set()
        self.task_count = 0
        self.total_data_rows = 0

    def error(self, code: str, message: str) -> None:
        self.issues.append(Issue("error", code, message))

    def warning(self, code: str, message: str) -> None:
        self.issues.append(Issue("warning", code, message))

    def run(self) -> dict[str, Any]:
        if not self.root.is_dir():
            self.error("ROOT_MISSING", f"Dataset root does not exist: {self.root}")
            return self.report()

        if not self._load_info_and_required_files():
            return self.report()

        self._check_zero_byte_files()
        self._read_data_parquets()
        self._read_episode_parquets()
        self._check_global_counts_and_indices()
        self._check_episode_correspondence()
        self._check_video_correspondence()
        self._check_unreferenced_files()
        return self.report()

    def report(self) -> dict[str, Any]:
        errors = sum(issue.severity == "error" for issue in self.issues)
        warnings = sum(issue.severity == "warning" for issue in self.issues)
        return {
            "dataset_root": str(self.root),
            "valid": errors == 0,
            "errors": errors,
            "warnings": warnings,
            "summary": {
                "declared_episodes": self.info.get("total_episodes"),
                "declared_frames": self.info.get("total_frames"),
                "actual_episode_metadata_rows": len(self.episode_rows),
                "actual_data_rows": self.total_data_rows,
                "actual_data_episodes": sorted(self.data_episodes),
                "referenced_video_files": len(self.referenced_videos),
            },
            "issues": [asdict(issue) for issue in self.issues],
        }

    def _load_info_and_required_files(self) -> bool:
        required = (
            self.root / "meta/info.json",
            self.root / "meta/stats.json",
            self.root / "meta/tasks.parquet",
        )
        ok = True
        for path in required:
            if not path.is_file():
                self.error("REQUIRED_FILE_MISSING", f"Missing required file: {path}")
                ok = False
            elif path.stat().st_size == 0:
                self.error("REQUIRED_FILE_EMPTY", f"Required file is empty: {path}")
                ok = False
        if not ok:
            return False

        try:
            self.info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.error("INFO_INVALID", f"Cannot parse meta/info.json: {exc}")
            return False

        required_info = {
            "fps",
            "features",
            "total_episodes",
            "total_frames",
            "total_tasks",
            "data_path",
        }
        missing = sorted(required_info - self.info.keys())
        if missing:
            self.error("INFO_FIELDS_MISSING", f"meta/info.json is missing fields: {missing}")
            return False

        try:
            fps = float(self.info["fps"])
            total_episodes = int(self.info["total_episodes"])
            total_frames = int(self.info["total_frames"])
        except (TypeError, ValueError) as exc:
            self.error("INFO_TYPES_INVALID", f"Invalid count or fps in meta/info.json: {exc}")
            return False
        if fps <= 0 or total_episodes < 0 or total_frames < 0:
            self.error(
                "INFO_VALUES_INVALID",
                f"Invalid fps/episode/frame values: fps={fps}, episodes={total_episodes}, frames={total_frames}",
            )
            return False

        tasks_path = self.root / "meta/tasks.parquet"
        try:
            self.task_count = pq.read_metadata(tasks_path).num_rows
            if self.task_count != int(self.info["total_tasks"]):
                self.error(
                    "TOTAL_TASKS_MISMATCH",
                    f"info total_tasks={self.info['total_tasks']}, tasks parquet rows={self.task_count}",
                )
        except Exception as exc:
            self.error("TASKS_PARQUET_INVALID", f"Cannot parse {tasks_path}: {exc}")
        stats_path = self.root / "meta/stats.json"
        try:
            json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.error("STATS_JSON_INVALID", f"Cannot parse {stats_path}: {exc}")
        return True

    def _check_zero_byte_files(self) -> None:
        for directory in ("data", "meta", "videos", "images"):
            base = self.root / directory
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.is_file() and path.stat().st_size == 0:
                    self.error("ZERO_BYTE_FILE", f"Zero-byte file: {path.relative_to(self.root)}")

    @staticmethod
    def _check_schema(
        path: Path,
        schema: pa.Schema,
        reference_path: Path | None,
        reference_schema: pa.Schema | None,
    ) -> tuple[Path, pa.Schema]:
        if reference_schema is not None and not schema.equals(reference_schema, check_metadata=False):
            raise ValueError(f"schema differs from {reference_path}")
        return (
            path if reference_path is None else reference_path,
            schema if reference_schema is None else reference_schema,
        )

    def _read_data_parquets(self) -> None:
        data_root = self.root / "data"
        paths = sorted(data_root.rglob("*.parquet")) if data_root.is_dir() else []
        if not paths:
            self.error("DATA_PARQUETS_MISSING", f"No data parquet files found under {data_root}")
            return

        required_columns = {"episode_index", "frame_index", "timestamp", "index", "task_index"}
        required_columns.update(
            key
            for key, feature in self.info.get("features", {}).items()
            if isinstance(feature, dict) and feature.get("dtype") not in {"video", "image"}
        )
        reference_path: Path | None = None
        reference_schema: pa.Schema | None = None

        for path in paths:
            self.data_files.add(path.resolve())
            try:
                table = pq.read_table(path)
            except Exception as exc:
                self.error("DATA_PARQUET_INVALID", f"Cannot read {path.relative_to(self.root)}: {exc}")
                continue

            try:
                reference_path, reference_schema = self._check_schema(
                    path, table.schema, reference_path, reference_schema
                )
            except ValueError as exc:
                self.error("DATA_SCHEMA_MISMATCH", f"{path.relative_to(self.root)}: {exc}")

            missing = sorted(required_columns - set(table.column_names))
            if missing:
                self.error(
                    "DATA_COLUMNS_MISSING",
                    f"{path.relative_to(self.root)} is missing columns: {missing}",
                )
                continue

            self.total_data_rows += table.num_rows
            episode_values = table["episode_index"].to_pylist()
            frame_values = table["frame_index"].to_pylist()
            timestamp_values = table["timestamp"].to_pylist()
            index_values = table["index"].to_pylist()
            self.task_indices.update(
                int(value) for value in table["task_index"].to_pylist() if value is not None
            )

            for episode, frame, timestamp, index in zip(
                episode_values, frame_values, timestamp_values, index_values, strict=True
            ):
                if None in (episode, frame, timestamp, index):
                    self.error("DATA_NULL_INDEX", f"Null index value in {path.relative_to(self.root)}")
                    continue
                record = self.data_episodes[int(episode)]
                record.count += 1
                record.indices.append(int(index))
                record.frame_indices.append(int(frame))
                record.timestamps.append(float(timestamp))
                record.files.add(path.resolve())

            self._check_finite_feature(path, table, "action")
            self._check_finite_feature(path, table, "observation.state")

    def _check_finite_feature(self, path: Path, table: pa.Table, feature: str) -> None:
        if feature not in table.column_names:
            return
        try:
            values = np.asarray(table[feature].to_pylist(), dtype=np.float64)
        except (TypeError, ValueError):
            self.error(
                "FEATURE_VALUES_INVALID",
                f"Cannot convert {feature} to a numeric array in {path.relative_to(self.root)}",
            )
            return
        if values.size and not np.isfinite(values).all():
            bad_rows = np.flatnonzero(~np.isfinite(values).all(axis=tuple(range(1, values.ndim))))
            self.error(
                "FEATURE_NONFINITE",
                f"{feature} contains NaN/Inf in {path.relative_to(self.root)}; rows={bad_rows[:20].tolist()}",
            )

    def _read_episode_parquets(self) -> None:
        episode_root = self.root / "meta/episodes"
        paths = sorted(episode_root.rglob("*.parquet")) if episode_root.is_dir() else []
        if not paths:
            self.error(
                "EPISODE_PARQUETS_MISSING",
                f"No episode metadata parquet files found under {episode_root}",
            )
            return

        required_columns = {
            "episode_index",
            "length",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
        }
        reference_path: Path | None = None
        reference_schema: pa.Schema | None = None

        for path in paths:
            try:
                table = pq.read_table(path)
            except Exception as exc:
                self.error(
                    "EPISODE_PARQUET_INVALID",
                    f"Cannot read {path.relative_to(self.root)}: {exc}",
                )
                continue

            try:
                reference_path, reference_schema = self._check_schema(
                    path, table.schema, reference_path, reference_schema
                )
            except ValueError as exc:
                self.error("EPISODE_SCHEMA_MISMATCH", f"{path.relative_to(self.root)}: {exc}")

            missing = sorted(required_columns - set(table.column_names))
            if missing:
                self.error(
                    "EPISODE_COLUMNS_MISSING",
                    f"{path.relative_to(self.root)} is missing columns: {missing}",
                )
                continue
            self.episode_rows.extend(table.to_pylist())

    def _check_global_counts_and_indices(self) -> None:
        declared_frames = int(self.info.get("total_frames", -1))
        declared_episodes = int(self.info.get("total_episodes", -1))
        if self.total_data_rows != declared_frames:
            self.error(
                "TOTAL_FRAMES_MISMATCH",
                f"info total_frames={declared_frames}, data parquet rows={self.total_data_rows}",
            )
        if len(self.episode_rows) != declared_episodes:
            self.error(
                "TOTAL_EPISODES_MISMATCH",
                f"info total_episodes={declared_episodes}, episode metadata rows={len(self.episode_rows)}",
            )
        metadata_length_sum = sum(int(row["length"]) for row in self.episode_rows)
        if metadata_length_sum != declared_frames:
            self.error(
                "METADATA_LENGTH_SUM_MISMATCH",
                f"info total_frames={declared_frames}, sum of episode metadata lengths={metadata_length_sum}",
            )
        invalid_tasks = sorted(index for index in self.task_indices if index < 0 or index >= self.task_count)
        if invalid_tasks:
            self.error(
                "TASK_INDEX_OUT_OF_RANGE",
                f"Data contains task_index values outside tasks parquet range 0..{self.task_count - 1}: "
                f"{invalid_tasks}",
            )

        actual_data_episodes = sorted(self.data_episodes)
        expected_episodes = list(range(declared_episodes))
        if actual_data_episodes != expected_episodes:
            missing = sorted(set(expected_episodes) - set(actual_data_episodes))
            unexpected = sorted(set(actual_data_episodes) - set(expected_episodes))
            self.error(
                "DATA_EPISODE_IDS_NONCONTIGUOUS",
                f"Expected episode ids {expected_episodes}; actual={actual_data_episodes}; "
                f"missing={missing}; unexpected={unexpected}",
            )

        metadata_ids = [
            int(row["episode_index"]) for row in self.episode_rows if row.get("episode_index") is not None
        ]
        if sorted(metadata_ids) != expected_episodes:
            self.error(
                "METADATA_EPISODE_IDS_NONCONTIGUOUS",
                f"Expected metadata episode ids {expected_episodes}; actual={sorted(metadata_ids)}",
            )
        if len(metadata_ids) != len(set(metadata_ids)):
            duplicates = sorted({episode for episode in metadata_ids if metadata_ids.count(episode) > 1})
            self.error("METADATA_EPISODE_IDS_DUPLICATE", f"Duplicate metadata episodes: {duplicates}")

        all_indices = [index for record in self.data_episodes.values() for index in record.indices]
        if len(all_indices) != len(set(all_indices)):
            values, counts = np.unique(np.asarray(all_indices, dtype=np.int64), return_counts=True)
            duplicates = values[counts > 1][:20].tolist()
            self.error("GLOBAL_INDEX_DUPLICATE", f"Duplicate global indices (first 20): {duplicates}")
        if sorted(all_indices) != list(range(declared_frames)):
            expected = set(range(declared_frames))
            actual = set(all_indices)
            self.error(
                "GLOBAL_INDEX_NONCONTIGUOUS",
                f"Global index is not 0..{declared_frames - 1}; "
                f"missing(first 20)={sorted(expected - actual)[:20]}, "
                f"unexpected(first 20)={sorted(actual - expected)[:20]}",
            )

    def _check_episode_correspondence(self) -> None:
        fps = float(self.info.get("fps", 0))
        metadata_by_episode: dict[int, dict[str, Any]] = {}
        for row in self.episode_rows:
            if row.get("episode_index") is not None:
                metadata_by_episode[int(row["episode_index"])] = row

        for episode_index, record in sorted(self.data_episodes.items()):
            order = np.argsort(np.asarray(record.frame_indices, dtype=np.int64))
            frame_indices = np.asarray(record.frame_indices, dtype=np.int64)[order]
            indices = np.asarray(record.indices, dtype=np.int64)[order]
            timestamps = np.asarray(record.timestamps, dtype=np.float64)[order]
            expected_frames = np.arange(record.count, dtype=np.int64)
            if not np.array_equal(frame_indices, expected_frames):
                self.error(
                    "FRAME_INDEX_NONCONTIGUOUS",
                    f"Episode {episode_index}: frame_index is not 0..{record.count - 1}",
                )
            if len(set(record.indices)) != record.count:
                self.error("EPISODE_INDEX_DUPLICATE", f"Episode {episode_index}: duplicate global index")
            expected_timestamps = frame_indices / fps
            if timestamps.size and not np.allclose(
                timestamps, expected_timestamps, rtol=0, atol=self.timestamp_tolerance_s
            ):
                max_error = float(np.max(np.abs(timestamps - expected_timestamps)))
                self.error(
                    "TIMESTAMP_MISMATCH",
                    f"Episode {episode_index}: timestamp differs from frame_index/fps; max_error={max_error:.6g}s",
                )

            row = metadata_by_episode.get(episode_index)
            if row is None:
                self.error("EPISODE_METADATA_MISSING", f"Episode {episode_index}: metadata row missing")
                continue
            if int(row["length"]) != record.count:
                self.error(
                    "EPISODE_LENGTH_MISMATCH",
                    f"Episode {episode_index}: metadata length={row['length']}, data rows={record.count}",
                )
            expected_from = int(indices.min()) if indices.size else 0
            expected_to = int(indices.max()) + 1 if indices.size else 0
            if int(row["dataset_from_index"]) != expected_from or int(row["dataset_to_index"]) != expected_to:
                self.error(
                    "EPISODE_DATASET_RANGE_MISMATCH",
                    f"Episode {episode_index}: metadata range="
                    f"[{row['dataset_from_index']}, {row['dataset_to_index']}), "
                    f"actual=[{expected_from}, {expected_to})",
                )

            data_path = self._format_path(
                self.info["data_path"],
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
            if data_path is None:
                continue
            if not data_path.is_file():
                self.error("REFERENCED_DATA_MISSING", f"Episode {episode_index}: missing {data_path}")
            elif data_path.resolve() not in record.files:
                actual = sorted(str(path.relative_to(self.root)) for path in record.files)
                self.error(
                    "DATA_FILE_REFERENCE_MISMATCH",
                    f"Episode {episode_index}: metadata points to {data_path.relative_to(self.root)}, "
                    f"but its rows are in {actual}",
                )

    def _video_keys(self) -> list[str]:
        features = self.info.get("features", {})
        if not isinstance(features, dict):
            self.error("FEATURES_INVALID", "meta/info.json features must be a mapping")
            return []
        return [
            key
            for key, feature in features.items()
            if isinstance(feature, dict) and feature.get("dtype") == "video"
        ]

    def _check_video_correspondence(self) -> None:
        video_keys = self._video_keys()
        if not video_keys:
            return
        template = self.info.get("video_path")
        if not template:
            self.error("VIDEO_PATH_MISSING", "Video features exist but info.video_path is missing")
            return

        fps = float(self.info["fps"])
        groups: dict[Path, list[VideoRange]] = defaultdict(list)
        for row in self.episode_rows:
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            for key in video_keys:
                prefix = f"videos/{key}"
                required = (
                    f"{prefix}/chunk_index",
                    f"{prefix}/file_index",
                    f"{prefix}/from_timestamp",
                    f"{prefix}/to_timestamp",
                )
                missing = [column for column in required if column not in row or row[column] is None]
                if missing:
                    self.error(
                        "VIDEO_METADATA_MISSING",
                        f"Episode {episode_index}, {key}: missing columns/values {missing}",
                    )
                    continue
                path = self._format_path(
                    template,
                    video_key=key,
                    chunk_index=int(row[f"{prefix}/chunk_index"]),
                    file_index=int(row[f"{prefix}/file_index"]),
                )
                if path is None:
                    continue
                self.referenced_videos.add(path.resolve())
                start = round(float(row[f"{prefix}/from_timestamp"]) * fps)
                end = round(float(row[f"{prefix}/to_timestamp"]) * fps)
                if end <= start:
                    self.error(
                        "VIDEO_RANGE_INVALID",
                        f"Episode {episode_index}, {key}: invalid frame range [{start}, {end})",
                    )
                if end - start != length:
                    self.error(
                        "VIDEO_EPISODE_LENGTH_MISMATCH",
                        f"Episode {episode_index}, {key}: video range has {end - start} frames, "
                        f"metadata length={length}",
                    )
                groups[path].append(VideoRange(episode_index, start, end, length, key))

        for path, ranges in sorted(groups.items(), key=lambda item: str(item[0])):
            if not path.is_file():
                self.error("REFERENCED_VIDEO_MISSING", f"Missing referenced video: {path}")
                continue
            if path.stat().st_size == 0:
                self.error("REFERENCED_VIDEO_EMPTY", f"Referenced video is empty: {path}")
                continue
            ranges.sort(key=lambda item: (item.start_frame, item.episode_index))
            if ranges[0].start_frame != 0:
                self.error(
                    "VIDEO_LEADING_GAP",
                    f"{path.relative_to(self.root)}: first referenced frame is {ranges[0].start_frame}, not 0",
                )
            for previous, current in zip(ranges, ranges[1:], strict=False):
                if current.start_frame > previous.end_frame:
                    self.error(
                        "VIDEO_FRAME_GAP",
                        f"{path.relative_to(self.root)}: {current.start_frame - previous.end_frame} "
                        f"unreferenced frames between episodes {previous.episode_index} and {current.episode_index}",
                    )
                elif current.start_frame < previous.end_frame:
                    self.error(
                        "VIDEO_RANGE_OVERLAP",
                        f"{path.relative_to(self.root)}: ranges overlap between episodes "
                        f"{previous.episode_index} and {current.episode_index}",
                    )
            self._inspect_video(path, ranges[-1].end_frame, fps, ranges[0].video_key)

    def _inspect_video(self, path: Path, expected_frames: int, expected_fps: float, video_key: str) -> None:
        try:
            with av.open(str(path)) as container:
                if not container.streams.video:
                    self.error("VIDEO_STREAM_MISSING", f"No video stream in {path}")
                    return
                stream = container.streams.video[0]
                header_frames = int(stream.frames or 0)
                actual_fps = float(stream.average_rate) if stream.average_rate else None
                shape = self.info["features"].get(video_key, {}).get("shape")
                if isinstance(shape, list) and len(shape) >= 2:
                    expected_height, expected_width = int(shape[0]), int(shape[1])
                    if stream.height != expected_height or stream.width != expected_width:
                        self.error(
                            "VIDEO_SHAPE_MISMATCH",
                            f"{path.relative_to(self.root)}: stream shape="
                            f"({stream.height}, {stream.width}), info shape="
                            f"({expected_height}, {expected_width})",
                        )
                if actual_fps is not None and not math.isclose(actual_fps, expected_fps, abs_tol=1e-3):
                    self.error(
                        "VIDEO_FPS_MISMATCH",
                        f"{path.relative_to(self.root)}: stream fps={actual_fps}, info fps={expected_fps}",
                    )
                actual_frames = header_frames
                if self.decode_videos:
                    actual_frames = sum(1 for _ in container.decode(stream))
                    if header_frames and actual_frames != header_frames:
                        self.error(
                            "VIDEO_DECODE_COUNT_MISMATCH",
                            f"{path.relative_to(self.root)}: header={header_frames}, decoded={actual_frames}",
                        )
                elif actual_frames == 0 and stream.duration is not None:
                    actual_frames = round(float(stream.duration * stream.time_base) * expected_fps)
                    self.warning(
                        "VIDEO_FRAME_COUNT_ESTIMATED",
                        f"{path.relative_to(self.root)}: stream header has no frame count; "
                        f"estimated {actual_frames} from duration",
                    )
                if actual_frames == 0:
                    self.warning(
                        "VIDEO_FRAME_COUNT_UNKNOWN",
                        f"{path.relative_to(self.root)}: cannot determine frame count without --decode-videos",
                    )
                elif actual_frames != expected_frames:
                    self.error(
                        "VIDEO_TOTAL_FRAMES_MISMATCH",
                        f"{path.relative_to(self.root)}: video has {actual_frames} frames, "
                        f"metadata covers {expected_frames}",
                    )
        except Exception as exc:
            self.error("VIDEO_OPEN_OR_DECODE_FAILED", f"Cannot inspect {path}: {exc}")

    def _check_unreferenced_files(self) -> None:
        video_root = self.root / "videos"
        if video_root.is_dir():
            actual_videos = {path.resolve() for path in video_root.rglob("*.mp4")}
            for path in sorted(actual_videos - self.referenced_videos):
                self.warning(
                    "UNREFERENCED_VIDEO",
                    f"Video is not referenced by episode metadata: {path.relative_to(self.root)}",
                )

        referenced_data: set[Path] = set()
        template = self.info.get("data_path")
        if template:
            for row in self.episode_rows:
                path = self._format_path(
                    template,
                    chunk_index=int(row["data/chunk_index"]),
                    file_index=int(row["data/file_index"]),
                )
                if path is not None:
                    referenced_data.add(path.resolve())
        for path in sorted(self.data_files - referenced_data):
            self.warning(
                "UNREFERENCED_DATA_PARQUET",
                f"Data parquet is not referenced by episode metadata: {path.relative_to(self.root)}",
            )

        images_root = self.root / "images"
        raw_images = list(images_root.rglob("*")) if images_root.is_dir() else []
        raw_image_files = [path for path in raw_images if path.is_file()]
        if raw_image_files and self._video_keys():
            episode_dirs = sorted(
                {
                    str(path.parent.relative_to(self.root))
                    for path in raw_image_files
                    if path.parent.name.startswith("episode-")
                }
            )
            self.warning(
                "RAW_IMAGES_REMAIN",
                f"Finalized video dataset still contains {len(raw_image_files)} raw image files; "
                f"episode directories={episode_dirs[:20]}",
            )

    def _format_path(self, template: str, **values: Any) -> Path | None:
        try:
            return self.root / template.format(**values)
        except (KeyError, ValueError) as exc:
            self.error("PATH_TEMPLATE_INVALID", f"Cannot format path template {template!r}: {exc}")
            return None


def _numpy_stats_from_row(row: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    flat = {
        key.removeprefix("stats/"): np.atleast_1d(np.asarray(value))
        for key, value in row.items()
        if key.startswith("stats/") and value is not None
    }
    return unflatten_dict(flat)


def _to_arrow_value(value: Any) -> Any:
    return value.tolist() if isinstance(value, np.ndarray) else value


class DatasetRepairer:
    """Conservatively normalize an internally consistent subset into a new dataset."""

    def __init__(self, checker: IntegrityChecker, output: Path) -> None:
        self.checker = checker
        self.source = checker.root
        self.output = output.expanduser().resolve()
        self.metadata_stage = self.output.with_name(f".{self.output.name}.metadata-fixed")
        self.candidate = self.output.with_name(f".{self.output.name}.validated-candidate")
        self.info = checker.info
        self.old_ids = sorted(checker.data_episodes)
        self.mapping = {old: new for new, old in enumerate(self.old_ids)}
        self.metadata_by_old = {int(row["episode_index"]): row for row in checker.episode_rows}
        self.new_ranges: dict[int, tuple[int, int]] = {}

    def _validate_preconditions(self, report: dict[str, Any]) -> None:
        if self.output == self.source:
            raise ValueError("--repair-output must differ from --dataset.root; in-place repair is forbidden")
        work_paths = (self.output, self.metadata_stage, self.candidate)
        if any(path.exists() for path in work_paths):
            raise FileExistsError(f"Refusing to overwrite existing path: {work_paths}")

        blocking = sorted(
            {
                issue["code"]
                for issue in report["issues"]
                if issue["severity"] == "error" and issue["code"] not in REPAIRABLE_ERROR_CODES
            }
        )
        if blocking:
            raise ValueError(f"Repair refused because damage is ambiguous or destructive: {blocking}")

        metadata_ids = [int(row["episode_index"]) for row in self.checker.episode_rows]
        if len(metadata_ids) != len(set(metadata_ids)):
            raise ValueError("Repair refused: duplicate episode metadata rows")
        if set(metadata_ids) != set(self.old_ids):
            raise ValueError(
                "Repair refused: data parquet and episode metadata do not describe the same episodes; "
                f"data={self.old_ids}, metadata={sorted(metadata_ids)}"
            )
        if not self.old_ids:
            raise ValueError("Repair refused: no complete episodes were found")

        cursor = 0
        for old_id in self.old_ids:
            record = self.checker.data_episodes[old_id]
            row = self.metadata_by_old[old_id]
            if record.count != int(row["length"]):
                raise ValueError(
                    f"Repair refused: episode {old_id} has {record.count} data rows but "
                    f"metadata length {row['length']}"
                )
            if len(record.files) != 1:
                raise ValueError(f"Repair refused: episode {old_id} spans {len(record.files)} data files")
            frames = sorted(record.frame_indices)
            if frames != list(range(record.count)):
                raise ValueError(f"Repair refused: episode {old_id} frame_index is not contiguous")
            self.new_ranges[old_id] = (cursor, cursor + record.count)
            cursor += record.count

    def _rewrite_data(self) -> None:
        for source_path in sorted(self.source.joinpath("data").rglob("*.parquet")):
            table = pq.read_table(source_path)
            old_episode_ids = [int(value) for value in table["episode_index"].to_pylist()]
            frame_indices = [int(value) for value in table["frame_index"].to_pylist()]
            new_episode_ids = [self.mapping[old_id] for old_id in old_episode_ids]
            new_indices = [
                self.new_ranges[old_id][0] + frame
                for old_id, frame in zip(old_episode_ids, frame_indices, strict=True)
            ]

            ep_pos = table.schema.get_field_index("episode_index")
            index_pos = table.schema.get_field_index("index")
            table = table.set_column(
                ep_pos,
                "episode_index",
                pa.array(new_episode_ids, type=table.schema.field(ep_pos).type),
            )
            table = table.set_column(
                index_pos,
                "index",
                pa.array(new_indices, type=table.schema.field(index_pos).type),
            )
            destination = self.metadata_stage / source_path.relative_to(self.source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination)

    def _link_referenced_videos(self) -> None:
        """Expose source videos to stage 2 without duplicating large files when possible."""

        for source_path in sorted(self.checker.referenced_videos):
            destination = self.metadata_stage / source_path.relative_to(self.source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                destination.hardlink_to(source_path)
            except OSError:
                shutil.copy2(source_path, destination)

    def _rewrite_episode_metadata_and_stats(self) -> None:
        all_episode_stats: list[dict[str, dict[str, np.ndarray]]] = []
        for source_path in sorted(self.source.joinpath("meta/episodes").rglob("*.parquet")):
            source_table = pq.read_table(source_path)
            repaired_rows: list[dict[str, Any]] = []
            for row in source_table.to_pylist():
                old_id = int(row["episode_index"])
                if old_id not in self.mapping:
                    continue
                new_id = self.mapping[old_id]
                start, end = self.new_ranges[old_id]
                row["episode_index"] = new_id
                row["dataset_from_index"] = start
                row["dataset_to_index"] = end

                episode_stats = _numpy_stats_from_row(row)
                length = end - start
                episode_stats["episode_index"] = get_feature_stats(
                    np.full((length, 1), new_id, dtype=np.int64), axis=0, keepdims=False
                )
                episode_stats["index"] = get_feature_stats(
                    np.arange(start, end, dtype=np.int64).reshape(-1, 1), axis=0, keepdims=False
                )
                for stat_key, value in flatten_dict({"stats": episode_stats}).items():
                    if stat_key in row:
                        row[stat_key] = _to_arrow_value(value)
                all_episode_stats.append(episode_stats)
                repaired_rows.append(row)

            destination = self.metadata_stage / source_path.relative_to(self.source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            repaired_table = pa.Table.from_pylist(repaired_rows, schema=source_table.schema)
            pq.write_table(repaired_table, destination)

        write_stats(aggregate_stats(all_episode_stats), self.metadata_stage)

    def _write_info_and_tasks(self) -> dict[str, Any]:
        tasks_source = self.source / "meta/tasks.parquet"
        tasks_destination = self.metadata_stage / "meta/tasks.parquet"
        tasks_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tasks_source, tasks_destination)

        repaired_info = json.loads(json.dumps(self.info))
        repaired_info["total_episodes"] = len(self.old_ids)
        repaired_info["total_frames"] = sum(self.checker.data_episodes[old].count for old in self.old_ids)
        repaired_info["splits"] = {"train": f"0:{len(self.old_ids)}"}
        (self.metadata_stage / "meta/info.json").write_text(
            json.dumps(repaired_info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        manifest = {
            "source": str(self.source),
            "output": str(self.output),
            "episode_mapping": {str(old): new for old, new in self.mapping.items()},
            "kept_episodes": len(self.old_ids),
            "kept_frames": repaired_info["total_frames"],
            "policy": "data parquet and matching episode metadata are authoritative",
            "workflow": [
                "normalize metadata and global indices",
                "repair video gaps with repair_lerobot_video_gaps.py",
                "fully decode and validate the final dataset",
            ],
            "dropped": "unreferenced videos and unfinished raw images are not copied",
        }
        return manifest

    def run(
        self, source_report: dict[str, Any], *, timestamp_tolerance_s: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._validate_preconditions(source_report)
        try:
            self.metadata_stage.mkdir(parents=True)
            self._rewrite_data()
            self._rewrite_episode_metadata_and_stats()
            self._link_referenced_videos()
            manifest = self._write_info_and_tasks()

            video_report = repair_video_gaps(self.metadata_stage, self.candidate)
            video_report["output"] = str(self.output)
            manifest["video_repair"] = video_report
            (self.candidate / "repair_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            repaired_checker = IntegrityChecker(
                self.candidate,
                decode_videos=True,
                timestamp_tolerance_s=timestamp_tolerance_s,
            )
            repaired_report = repaired_checker.run()
            if not repaired_report["valid"]:
                raise RuntimeError(
                    f"Repaired candidate dataset failed validation at {self.candidate}: "
                    f"{repaired_report['errors']} errors"
                )
            self.candidate.rename(self.output)
            repaired_report["dataset_root"] = str(self.output)
            return manifest, repaired_report
        except BaseException:
            shutil.rmtree(self.candidate, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(self.metadata_stage, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset.root",
        "--dataset-root",
        dest="dataset_root",
        type=Path,
        required=True,
        help="Local LeRobot dataset root.",
    )
    parser.add_argument(
        "--decode-videos",
        action="store_true",
        help="Decode every frame of every referenced video instead of trusting stream headers.",
    )
    parser.add_argument(
        "--timestamp-tolerance-s",
        type=float,
        default=1e-4,
        help="Tolerance for timestamp == frame_index / fps checks (default: 1e-4).",
    )
    parser.add_argument("--output-json", type=Path, help="Optional path for a machine-readable report.")
    parser.add_argument(
        "--repair-output",
        type=Path,
        help=(
            "Write a conservatively repaired dataset to this new path. Forces full video decode, "
            "refuses unsafe damage, and never modifies the source."
        ),
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Return exit status 1 when warnings are found even if there are no errors.",
    )
    return parser.parse_args()


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"Dataset: {report['dataset_root']}")
    print(
        "Declared/actual: "
        f"episodes={summary['declared_episodes']}/{summary['actual_episode_metadata_rows']}, "
        f"frames={summary['declared_frames']}/{summary['actual_data_rows']}, "
        f"videos={summary['referenced_video_files']}"
    )
    for issue in report["issues"]:
        print(f"[{issue['severity'].upper()}] {issue['code']}: {issue['message']}")
    status = "VALID" if report["valid"] else "INVALID"
    print(f"Result: {status} ({report['errors']} errors, {report['warnings']} warnings)")


def main() -> None:
    args = parse_args()
    if args.timestamp_tolerance_s < 0:
        raise ValueError("--timestamp-tolerance-s must be non-negative")
    checker = IntegrityChecker(
        args.dataset_root,
        decode_videos=args.decode_videos or args.repair_output is not None,
        timestamp_tolerance_s=args.timestamp_tolerance_s,
    )
    report = checker.run()
    print_report(report)
    if args.repair_output:
        repairer = DatasetRepairer(checker, args.repair_output)
        manifest, repaired_report = repairer.run(
            report,
            timestamp_tolerance_s=args.timestamp_tolerance_s,
        )
        print("\nRepaired dataset validation:")
        print_report(repaired_report)
        report["repair"] = {"manifest": manifest, "validation": repaired_report}
    if args.output_json:
        args.output_json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output_json.expanduser().resolve().write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    failed = (
        False
        if args.repair_output and report.get("repair", {}).get("validation", {}).get("valid")
        else report["errors"] > 0 or (args.fail_on_warnings and report["warnings"] > 0)
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
