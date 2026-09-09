#!/usr/bin/env python

"""Repair LeRobot v3 datasets whose video files contain unindexed frame gaps.

The data parquet files are treated as authoritative.  For every episode, the
script keeps exactly ``length`` frames beginning at the episode's recorded
``from_timestamp``.  Video files containing gaps or extra frames are repacked,
and episode video timestamps are rewritten to match the clean output.

The source dataset is never modified.  Unfinished raw ``images/`` leftovers
are intentionally not copied to the repaired dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import av
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.configs import encoder_config_from_video_info
from lerobot.datasets.dataset_tools import _keep_episodes_from_video_with_av


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def video_frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # Do not trust the container header here: the purpose of this script is
        # to repair datasets whose physical frames may disagree with metadata.
        return sum(1 for _ in container.decode(stream))


def repair_video_gaps(source: Path, output: Path) -> dict[str, int | str]:
    """Compact referenced episode ranges and return a machine-readable summary."""

    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    staging = output.with_name(f".{output.name}.incomplete")

    if output.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {output} or {staging}")
    if not (source / "meta/info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {source}")

    info = json.loads((source / "meta/info.json").read_text())
    fps = float(info["fps"])
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    video_path_template = info.get("video_path")
    if video_keys and not video_path_template:
        raise ValueError("info.json declares video features but has no video_path template")

    episode_tables: list[tuple[Path, pa.Schema, list[dict]]] = []
    rows_by_episode: dict[int, dict] = {}
    for path in sorted((source / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        rows = table.to_pylist()
        episode_tables.append((path.relative_to(source), table.schema, rows))
        for row in rows:
            episode_index = int(row["episode_index"])
            if episode_index in rows_by_episode:
                raise ValueError(f"Duplicate episode_index {episode_index}")
            rows_by_episode[episode_index] = row

    if len(rows_by_episode) != int(info["total_episodes"]):
        raise ValueError(
            f"Episode metadata count {len(rows_by_episode)} != info total_episodes {info['total_episodes']}"
        )
    if sum(int(row["length"]) for row in rows_by_episode.values()) != int(info["total_frames"]):
        raise ValueError("Episode lengths do not add up to info total_frames")

    groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows_by_episode.values():
        for key in video_keys:
            prefix = f"videos/{key}"
            groups[(key, int(row[f"{prefix}/chunk_index"]), int(row[f"{prefix}/file_index"]))].append(row)

    try:
        (staging / "meta").mkdir(parents=True)
        shutil.copy2(source / "meta/info.json", staging / "meta/info.json")
        shutil.copy2(source / "meta/stats.json", staging / "meta/stats.json")
        shutil.copy2(source / "meta/tasks.parquet", staging / "meta/tasks.parquet")
        shutil.copytree(source / "data", staging / "data")

        repaired_files = 0
        copied_files = 0
        removed_frames = 0

        for (key, chunk_index, file_index), rows in sorted(groups.items()):
            rows.sort(key=lambda row: float(row[f"videos/{key}/from_timestamp"]))
            source_path = source / video_path_template.format(
                video_key=key, chunk_index=chunk_index, file_index=file_index
            )
            output_path = staging / video_path_template.format(
                video_key=key, chunk_index=chunk_index, file_index=file_index
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            source_ranges: list[tuple[int, int]] = []
            target_frame = 0
            previous_end = 0
            for row in rows:
                prefix = f"videos/{key}"
                source_start = round(float(row[f"{prefix}/from_timestamp"]) * fps)
                source_end = source_start + int(row["length"])
                if source_start < 0 or source_start < previous_end:
                    raise ValueError(
                        f"{source_path}: invalid or overlapping source range "
                        f"[{source_start}, {source_end}) after frame {previous_end}"
                    )
                source_ranges.append((source_start, source_end))
                previous_end = source_end
                row[f"{prefix}/from_timestamp"] = target_frame / fps
                target_frame += int(row["length"])
                row[f"{prefix}/to_timestamp"] = target_frame / fps

            actual_frames = video_frame_count(source_path)
            if source_ranges[-1][1] > actual_frames:
                raise ValueError(
                    f"{source_path}: required frame {source_ranges[-1][1]}, but video has {actual_frames} frames"
                )

            contiguous = source_ranges == [
                (
                    sum(end - start for start, end in source_ranges[:i]),
                    sum(end - start for start, end in source_ranges[: i + 1]),
                )
                for i in range(len(source_ranges))
            ]
            if contiguous and actual_frames == target_frame:
                shutil.copy2(source_path, output_path)
                copied_files += 1
                print(f"COPY   {key} file-{file_index:03d}: {actual_frames} frames")
            else:
                encoder = encoder_config_from_video_info(info["features"][key].get("info"))
                _keep_episodes_from_video_with_av(source_path, output_path, source_ranges, fps, encoder)
                clean_frames = video_frame_count(output_path)
                if clean_frames != target_frame:
                    raise ValueError(f"{output_path}: encoded {clean_frames} frames, expected {target_frame}")
                repaired_files += 1
                removed_frames += actual_frames - target_frame
                print(f"REPACK {key} file-{file_index:03d}: {actual_frames} -> {clean_frames} frames")

        for relative_path, schema, rows in episode_tables:
            path = staging / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)

        staging.rename(output)
        return {
            "output": str(output),
            "episodes": len(rows_by_episode),
            "frames": int(info["total_frames"]),
            "repacked_files": repaired_files,
            "copied_files": copied_files,
            "removed_video_frames": removed_frames,
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    report = repair_video_gaps(args.source, args.output)
    print(
        f"DONE: {report['output']}\n"
        f"episodes={report['episodes']}, frames={report['frames']}, "
        f"repacked_files={report['repacked_files']}, copied_files={report['copied_files']}, "
        f"removed_video_frames={report['removed_video_frames']}"
    )


if __name__ == "__main__":
    main()
