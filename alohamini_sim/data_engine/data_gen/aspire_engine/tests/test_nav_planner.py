"""Standalone tests for the pure-numpy navigation planner."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ASPIRE_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASPIRE_ENGINE_DIR))

from nav_planner import OccupancyGrid, plan_path


def _path_length(path):
    return sum(float(np.linalg.norm(np.asarray(b) - np.asarray(a))) for a, b in zip(path[:-1], path[1:]))


def _inside_box(point, center, half):
    point = np.asarray(point, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(half, dtype=np.float64)
    return bool(np.all(np.abs(point - center) <= half + 1e-9))


def test_plan_around_single_box():
    center = np.array([0.50, 0.50])
    half = np.array([0.10, 0.20])
    robot_radius = 0.05
    obstacles = [{"center": center, "half": half}]

    path = plan_path(obstacles, (0.10, 0.50), (0.90, 0.50), robot_radius, (0.0, 1.0, 0.0, 1.0))

    assert path is not None, "expected a path around the obstacle"
    inflated_half = half + robot_radius
    for waypoint in path:
        assert not _inside_box(waypoint, center, inflated_half), f"waypoint inside inflated box: {waypoint}"
    length = _path_length(path)
    assert 0.80 < length < 1.80, f"unexpected path length {length:.3f}"


def test_goal_inside_obstacle_retargets():
    center = np.array([0.50, 0.50])
    half = np.array([0.10, 0.20])
    robot_radius = 0.05
    obstacles = [{"center": center, "half": half}]

    path = plan_path(obstacles, (0.10, 0.50), (0.50, 0.50), robot_radius, (0.0, 1.0, 0.0, 1.0))

    assert path is not None, "expected retargeted path"
    final = np.asarray(path[-1])
    assert not _inside_box(final, center, half + robot_radius), f"final waypoint still inside inflated box: {final}"
    assert float(np.linalg.norm(final - center)) < 0.35, f"retargeted endpoint too far from blocked goal: {final}"


def test_straight_line_shortcuts_to_endpoints():
    path = plan_path([], (0.10, 0.10), (0.14, 0.10), 0.05, (0.0, 1.0, 0.0, 1.0))

    assert path is not None, "expected a direct path"
    assert 1 <= len(path) <= 3, f"straight-line path should be endpoint-like, got {len(path)} waypoints"
    assert math.isclose(_path_length(path), 0.04, abs_tol=0.04), f"unexpected direct path length: {path}"


def main():
    tests = [
        test_plan_around_single_box,
        test_goal_inside_obstacle_retargets,
        test_straight_line_shortcuts_to_endpoints,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - standalone runner reports all failures.
            failed += 1
            print(f"FAILED {test.__name__}: {exc}")
        else:
            passed += 1
            print(f"PASSED {test.__name__}")
    print(f"PASSED {passed} FAILED {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
