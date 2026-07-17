"""Pure-numpy 2D navigation planner for mobile-base station routing."""

from __future__ import annotations

from collections import deque
import heapq
import math

import numpy as np


class OccupancyGrid:
    """Axis-aligned XY occupancy grid with origin at the lower-left world corner."""

    def __init__(self, origin_xy, size_xy, resolution=0.02):
        self.origin_xy = np.asarray(origin_xy, dtype=np.float64)
        self.size_xy = np.asarray(size_xy, dtype=np.float64)
        self.resolution = float(resolution)
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if np.any(self.size_xy <= 0.0):
            raise ValueError("size_xy must be positive")
        self.nx = int(math.ceil(float(self.size_xy[0]) / self.resolution))
        self.ny = int(math.ceil(float(self.size_xy[1]) / self.resolution))
        self.occupied = np.zeros((self.ny, self.nx), dtype=bool)

    def world_to_cell(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        ij = np.floor((xy[:2] - self.origin_xy) / self.resolution).astype(np.int64)
        return int(ij[0]), int(ij[1])

    def cell_to_world(self, ij):
        ij = np.asarray(ij, dtype=np.float64)
        return self.origin_xy + (ij[:2] + 0.5) * self.resolution

    def in_bounds(self, ij):
        i, j = int(ij[0]), int(ij[1])
        return 0 <= i < self.nx and 0 <= j < self.ny

    def is_occupied(self, ij):
        if not self.in_bounds(ij):
            return True
        i, j = int(ij[0]), int(ij[1])
        return bool(self.occupied[j, i])

    def add_box(self, center_xy, half_xy, margin=0.0):
        center_xy = np.asarray(center_xy, dtype=np.float64)
        half_xy = np.asarray(half_xy, dtype=np.float64) + float(margin)
        cell_half = 0.5 * self.resolution
        lo = center_xy[:2] - half_xy[:2] - cell_half
        hi = center_xy[:2] + half_xy[:2] + cell_half
        ij0 = np.floor((lo - self.origin_xy) / self.resolution).astype(np.int64)
        ij1 = np.floor((hi - self.origin_xy) / self.resolution).astype(np.int64)
        i0 = max(0, int(ij0[0]))
        j0 = max(0, int(ij0[1]))
        i1 = min(self.nx - 1, int(ij1[0]))
        j1 = min(self.ny - 1, int(ij1[1]))
        if i0 <= i1 and j0 <= j1:
            self.occupied[j0 : j1 + 1, i0 : i1 + 1] = True
        return self

    def inflate(self, radius):
        radius = float(radius)
        if radius <= 0.0 or not np.any(self.occupied):
            return self

        r_cells = int(math.ceil(radius / self.resolution))
        if r_cells <= 0:
            return self

        src = self.occupied.copy()
        inflated = src.copy()
        for dj in range(-r_cells, r_cells + 1):
            for di in range(-r_cells, r_cells + 1):
                if (di * self.resolution) ** 2 + (dj * self.resolution) ** 2 > radius**2 + 1e-12:
                    continue
                src_j0 = max(0, -dj)
                src_j1 = min(self.ny, self.ny - dj)
                src_i0 = max(0, -di)
                src_i1 = min(self.nx, self.nx - di)
                dst_j0 = src_j0 + dj
                dst_j1 = src_j1 + dj
                dst_i0 = src_i0 + di
                dst_i1 = src_i1 + di
                inflated[dst_j0:dst_j1, dst_i0:dst_i1] |= src[src_j0:src_j1, src_i0:src_i1]
        self.occupied = inflated
        return self


def _nearest_free_cell(grid, ij):
    if grid.in_bounds(ij) and not grid.is_occupied(ij):
        return int(ij[0]), int(ij[1])

    start_i = int(np.clip(int(ij[0]), 0, grid.nx - 1))
    start_j = int(np.clip(int(ij[1]), 0, grid.ny - 1))
    queue = deque([(start_i, start_j)])
    seen = np.zeros((grid.ny, grid.nx), dtype=bool)
    seen[start_j, start_i] = True
    neighbors = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ]

    while queue:
        cur = queue.popleft()
        if not grid.is_occupied(cur):
            return cur
        for di, dj in neighbors:
            nxt = (cur[0] + di, cur[1] + dj)
            if not grid.in_bounds(nxt):
                continue
            if seen[nxt[1], nxt[0]]:
                continue
            seen[nxt[1], nxt[0]] = True
            queue.append(nxt)
    return None


def _octile(a, b):
    dx = abs(int(a[0]) - int(b[0]))
    dy = abs(int(a[1]) - int(b[1]))
    return (dx + dy) + (math.sqrt(2.0) - 2.0) * min(dx, dy)


def _reconstruct(parent, goal):
    cells = [goal]
    cur = goal
    while cur in parent:
        cur = parent[cur]
        cells.append(cur)
    cells.reverse()
    return cells


def astar(grid, start_xy, goal_xy):
    """Return a world-coordinate cell-center path, or None if no route exists."""

    start = grid.world_to_cell(start_xy)
    goal = grid.world_to_cell(goal_xy)
    if not grid.in_bounds(start):
        return None
    if grid.is_occupied(start):
        start = _nearest_free_cell(grid, start)
        if start is None:
            return None

    goal = _nearest_free_cell(grid, goal)
    if goal is None:
        return None
    if start == goal:
        p = grid.cell_to_world(start)
        return [(float(p[0]), float(p[1]))]

    neighbors = [
        (-1, -1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (0, 1, 1.0),
        (1, 1, math.sqrt(2.0)),
    ]
    g_score = np.full((grid.ny, grid.nx), np.inf, dtype=np.float64)
    closed = np.zeros((grid.ny, grid.nx), dtype=bool)
    parent = {}
    g_score[start[1], start[0]] = 0.0
    counter = 0
    open_heap = [(_octile(start, goal), 0.0, counter, start)]

    while open_heap:
        _, cur_g, _, cur = heapq.heappop(open_heap)
        if closed[cur[1], cur[0]]:
            continue
        if cur == goal:
            cells = _reconstruct(parent, goal)
            return [
                (float(grid.cell_to_world(c)[0]), float(grid.cell_to_world(c)[1]))
                for c in cells
            ]
        closed[cur[1], cur[0]] = True

        for di, dj, step_cost in neighbors:
            nxt = (cur[0] + di, cur[1] + dj)
            if not grid.in_bounds(nxt) or closed[nxt[1], nxt[0]] or grid.is_occupied(nxt):
                continue
            if di != 0 and dj != 0:
                if grid.is_occupied((cur[0] + di, cur[1])) or grid.is_occupied((cur[0], cur[1] + dj)):
                    continue
            nxt_g = cur_g + step_cost
            if nxt_g >= g_score[nxt[1], nxt[0]]:
                continue
            parent[nxt] = cur
            g_score[nxt[1], nxt[0]] = nxt_g
            counter += 1
            heapq.heappush(open_heap, (nxt_g + _octile(nxt, goal), nxt_g, counter, nxt))

    return None


def _supercover_cells(a, b):
    x0, y0 = int(a[0]), int(a[1])
    x1, y1 = int(b[0]), int(b[1])
    dx = x1 - x0
    dy = y1 - y0
    nx = abs(dx)
    ny = abs(dy)
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    x, y = x0, y0
    cells = [(x, y)]
    ix = iy = 0

    while ix < nx or iy < ny:
        decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
        if decision == 0:
            if ix < nx:
                cells.append((x + sx, y))
            if iy < ny:
                cells.append((x, y + sy))
            x += sx
            y += sy
            ix += 1
            iy += 1
        elif decision < 0:
            x += sx
            ix += 1
        else:
            y += sy
            iy += 1
        cells.append((x, y))

    out = []
    seen = set()
    for cell in cells:
        if cell not in seen:
            out.append(cell)
            seen.add(cell)
    return out


def _line_is_free(grid, p0, p1):
    c0 = grid.world_to_cell(p0)
    c1 = grid.world_to_cell(p1)
    for cell in _supercover_cells(c0, c1):
        if grid.is_occupied(cell):
            return False
    return True


def _dedupe(points, eps=1e-9):
    out = []
    last = None
    for point in points:
        p = np.asarray(point, dtype=np.float64)
        if last is None or float(np.linalg.norm(p - last)) > eps:
            out.append((float(p[0]), float(p[1])))
            last = p
    return out


def _resample_polyline(points, spacing=0.05):
    if len(points) <= 2:
        return points
    out = [points[0]]
    for p0, p1 in zip(points[:-1], points[1:]):
        a = np.asarray(p0, dtype=np.float64)
        b = np.asarray(p1, dtype=np.float64)
        dist = float(np.linalg.norm(b - a))
        steps = max(1, int(math.ceil(dist / spacing)))
        for k in range(1, steps + 1):
            p = a + (b - a) * (k / steps)
            out.append((float(p[0]), float(p[1])))
    return _dedupe(out)


def shortcut(path, grid):
    """Greedily remove unnecessary A* waypoints, then resample multi-leg paths."""

    path = _dedupe(path)
    if len(path) <= 2:
        return path

    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_is_free(grid, path[i], path[j]):
            j -= 1
        smoothed.append(path[j])
        i = j
    return _resample_polyline(smoothed, spacing=0.05)


def plan_path(obstacles, start_xy, goal_xy, robot_radius, bounds):
    """Build an inflated occupancy grid and return smoothed world XY waypoints."""

    x_min, x_max, y_min, y_max = [float(v) for v in bounds]
    grid = OccupancyGrid(
        origin_xy=(x_min, y_min),
        size_xy=(x_max - x_min, y_max - y_min),
        resolution=0.02,
    )
    for obs in obstacles:
        grid.add_box(obs["center"], obs["half"], margin=float(obs.get("margin", 0.0)))
    grid.inflate(robot_radius)

    path = astar(grid, start_xy, goal_xy)
    if path is None:
        return None
    return shortcut(path, grid)
