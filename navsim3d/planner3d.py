from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .map3d import GridMap3D

Node = Tuple[int, int, int]


@dataclass
class PlanResult:
    path: List[Node]
    cost: float


def manhattan(a: Node, b: Node) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def euclidean(a: Node, b: Node) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    )


def _neighbor_offsets(connectivity: int) -> List[Tuple[int, int, int]]:
    if connectivity == 6:
        return [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]
    if connectivity == 26:
        offsets = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    offsets.append((dx, dy, dz))
        return offsets
    raise ValueError("Connectivity must be 6 or 26.")


def _neighbors(
    grid: GridMap3D, node: Node, connectivity: int
) -> Iterable[Tuple[Node, float]]:
    x, y, z = node
    for dx, dy, dz in _neighbor_offsets(connectivity):
        nxt = (x + dx, y + dy, z + dz)
        if grid.in_bounds(nxt) and grid.is_free(nxt):
            step_cost = math.sqrt(dx * dx + dy * dy + dz * dz)
            yield nxt, step_cost


def reconstruct(came_from: Dict[Node, Node], start: Node, goal: Node) -> List[Node]:
    node = goal
    path = [node]
    while node != start:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


def astar(
    grid: GridMap3D,
    start: Node,
    goal: Node,
    connectivity: int = 26,
) -> Optional[PlanResult]:
    if not grid.in_bounds(start) or not grid.in_bounds(goal):
        return None
    if not grid.is_free(start) or not grid.is_free(goal):
        return None

    heuristic = euclidean if connectivity == 26 else manhattan

    open_heap: List[Tuple[float, Node]] = []
    heapq.heappush(open_heap, (0.0, start))
    came_from: Dict[Node, Node] = {}
    g_cost: Dict[Node, float] = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = reconstruct(came_from, start, goal)
            return PlanResult(path=path, cost=g_cost[current])

        for nxt, step_cost in _neighbors(grid, current, connectivity):
            tentative = g_cost[current] + step_cost
            if nxt not in g_cost or tentative < g_cost[nxt]:
                came_from[nxt] = current
                g_cost[nxt] = tentative
                f_cost = tentative + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f_cost, nxt))

    return None


def plan_path(
    grid: GridMap3D,
    start: Node,
    goal: Node,
    connectivity: int = 26,
) -> Optional[PlanResult]:
    return astar(grid, start, goal, connectivity=connectivity)
