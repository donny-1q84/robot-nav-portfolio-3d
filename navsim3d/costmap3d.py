from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

from .map3d import Grid3D, GridMap3D

Node = Tuple[int, int, int]


def _overlay_grid(grid: Grid3D, occupied: Iterable[Node]) -> Grid3D:
    depth = len(grid)
    height = len(grid[0]) if depth else 0
    width = len(grid[0][0]) if depth and height else 0
    overlaid = [[row[:] for row in layer] for layer in grid]
    for x, y, z in occupied:
        if 0 <= x < width and 0 <= y < height and 0 <= z < depth:
            overlaid[z][y][x] = 1
    return overlaid


def _inflate_grid(grid: Grid3D, radius: float) -> Grid3D:
    depth = len(grid)
    height = len(grid[0]) if depth else 0
    width = len(grid[0][0]) if depth and height else 0
    inflated = [[row[:] for row in layer] for layer in grid]
    if radius <= 0.0:
        return inflated

    rad = int(math.ceil(radius))
    radius_sq = radius * radius + 1e-9

    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if grid[z][y][x] != 1:
                    continue
                for dz in range(-rad, rad + 1):
                    for dy in range(-rad, rad + 1):
                        for dx in range(-rad, rad + 1):
                            if dx * dx + dy * dy + dz * dz > radius_sq:
                                continue
                            nz = z + dz
                            ny = y + dy
                            nx = x + dx
                            if 0 <= nx < width and 0 <= ny < height and 0 <= nz < depth:
                                inflated[nz][ny][nx] = 1
    return inflated


@dataclass(frozen=True)
class CostMap3D:
    base: GridMap3D
    inflated: Grid3D
    inflation_radius: float

    @classmethod
    def from_grid(
        cls,
        grid: GridMap3D,
        inflation_radius: float,
        occupied: Iterable[Node] | None = None,
    ) -> "CostMap3D":
        radius = max(0.0, float(inflation_radius))
        base_grid = _overlay_grid(grid.grid, occupied) if occupied else grid.grid
        inflated = _inflate_grid(base_grid, radius)
        return cls(base=grid, inflated=inflated, inflation_radius=radius)

    @property
    def depth(self) -> int:
        return self.base.depth

    @property
    def height(self) -> int:
        return self.base.height

    @property
    def width(self) -> int:
        return self.base.width

    def in_bounds(self, node: Node) -> bool:
        return self.base.in_bounds(node)

    def is_occupied(self, node: Node) -> bool:
        x, y, z = node
        return self.inflated[z][y][x] == 1

    def inflated_map(self) -> GridMap3D:
        return GridMap3D(grid=self.inflated)
