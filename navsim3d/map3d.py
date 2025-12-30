from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Grid3D = list[list[list[int]]]


@dataclass(frozen=True)
class GridMap3D:
    grid: Grid3D

    @property
    def depth(self) -> int:
        return len(self.grid)

    @property
    def height(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    @property
    def width(self) -> int:
        return len(self.grid[0][0]) if self.grid and self.grid[0] else 0

    def in_bounds(self, node: Tuple[int, int, int]) -> bool:
        x, y, z = node
        return 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth

    def is_free(self, node: Tuple[int, int, int]) -> bool:
        x, y, z = node
        return self.grid[z][y][x] == 0


def _empty_grid(width: int, height: int, depth: int) -> Grid3D:
    return [[[0 for _ in range(width)] for _ in range(height)] for _ in range(depth)]


def _add_box(
    grid: Grid3D,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    z0: int,
    z1: int,
) -> None:
    depth = len(grid)
    height = len(grid[0]) if depth else 0
    width = len(grid[0][0]) if depth and height else 0
    for z in range(max(0, z0), min(depth, z1)):
        for y in range(max(0, y0), min(height, y1)):
            for x in range(max(0, x0), min(width, x1)):
                grid[z][y][x] = 1


def demo_grid() -> GridMap3D:
    width, height, depth = 12, 12, 6
    grid = _empty_grid(width, height, depth)

    # Central pillar.
    _add_box(grid, 4, 8, 4, 8, 0, 6)

    # Low wall with a gap.
    _add_box(grid, 1, 11, 3, 4, 0, 3)
    for z in range(0, 3):
        grid[z][3][6] = 0

    # High wall with a mid-level gap.
    _add_box(grid, 8, 9, 1, 10, 3, 6)
    for y in range(1, 10):
        grid[4][y][8] = 0

    return GridMap3D(grid=grid)
