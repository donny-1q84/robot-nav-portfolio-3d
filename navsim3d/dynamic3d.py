from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .map3d import GridMap3D

Node = Tuple[int, int, int]


@dataclass
class DynamicObstacle3D:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float

    def step(self, dt: float, grid: GridMap3D) -> None:
        nx = self.x + self.vx * dt
        ny = self.y + self.vy * dt
        nz = self.z + self.vz * dt

        if nx < 0.0 or nx > grid.width - 1:
            self.vx *= -1.0
            nx = max(0.0, min(grid.width - 1, nx))
        if ny < 0.0 or ny > grid.height - 1:
            self.vy *= -1.0
            ny = max(0.0, min(grid.height - 1, ny))
        if nz < 0.0 or nz > grid.depth - 1:
            self.vz *= -1.0
            nz = max(0.0, min(grid.depth - 1, nz))

        cell = (int(round(nx)), int(round(ny)), int(round(nz)))
        if grid.in_bounds(cell) and not grid.is_free(cell):
            self.vx *= -1.0
            self.vy *= -1.0
            self.vz *= -1.0
            nx = self.x
            ny = self.y
            nz = self.z

        self.x = nx
        self.y = ny
        self.z = nz

    def cell(self, grid: GridMap3D) -> Node | None:
        cell = (int(round(self.x)), int(round(self.y)), int(round(self.z)))
        if grid.in_bounds(cell):
            return cell
        return None


@dataclass
class DynamicObstacleField3D:
    obstacles: List[DynamicObstacle3D]

    def step(self, dt: float, grid: GridMap3D) -> None:
        for obstacle in self.obstacles:
            obstacle.step(dt, grid)

    def cells(self, grid: GridMap3D) -> List[Node]:
        cells: List[Node] = []
        for obstacle in self.obstacles:
            cell = obstacle.cell(grid)
            if cell is not None:
                cells.append(cell)
        return cells

    def positions(self) -> List[Tuple[float, float, float]]:
        return [(obs.x, obs.y, obs.z) for obs in self.obstacles]
