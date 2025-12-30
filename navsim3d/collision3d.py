from __future__ import annotations

from typing import Iterable, Tuple

from .costmap3d import CostMap3D

Point3D = Tuple[float, float, float]


def _point_to_cell(point: Point3D) -> Tuple[int, int, int]:
    x, y, z = point
    return int(round(x)), int(round(y)), int(round(z))


def point_in_collision(costmap: CostMap3D, point: Point3D) -> bool:
    cell = _point_to_cell(point)
    if not costmap.in_bounds(cell):
        return True
    return costmap.is_occupied(cell)


def path_in_collision(costmap: CostMap3D, path: Iterable[Point3D]) -> bool:
    return any(point_in_collision(costmap, point) for point in path)


def trajectory_in_collision(costmap: CostMap3D, poses: Iterable[Point3D]) -> bool:
    return any(point_in_collision(costmap, pose) for pose in poses)
