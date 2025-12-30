from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from .collision3d import path_in_collision
from .control3d import UAVParams, guidance_command
from .costmap3d import CostMap3D
from .dynamic3d import DynamicObstacleField3D
from .map3d import GridMap3D
from .planner3d import plan_path

Pose3D = Tuple[float, float, float]
PoseState = Tuple[float, float, float, float, float]


@dataclass
class SimParams:
    dt: float = 0.1
    max_steps: int = 600
    goal_tolerance: float = 0.4


def simulate_path(
    path: List[Pose3D],
    start_pose: Pose3D,
    params: SimParams,
    ctrl_params: UAVParams,
) -> List[Pose3D]:
    poses: List[Pose3D] = [start_pose]
    target_idx = 0
    yaw, pitch = _initial_orientation(path, start_pose)
    state: PoseState = (start_pose[0], start_pose[1], start_pose[2], yaw, pitch)

    for _ in range(params.max_steps):
        x, y, z, yaw, pitch = state
        gx, gy, gz = path[-1]
        if math.sqrt((gx - x) ** 2 + (gy - y) ** 2 + (gz - z) ** 2) <= params.goal_tolerance:
            break

        speed, yaw_rate, pitch_rate, target_idx = guidance_command(
            state, path, ctrl_params, target_idx
        )
        yaw = _wrap_angle(yaw + yaw_rate * params.dt)
        pitch = _clamp(
            pitch + pitch_rate * params.dt,
            -ctrl_params.max_pitch,
            ctrl_params.max_pitch,
        )

        vx = speed * math.cos(pitch) * math.cos(yaw)
        vy = speed * math.cos(pitch) * math.sin(yaw)
        vz = speed * math.sin(pitch)
        x += vx * params.dt
        y += vy * params.dt
        z += vz * params.dt
        state = (x, y, z, yaw, pitch)
        poses.append((x, y, z))

    return poses


def simulate_dynamic(
    path: List[Pose3D],
    start_pose: Pose3D,
    params: SimParams,
    base_grid: GridMap3D,
    inflation_radius: float,
    dynamic_field: DynamicObstacleField3D,
    goal: Tuple[int, int, int],
    replan_interval: int,
    max_replans: int,
    connectivity: int,
    ctrl_params: UAVParams,
) -> Tuple[List[Pose3D], List[Pose3D], List[List[Pose3D]]]:
    poses: List[Pose3D] = [start_pose]
    dynamic_history: List[List[Pose3D]] = [dynamic_field.positions()]
    target_idx = 0
    yaw, pitch = _initial_orientation(path, start_pose)
    state: PoseState = (start_pose[0], start_pose[1], start_pose[2], yaw, pitch)
    current_path = path
    stuck_steps = 0
    steps_since_replan = 0
    replans = 0

    for _ in range(params.max_steps):
        x, y, z, yaw, pitch = state
        gx, gy, gz = goal
        if math.sqrt((gx - x) ** 2 + (gy - y) ** 2 + (gz - z) ** 2) <= params.goal_tolerance:
            break

        dynamic_field.step(params.dt, base_grid)
        full_costmap = CostMap3D.from_grid(
            base_grid, inflation_radius, occupied=dynamic_field.cells(base_grid)
        )

        needs_replan = False
        if replan_interval > 0 and steps_since_replan >= replan_interval:
            needs_replan = True
        if path_in_collision(full_costmap, current_path):
            needs_replan = True

        if needs_replan:
            start_cell = _pose_to_cell((x, y, z))
            plan = plan_path(
                full_costmap.inflated_map(), start_cell, goal, connectivity=connectivity
            )
            if plan is None:
                break
            current_path = _grid_to_path(plan.path)
            target_idx = 0
            steps_since_replan = 0
            replans += 1
            if replans >= max_replans:
                break

        speed, yaw_rate, pitch_rate, target_idx = guidance_command(
            state, current_path, ctrl_params, target_idx
        )
        if speed < 1e-3:
            stuck_steps += 1
            if stuck_steps >= 10:
                steps_since_replan = replan_interval
        else:
            stuck_steps = 0

        yaw = _wrap_angle(yaw + yaw_rate * params.dt)
        pitch = _clamp(
            pitch + pitch_rate * params.dt,
            -ctrl_params.max_pitch,
            ctrl_params.max_pitch,
        )
        vx = speed * math.cos(pitch) * math.cos(yaw)
        vy = speed * math.cos(pitch) * math.sin(yaw)
        vz = speed * math.sin(pitch)
        x += vx * params.dt
        y += vy * params.dt
        z += vz * params.dt
        state = (x, y, z, yaw, pitch)
        poses.append((x, y, z))
        dynamic_history.append(dynamic_field.positions())
        steps_since_replan += 1

    return poses, current_path, dynamic_history


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _initial_orientation(path: List[Pose3D], start_pose: Pose3D) -> Tuple[float, float]:
    for point in path:
        dx = point[0] - start_pose[0]
        dy = point[1] - start_pose[1]
        dz = point[2] - start_pose[2]
        if abs(dx) + abs(dy) + abs(dz) > 1e-6:
            yaw = math.atan2(dy, dx)
            pitch = math.atan2(dz, max(math.hypot(dx, dy), 1e-9))
            return yaw, pitch
    return 0.0, 0.0


def _grid_to_path(plan: List[Tuple[int, int, int]]) -> List[Pose3D]:
    return [(float(x), float(y), float(z)) for x, y, z in plan]


def _pose_to_cell(pose: Pose3D) -> Tuple[int, int, int]:
    return int(round(pose[0])), int(round(pose[1])), int(round(pose[2]))
