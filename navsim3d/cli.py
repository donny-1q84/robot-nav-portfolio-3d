from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import yaml

from navsim3d.collision3d import path_in_collision, trajectory_in_collision
from navsim3d.control3d import UAVParams
from navsim3d.costmap3d import CostMap3D
from navsim3d.dynamic3d import DynamicObstacle3D, DynamicObstacleField3D
from navsim3d.map3d import GridMap3D, demo_grid
from navsim3d.planner3d import plan_path
from navsim3d.sim3d import SimParams, simulate_dynamic, simulate_path
from navsim3d.viz3d import plot_scene, render_gif


@dataclass
class DemoConfig:
    start: Tuple[int, int, int]
    goal: Tuple[int, int, int]
    output_png: Path
    output_gif: Path | None
    inflation_radius: float
    connectivity: int
    lookahead: float
    speed: float
    yaw_rate_max: float
    pitch_rate_max: float
    max_pitch: float
    yaw_gain: float
    pitch_gain: float
    dynamic_enabled: bool
    dynamic_replan_interval: int
    dynamic_max_replans: int
    dynamic_obstacles: List[DynamicObstacle3D]


def _parse_point(value: str) -> Tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Point must be in x,y,z format.")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _load_config(path: Path) -> DemoConfig:
    data = yaml.safe_load(path.read_text()) or {}
    dyn_cfg = data.get("dynamic_obstacles", {}) or {}
    obstacles: List[DynamicObstacle3D] = []
    for obstacle in dyn_cfg.get("obstacles", []) or []:
        position = obstacle.get("position", [0.0, 0.0, 0.0])
        velocity = obstacle.get("velocity", [0.0, 0.0, 0.0])
        obstacles.append(
            DynamicObstacle3D(
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
                vx=float(velocity[0]),
                vy=float(velocity[1]),
                vz=float(velocity[2]),
            )
        )
    return DemoConfig(
        start=tuple(data.get("start", [0, 0, 0])),
        goal=tuple(data.get("goal", [11, 11, 5])),
        output_png=Path(data.get("output_png", "output.png")),
        output_gif=Path(data["output_gif"]) if data.get("output_gif") else None,
        inflation_radius=float(data.get("inflation_radius", 0.0)),
        connectivity=int(data.get("connectivity", 26)),
        lookahead=float(data.get("lookahead", 1.2)),
        speed=float(data.get("speed", 1.0)),
        yaw_rate_max=float(data.get("yaw_rate_max", 1.5)),
        pitch_rate_max=float(data.get("pitch_rate_max", 1.0)),
        max_pitch=float(data.get("max_pitch", 0.9)),
        yaw_gain=float(data.get("yaw_gain", 1.2)),
        pitch_gain=float(data.get("pitch_gain", 1.0)),
        dynamic_enabled=bool(dyn_cfg.get("enabled", False)),
        dynamic_replan_interval=int(dyn_cfg.get("replan_interval", 10)),
        dynamic_max_replans=int(dyn_cfg.get("max_replans", 50)),
        dynamic_obstacles=obstacles,
    )


def _grid_to_path(plan: list[Tuple[int, int, int]]) -> list[Tuple[float, float, float]]:
    return [(float(x), float(y), float(z)) for x, y, z in plan]


def run_demo(
    grid: GridMap3D,
    cfg: DemoConfig,
    out_png: Path | None = None,
    out_gif: Path | None = None,
) -> None:
    dynamic_field = None
    dynamic_cells = None
    if cfg.dynamic_enabled and cfg.dynamic_obstacles:
        dynamic_field = DynamicObstacleField3D(cfg.dynamic_obstacles)
        dynamic_cells = dynamic_field.cells(grid)

    static_costmap = CostMap3D.from_grid(grid, cfg.inflation_radius)
    costmap = CostMap3D.from_grid(grid, cfg.inflation_radius, occupied=dynamic_cells)
    plan = plan_path(costmap.inflated_map(), cfg.start, cfg.goal, cfg.connectivity)
    if plan is None:
        raise SystemExit("No path found for the given start/goal.")

    path = _grid_to_path(plan.path)
    start_pose = (float(cfg.start[0]), float(cfg.start[1]), float(cfg.start[2]))
    ctrl_params = UAVParams(
        lookahead=cfg.lookahead,
        speed=cfg.speed,
        yaw_rate_max=cfg.yaw_rate_max,
        pitch_rate_max=cfg.pitch_rate_max,
        max_pitch=cfg.max_pitch,
        yaw_gain=cfg.yaw_gain,
        pitch_gain=cfg.pitch_gain,
    )
    dynamic_history = None
    if cfg.dynamic_enabled and dynamic_field is not None:
        poses, path, dynamic_history = simulate_dynamic(
            path,
            start_pose,
            SimParams(),
            grid,
            cfg.inflation_radius,
            dynamic_field,
            cfg.goal,
            cfg.dynamic_replan_interval,
            cfg.dynamic_max_replans,
            cfg.connectivity,
            ctrl_params,
        )
        costmap = CostMap3D.from_grid(
            grid, cfg.inflation_radius, occupied=dynamic_field.cells(grid)
        )
    else:
        poses = simulate_path(path, start_pose, SimParams(), ctrl_params)

    png_path = out_png if out_png is not None else cfg.output_png
    if path_in_collision(costmap, path):
        print("Warning: planned path intersects inflated obstacles.")
    if trajectory_in_collision(costmap, poses):
        print("Warning: trajectory intersects inflated obstacles.")

    display_grid = static_costmap.inflated if dynamic_history else costmap.inflated
    plot_scene(
        grid,
        path,
        poses,
        cfg.start,
        cfg.goal,
        str(png_path),
        display_grid=display_grid,
        dynamic_positions=dynamic_history[-1] if dynamic_history else None,
    )

    if out_gif is not None:
        render_gif(
            grid,
            path,
            poses,
            cfg.start,
            cfg.goal,
            str(out_gif),
            display_grid=display_grid,
            dynamic_history=dynamic_history,
        )
    elif cfg.output_gif is not None:
        render_gif(
            grid,
            path,
            poses,
            cfg.start,
            cfg.goal,
            str(cfg.output_gif),
            display_grid=display_grid,
            dynamic_history=dynamic_history,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="3D robot navigation demo.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--start", type=_parse_point, default=None)
    parser.add_argument("--goal", type=_parse_point, default=None)
    parser.add_argument("--png", type=Path, default=None)
    parser.add_argument("--gif", type=Path, default=None)
    parser.add_argument("--inflation-radius", type=float, default=None)
    parser.add_argument("--connectivity", type=int, choices=[6, 26], default=None)
    parser.add_argument("--lookahead", type=float, default=None)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--yaw-rate-max", type=float, default=None)
    parser.add_argument("--pitch-rate-max", type=float, default=None)
    parser.add_argument("--max-pitch", type=float, default=None)
    parser.add_argument("--yaw-gain", type=float, default=None)
    parser.add_argument("--pitch-gain", type=float, default=None)
    parser.add_argument(
        "--dynamic",
        dest="dynamic_enabled",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-dynamic",
        dest="dynamic_enabled",
        action="store_false",
        default=None,
    )
    parser.add_argument("--replan-interval", type=int, default=None)
    parser.add_argument("--max-replans", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.start is not None:
        cfg.start = args.start
    if args.goal is not None:
        cfg.goal = args.goal
    if args.png is not None:
        cfg.output_png = args.png
    if args.gif is not None:
        cfg.output_gif = args.gif
    if args.inflation_radius is not None:
        cfg.inflation_radius = args.inflation_radius
    if args.connectivity is not None:
        cfg.connectivity = args.connectivity
    if args.lookahead is not None:
        cfg.lookahead = args.lookahead
    if args.speed is not None:
        cfg.speed = args.speed
    if args.yaw_rate_max is not None:
        cfg.yaw_rate_max = args.yaw_rate_max
    if args.pitch_rate_max is not None:
        cfg.pitch_rate_max = args.pitch_rate_max
    if args.max_pitch is not None:
        cfg.max_pitch = args.max_pitch
    if args.yaw_gain is not None:
        cfg.yaw_gain = args.yaw_gain
    if args.pitch_gain is not None:
        cfg.pitch_gain = args.pitch_gain
    if args.dynamic_enabled is not None:
        cfg.dynamic_enabled = args.dynamic_enabled
    if args.replan_interval is not None:
        cfg.dynamic_replan_interval = args.replan_interval
    if args.max_replans is not None:
        cfg.dynamic_max_replans = args.max_replans

    grid = demo_grid()
    run_demo(grid, cfg, out_png=args.png, out_gif=args.gif)


if __name__ == "__main__":
    main()
