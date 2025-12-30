from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import yaml

from navsim3d.control3d import UAVParams
from navsim3d.costmap3d import CostMap3D
from navsim3d.dynamic3d import DynamicObstacle3D, DynamicObstacleField3D
from navsim3d.learned_pred import load_predictor
from navsim3d.map3d import demo_grid
from navsim3d.swarm3d import (
    SwarmMetrics,
    SwarmParams,
    generate_swarm_tasks,
    plan_swarm_paths,
    simulate_swarm,
    simulate_swarm_cbs,
    simulate_swarm_cooperative,
    simulate_swarm_prioritized,
)
from navsim3d.viz3d import plot_swarm_scene, render_swarm_gif


@dataclass
class SwarmConfig:
    mode: str
    map_name: str
    output_png: Path
    output_gif: Path | None
    inflation_radius: float
    connectivity: int
    agents: int
    seed: int
    min_start_separation: float
    min_goal_separation: float
    min_goal_distance: float
    starts: List[Tuple[int, int, int]] | None
    goals: List[Tuple[int, int, int]] | None
    dynamic_enabled: bool
    dynamic_obstacles: List[DynamicObstacle3D]
    prediction_enabled: bool
    prediction_history_steps: int
    prediction_horizon_steps: int
    prediction_model_path: Path
    swarm_params: SwarmParams
    ctrl_params: UAVParams


def _load_config(path: Path) -> SwarmConfig:
    data = yaml.safe_load(path.read_text()) or {}
    cfg = data.get("swarm", {}) or {}

    seed = int(cfg.get("seed", 4))
    comm_seed = int(cfg.get("comm_seed", seed))
    comm_dropout = float(cfg.get("comm_dropout", 0.0))
    comm_dropout = max(0.0, min(1.0, comm_dropout))

    dyn_cfg = cfg.get("dynamic_obstacles", {}) or {}
    dyn_enabled = bool(dyn_cfg.get("enabled", False))
    dyn_obstacles = _parse_dynamic_obstacles(dyn_cfg)

    pred_cfg = cfg.get("learned_prediction", {}) or {}
    pred_enabled = bool(pred_cfg.get("enabled", False))
    pred_history = int(pred_cfg.get("history_steps", 4))
    pred_horizon = int(pred_cfg.get("horizon_steps", 5))
    pred_model = Path(pred_cfg.get("model_path", "models/obstacle_mlp.json"))
    map_name = str(cfg.get("map", "demo"))

    starts = _parse_points(cfg.get("starts"))
    goals = _parse_points(cfg.get("goals"))

    return SwarmConfig(
        mode=str(cfg.get("mode", "distributed")),
        map_name=map_name,
        output_png=Path(cfg.get("output_png", "swarm.png")),
        output_gif=Path(cfg["output_gif"]) if cfg.get("output_gif") else None,
        inflation_radius=float(cfg.get("inflation_radius", 0.0)),
        connectivity=int(cfg.get("connectivity", 26)),
        agents=int(cfg.get("agents", 12)),
        seed=seed,
        min_start_separation=float(cfg.get("min_start_separation", 1.5)),
        min_goal_separation=float(cfg.get("min_goal_separation", 1.5)),
        min_goal_distance=float(cfg.get("min_goal_distance", 6.0)),
        starts=starts,
        goals=goals,
        dynamic_enabled=dyn_enabled,
        dynamic_obstacles=dyn_obstacles,
        prediction_enabled=pred_enabled,
        prediction_history_steps=pred_history,
        prediction_horizon_steps=pred_horizon,
        prediction_model_path=pred_model,
        swarm_params=SwarmParams(
            dt=float(cfg.get("dt", 0.1)),
            max_steps=int(cfg.get("max_steps", 700)),
            goal_tolerance=float(cfg.get("goal_tolerance", 0.4)),
            min_separation=float(cfg.get("min_separation", 0.8)),
            neighbor_radius=float(cfg.get("neighbor_radius", 2.5)),
            avoidance_gain=float(cfg.get("avoidance_gain", 1.2)),
            obstacle_avoidance_gain=float(cfg.get("obstacle_avoidance_gain", 0.9)),
            obstacle_avoidance_radius=float(cfg.get("obstacle_avoidance_radius", 2.5)),
            start_delay_steps=int(cfg.get("start_delay_steps", 3)),
            respect_obstacles=bool(cfg.get("respect_obstacles", False)),
            goal_hold_steps=int(cfg.get("goal_hold_steps", 30)),
            comm_range=float(cfg.get("comm_range", 0.0)),
            comm_delay_steps=int(cfg.get("comm_delay_steps", 0)),
            comm_dropout=comm_dropout,
            comm_max_neighbors=int(cfg.get("comm_max_neighbors", 0)),
            comm_seed=comm_seed,
        ),
        ctrl_params=UAVParams(
            lookahead=float(cfg.get("lookahead", 1.2)),
            speed=float(cfg.get("speed", 1.0)),
            yaw_rate_max=float(cfg.get("yaw_rate_max", 1.5)),
            pitch_rate_max=float(cfg.get("pitch_rate_max", 1.0)),
            max_pitch=float(cfg.get("max_pitch", 0.9)),
            yaw_gain=float(cfg.get("yaw_gain", 1.2)),
            pitch_gain=float(cfg.get("pitch_gain", 1.0)),
        ),
    )


def _parse_points(raw: object) -> List[Tuple[int, int, int]] | None:
    if raw is None:
        return None
    points: List[Tuple[int, int, int]] = []
    for item in raw:
        if len(item) != 3:
            raise ValueError("Each point must have 3 coordinates.")
        points.append((int(item[0]), int(item[1]), int(item[2])))
    return points


def _parse_dynamic_obstacles(raw: dict) -> List[DynamicObstacle3D]:
    obstacles: List[DynamicObstacle3D] = []
    for obstacle in raw.get("obstacles", []) or []:
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
    return obstacles


def _parse_sweep(value: str) -> List[int]:
    counts: List[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            count = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid sweep entry: {item}") from exc
        if count <= 0:
            raise ValueError("Sweep entries must be positive integers.")
        counts.append(count)
    if not counts:
        raise ValueError("Sweep list is empty.")
    return counts


def _parse_nonnegative_sweep(value: str) -> List[int]:
    counts: List[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            count = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid sweep entry: {item}") from exc
        if count < 0:
            raise ValueError("Sweep entries must be non-negative integers.")
        counts.append(count)
    if not counts:
        raise ValueError("Sweep list is empty.")
    return counts


def _parse_float_sweep(value: str) -> List[float]:
    values: List[float] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            parsed = float(item)
        except ValueError as exc:
            raise ValueError(f"Invalid sweep entry: {item}") from exc
        values.append(parsed)
    if not values:
        raise ValueError("Sweep list is empty.")
    return values


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _clone_dynamic_field(cfg: SwarmConfig) -> DynamicObstacleField3D | None:
    if not cfg.dynamic_enabled or not cfg.dynamic_obstacles:
        return None
    obstacles = [
        DynamicObstacle3D(
            x=obs.x,
            y=obs.y,
            z=obs.z,
            vx=obs.vx,
            vy=obs.vy,
            vz=obs.vz,
        )
        for obs in cfg.dynamic_obstacles
    ]
    return DynamicObstacleField3D(obstacles)


def _run_swarm(
    grid,
    costmap,
    starts,
    goals,
    cfg: SwarmConfig,
) -> Tuple[
    List[List[Tuple[float, float, float]]],
    List[List[Tuple[float, float, float]]],
    SwarmMetrics,
]:
    paths = plan_swarm_paths(costmap, starts, goals, connectivity=cfg.connectivity)
    dynamic_field = _clone_dynamic_field(cfg)
    predictor = None
    if cfg.prediction_enabled and cfg.mode == "distributed":
        try:
            predictor = load_predictor(
                cfg.prediction_model_path,
                cfg.prediction_horizon_steps,
                expected_history_steps=cfg.prediction_history_steps,
            )
        except Exception as exc:
            raise SystemExit(f"Failed to load predictor: {exc}") from exc
    if cfg.mode == "prioritized":
        trajectories, metrics = simulate_swarm_prioritized(
            grid,
            paths,
            goals,
            cfg.swarm_params,
            dynamic_field,
        )
    elif cfg.mode == "cooperative":
        trajectories, metrics = simulate_swarm_cooperative(
            grid,
            costmap,
            starts,
            goals,
            cfg.swarm_params,
            cfg.connectivity,
            dynamic_field,
        )
    elif cfg.mode == "cbs":
        trajectories, metrics = simulate_swarm_cbs(
            grid,
            costmap,
            starts,
            goals,
            cfg.swarm_params,
            cfg.connectivity,
            dynamic_field,
        )
    else:
        trajectories, metrics = simulate_swarm(
            grid,
            costmap,
            paths,
            starts,
            goals,
            cfg.swarm_params,
            cfg.ctrl_params,
            dynamic_field,
            predictor,
        )
    return paths, trajectories, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="3D multi-agent swarm demo.")
    parser.add_argument("--config", type=Path, default=Path("configs/swarm.yaml"))
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--png", type=Path, default=None)
    parser.add_argument("--gif", type=Path, default=None)
    parser.add_argument(
        "--map",
        type=str,
        choices=["demo", "large"],
        default=None,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["distributed", "prioritized", "cooperative", "cbs"],
        default=None,
    )
    parser.add_argument(
        "--predictive",
        dest="predictive_enabled",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-predictive",
        dest="predictive_enabled",
        action="store_false",
        default=None,
    )
    parser.add_argument("--predictor", type=Path, default=None)
    parser.add_argument("--prediction-history", type=int, default=None)
    parser.add_argument("--prediction-horizon", type=int, default=None)
    parser.add_argument(
        "--respect-obstacles",
        dest="respect_obstacles",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-respect-obstacles",
        dest="respect_obstacles",
        action="store_false",
        default=None,
    )
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
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Comma-separated agent counts, e.g. 4,8,12",
    )
    parser.add_argument(
        "--comm-dropout-sweep",
        type=str,
        default=None,
        help="Comma-separated dropout rates, e.g. 0,0.2,0.4",
    )
    parser.add_argument(
        "--comm-delay-sweep",
        type=str,
        default=None,
        help="Comma-separated delay steps, e.g. 0,3,6",
    )
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.agents is not None:
        cfg.agents = args.agents
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.swarm_params.comm_seed = args.seed
    if args.png is not None:
        cfg.output_png = args.png
    if args.gif is not None:
        cfg.output_gif = args.gif
    if args.mode is not None:
        cfg.mode = args.mode
    if args.map is not None:
        cfg.map_name = args.map
    if args.predictive_enabled is not None:
        cfg.prediction_enabled = args.predictive_enabled
    if args.predictor is not None:
        cfg.prediction_model_path = args.predictor
    if args.prediction_history is not None:
        cfg.prediction_history_steps = args.prediction_history
    if args.prediction_horizon is not None:
        cfg.prediction_horizon_steps = args.prediction_horizon
    if args.respect_obstacles is not None:
        cfg.swarm_params.respect_obstacles = args.respect_obstacles
    if args.dynamic_enabled is not None:
        cfg.dynamic_enabled = args.dynamic_enabled

    if cfg.map_name == "large":
        from navsim3d.map3d import large_demo_grid

        grid = large_demo_grid()
    else:
        grid = demo_grid()
    costmap = CostMap3D.from_grid(grid, cfg.inflation_radius)

    sweep_flags = [
        args.sweep is not None,
        args.comm_dropout_sweep is not None,
        args.comm_delay_sweep is not None,
    ]
    if sum(sweep_flags) > 1:
        raise SystemExit("Use only one sweep mode at a time.")

    if args.sweep is not None:
        counts = _parse_sweep(args.sweep)
        rows: List[dict] = []
        for count in counts:
            starts, goals = generate_swarm_tasks(
                grid,
                count,
                cfg.seed,
                cfg.min_start_separation,
                cfg.min_goal_separation,
                cfg.min_goal_distance,
            )
            paths, _, metrics = _run_swarm(grid, costmap, starts, goals, cfg)
            success_rate = metrics.reached / metrics.total if metrics.total else 0.0
            rows.append(
                {
                    "agents": metrics.total,
                    "reached": metrics.reached,
                    "success": f"{success_rate:.2f}",
                    "min_separation": f"{metrics.min_separation:.2f}",
                    "collision_steps": metrics.collision_steps,
                    "mean_path": f"{metrics.mean_path_length:.2f}",
                    "mean_traj": f"{metrics.mean_traj_length:.2f}",
                    "mean_steps_to_goal": (
                        f"{metrics.mean_steps_to_goal:.1f}"
                        if metrics.mean_steps_to_goal is not None
                        else ""
                    ),
                    "mean_neighbors": f"{metrics.mean_neighbors:.2f}",
                    "dynamic_collision_steps": metrics.dynamic_collision_steps,
                    "steps": metrics.steps,
                }
            )
            print(
                "Swarm sweep:",
                f"agents={metrics.total}",
                f"success={success_rate:.2f}",
                f"min_sep={metrics.min_separation:.2f}",
                f"mean_neighbors={metrics.mean_neighbors:.2f}",
                f"mode={cfg.mode}",
            )
        if args.csv is not None:
            _write_csv(args.csv, rows)
        return

    if args.comm_dropout_sweep is not None or args.comm_delay_sweep is not None:
        dropouts = (
            _parse_float_sweep(args.comm_dropout_sweep)
            if args.comm_dropout_sweep is not None
            else [cfg.swarm_params.comm_dropout]
        )
        delays = (
            _parse_nonnegative_sweep(args.comm_delay_sweep)
            if args.comm_delay_sweep is not None
            else [cfg.swarm_params.comm_delay_steps]
        )
        rows: List[dict] = []
        starts, goals = generate_swarm_tasks(
            grid,
            cfg.agents,
            cfg.seed,
            cfg.min_start_separation,
            cfg.min_goal_separation,
            cfg.min_goal_distance,
        )
        for dropout in dropouts:
            dropout = max(0.0, min(1.0, dropout))
            for delay in delays:
                cfg.swarm_params.comm_dropout = dropout
                cfg.swarm_params.comm_delay_steps = delay
                _, _, metrics = _run_swarm(grid, costmap, starts, goals, cfg)
                success_rate = metrics.reached / metrics.total if metrics.total else 0.0
                rows.append(
                    {
                        "agents": metrics.total,
                        "comm_dropout": f"{dropout:.2f}",
                        "comm_delay_steps": delay,
                        "reached": metrics.reached,
                        "success": f"{success_rate:.2f}",
                        "min_separation": f"{metrics.min_separation:.2f}",
                        "collision_steps": metrics.collision_steps,
                        "mean_path": f"{metrics.mean_path_length:.2f}",
                        "mean_traj": f"{metrics.mean_traj_length:.2f}",
                        "mean_steps_to_goal": (
                            f"{metrics.mean_steps_to_goal:.1f}"
                            if metrics.mean_steps_to_goal is not None
                            else ""
                        ),
                        "mean_neighbors": f"{metrics.mean_neighbors:.2f}",
                        "dynamic_collision_steps": metrics.dynamic_collision_steps,
                        "steps": metrics.steps,
                    }
                )
                print(
                    "Comm sweep:",
                    f"dropout={dropout:.2f}",
                    f"delay={delay}",
                    f"success={success_rate:.2f}",
                    f"min_sep={metrics.min_separation:.2f}",
                )
        if args.csv is not None:
            _write_csv(args.csv, rows)
        return

    if cfg.starts is not None or cfg.goals is not None:
        if cfg.starts is None or cfg.goals is None:
            raise SystemExit("Both starts and goals must be provided together.")
        if len(cfg.starts) != len(cfg.goals):
            raise SystemExit("Starts/goals length mismatch.")
        starts = cfg.starts
        goals = cfg.goals
    else:
        starts, goals = generate_swarm_tasks(
            grid,
            cfg.agents,
            cfg.seed,
            cfg.min_start_separation,
            cfg.min_goal_separation,
            cfg.min_goal_distance,
        )

    paths, trajectories, metrics = _run_swarm(grid, costmap, starts, goals, cfg)

    plot_swarm_scene(
        grid,
        paths,
        trajectories,
        starts,
        goals,
        str(cfg.output_png),
        display_grid=costmap.inflated,
    )
    if cfg.output_gif is not None:
        render_swarm_gif(
            grid,
            paths,
            trajectories,
            starts,
            goals,
            str(cfg.output_gif),
            display_grid=costmap.inflated,
        )

    success_rate = metrics.reached / metrics.total if metrics.total else 0.0
    print(
        "Swarm metrics:",
        f"agents={metrics.total}",
        f"reached={metrics.reached}",
        f"success={success_rate:.2f}",
        f"min_sep={metrics.min_separation:.2f}",
        f"collision_steps={metrics.collision_steps}",
        f"dynamic_collision_steps={metrics.dynamic_collision_steps}",
        f"mean_path={metrics.mean_path_length:.2f}",
        f"mean_traj={metrics.mean_traj_length:.2f}",
        f"mean_neighbors={metrics.mean_neighbors:.2f}",
        f"steps={metrics.steps}",
    )


if __name__ == "__main__":
    main()
