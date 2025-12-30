from __future__ import annotations

import argparse
import copy
import csv
import math
import statistics
from pathlib import Path
from typing import Callable, Dict, List

from navsim3d.costmap3d import CostMap3D
from navsim3d.dynamic3d import DynamicObstacle3D, DynamicObstacleField3D
from navsim3d.learned_pred import load_predictor
from navsim3d.map3d import demo_grid, large_demo_grid
from navsim3d.swarm3d import (
    SwarmMetrics,
    generate_swarm_tasks,
    plan_swarm_paths,
    simulate_swarm,
    simulate_swarm_cbs,
    simulate_swarm_cooperative,
    simulate_swarm_prioritized,
)
from navsim3d import swarm_cli


REPORTS = Path("reports")


def _mean_ci(values: List[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    mean = sum(values) / len(values)
    stdev = statistics.stdev(values)
    ci = 1.96 * stdev / math.sqrt(len(values))
    return mean, ci


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _clone_dynamic_field(cfg) -> DynamicObstacleField3D | None:
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


def _load_predictor(cfg) -> object | None:
    if not cfg.prediction_enabled:
        return None
    return load_predictor(
        cfg.prediction_model_path,
        cfg.prediction_horizon_steps,
        expected_history_steps=cfg.prediction_history_steps,
    )


def _run_once(cfg, grid, costmap, starts, goals) -> SwarmMetrics:
    paths = plan_swarm_paths(costmap, starts, goals, connectivity=cfg.connectivity)
    dynamic_field = _clone_dynamic_field(cfg)
    predictor = _load_predictor(cfg)

    if cfg.mode == "prioritized":
        _, metrics = simulate_swarm_prioritized(
            grid,
            paths,
            goals,
            cfg.swarm_params,
            dynamic_field,
        )
    elif cfg.mode == "cooperative":
        _, metrics = simulate_swarm_cooperative(
            grid,
            costmap,
            starts,
            goals,
            cfg.swarm_params,
            cfg.connectivity,
            dynamic_field,
        )
    elif cfg.mode == "cbs":
        _, metrics = simulate_swarm_cbs(
            grid,
            costmap,
            starts,
            goals,
            cfg.swarm_params,
            cfg.connectivity,
            dynamic_field,
        )
    else:
        _, metrics = simulate_swarm(
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
    return metrics


def _aggregate_metrics(results: List[SwarmMetrics]) -> Dict[str, float]:
    success = [m.reached / m.total if m.total else 0.0 for m in results]
    min_sep = [m.min_separation for m in results]
    collisions = [float(m.collision_steps) for m in results]
    dyn_collisions = [float(m.dynamic_collision_steps) for m in results]
    neighbors = [m.mean_neighbors for m in results]
    steps_to_goal = [
        m.mean_steps_to_goal for m in results if m.mean_steps_to_goal is not None
    ]
    steps = [float(m.steps) for m in results]

    success_mean, success_ci = _mean_ci(success)
    min_sep_mean, min_sep_ci = _mean_ci(min_sep)
    collisions_mean, collisions_ci = _mean_ci(collisions)
    dyn_mean, dyn_ci = _mean_ci(dyn_collisions)
    neighbors_mean, neighbors_ci = _mean_ci(neighbors)
    steps_mean, steps_ci = _mean_ci(steps)
    steps_goal_mean, steps_goal_ci = _mean_ci(steps_to_goal) if steps_to_goal else (0.0, 0.0)

    return {
        "success_mean": success_mean,
        "success_ci": success_ci,
        "min_separation_mean": min_sep_mean,
        "min_separation_ci": min_sep_ci,
        "collision_steps_mean": collisions_mean,
        "collision_steps_ci": collisions_ci,
        "dynamic_collision_steps_mean": dyn_mean,
        "dynamic_collision_steps_ci": dyn_ci,
        "mean_neighbors_mean": neighbors_mean,
        "mean_neighbors_ci": neighbors_ci,
        "mean_steps_mean": steps_mean,
        "mean_steps_ci": steps_ci,
        "mean_steps_to_goal_mean": steps_goal_mean,
        "mean_steps_to_goal_ci": steps_goal_ci,
    }


def _run_sweep(
    cfg_factory: Callable[[int], object],
    grid_factory: Callable[[], object],
    counts: List[int],
    seeds: List[int],
    out_path: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    for count in counts:
        results: List[SwarmMetrics] = []
        for seed in seeds:
            cfg = cfg_factory(seed)
            cfg.agents = count
            grid = grid_factory()
            costmap = CostMap3D.from_grid(grid, cfg.inflation_radius)
            starts, goals = generate_swarm_tasks(
                grid,
                cfg.agents,
                seed,
                cfg.min_start_separation,
                cfg.min_goal_separation,
                cfg.min_goal_distance,
            )
            metrics = _run_once(cfg, grid, costmap, starts, goals)
            results.append(metrics)
        agg = _aggregate_metrics(results)
        rows.append(
            {
                "agents": count,
                **agg,
            }
        )
    _write_csv(out_path, rows)


def _run_comm_sweep(
    cfg_factory: Callable[[int], object],
    grid_factory: Callable[[], object],
    sweep_values: List[float],
    seeds: List[int],
    out_path: Path,
    key: str,
) -> None:
    rows: List[Dict[str, object]] = []
    for value in sweep_values:
        results: List[SwarmMetrics] = []
        for seed in seeds:
            cfg = cfg_factory(seed)
            grid = grid_factory()
            costmap = CostMap3D.from_grid(grid, cfg.inflation_radius)
            starts, goals = generate_swarm_tasks(
                grid,
                cfg.agents,
                seed,
                cfg.min_start_separation,
                cfg.min_goal_separation,
                cfg.min_goal_distance,
            )
            if key == "comm_dropout":
                cfg.swarm_params.comm_dropout = float(value)
            else:
                cfg.swarm_params.comm_delay_steps = int(value)
            metrics = _run_once(cfg, grid, costmap, starts, goals)
            results.append(metrics)
        agg = _aggregate_metrics(results)
        rows.append(
            {
                key: value,
                **agg,
            }
        )
    _write_csv(out_path, rows)


def _run_horizon_sweep(
    cfg_factory: Callable[[int], object],
    grid_factory: Callable[[], object],
    horizons: List[int],
    seeds: List[int],
    out_path: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    for horizon in horizons:
        results: List[SwarmMetrics] = []
        for seed in seeds:
            cfg = cfg_factory(seed)
            cfg.prediction_enabled = horizon > 0
            cfg.prediction_horizon_steps = max(1, horizon)
            grid = grid_factory()
            costmap = CostMap3D.from_grid(grid, cfg.inflation_radius)
            starts, goals = generate_swarm_tasks(
                grid,
                cfg.agents,
                seed,
                cfg.min_start_separation,
                cfg.min_goal_separation,
                cfg.min_goal_distance,
            )
            metrics = _run_once(cfg, grid, costmap, starts, goals)
            results.append(metrics)
        agg = _aggregate_metrics(results)
        rows.append(
            {
                "prediction_horizon": horizon,
                **agg,
            }
        )
    _write_csv(out_path, rows)


def _run_mode_compare(
    cfg_factory: Callable[[int], object],
    grid_factory: Callable[[], object],
    modes: List[str],
    counts: List[int],
    seeds: List[int],
    out_path: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    for count in counts:
        for mode in modes:
            results: List[SwarmMetrics] = []
            for seed in seeds:
                cfg = cfg_factory(seed)
                cfg.agents = count
                cfg.mode = mode
                cfg.dynamic_enabled = False
                cfg.prediction_enabled = False
                cfg.swarm_params.start_delay_steps = 0
                grid = grid_factory()
                costmap = CostMap3D.from_grid(grid, cfg.inflation_radius)
                starts, goals = generate_swarm_tasks(
                    grid,
                    cfg.agents,
                    seed,
                    cfg.min_start_separation,
                    cfg.min_goal_separation,
                    cfg.min_goal_distance,
                )
                metrics = _run_once(cfg, grid, costmap, starts, goals)
                results.append(metrics)
            agg = _aggregate_metrics(results)
            rows.append(
                {
                    "agents": count,
                    "mode": mode,
                    **agg,
                }
            )
    _write_csv(out_path, rows)


def _load_cfg(path: Path) -> object:
    return swarm_cli._load_config(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed swarm experiments.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    base_cfg = _load_cfg(Path("configs/swarm.yaml"))
    large_cfg = _load_cfg(Path("configs/swarm_large.yaml"))

    def base_cfg_factory(seed: int):
        cfg = copy.deepcopy(base_cfg)
        cfg.seed = seed
        cfg.swarm_params.comm_seed = seed
        return cfg

    def large_cfg_factory(seed: int):
        cfg = copy.deepcopy(large_cfg)
        cfg.seed = seed
        cfg.swarm_params.comm_seed = seed
        return cfg

    REPORTS.mkdir(exist_ok=True)
    _run_sweep(
        base_cfg_factory,
        demo_grid,
        [4, 8, 12, 16, 20],
        seeds,
        REPORTS / "scale_sweep_agg.csv",
    )

    def dynamic_cfg_factory(seed: int):
        cfg = base_cfg_factory(seed)
        cfg.dynamic_enabled = True
        cfg.prediction_enabled = False
        return cfg

    def dynamic_pred_cfg_factory(seed: int):
        cfg = base_cfg_factory(seed)
        cfg.dynamic_enabled = True
        cfg.prediction_enabled = True
        return cfg

    _run_sweep(
        dynamic_cfg_factory,
        demo_grid,
        [4, 8, 12, 16, 20],
        seeds,
        REPORTS / "dynamic_sweep_agg.csv",
    )
    _run_sweep(
        dynamic_pred_cfg_factory,
        demo_grid,
        [4, 8, 12, 16, 20],
        seeds,
        REPORTS / "dynamic_sweep_predictive_agg.csv",
    )

    def comm_cfg_factory(seed: int):
        cfg = base_cfg_factory(seed)
        cfg.agents = 8
        return cfg

    _run_comm_sweep(
        comm_cfg_factory,
        demo_grid,
        [0.0, 0.2, 0.4, 0.6, 0.8],
        seeds,
        REPORTS / "comm_dropout_agg.csv",
        "comm_dropout",
    )
    _run_comm_sweep(
        comm_cfg_factory,
        demo_grid,
        [0, 3, 6, 9, 12],
        seeds,
        REPORTS / "comm_delay_agg.csv",
        "comm_delay_steps",
    )

    def horizon_cfg_factory(seed: int):
        cfg = base_cfg_factory(seed)
        cfg.dynamic_enabled = True
        cfg.agents = 8
        return cfg

    _run_horizon_sweep(
        horizon_cfg_factory,
        demo_grid,
        [0, 1, 3, 5, 8],
        seeds,
        REPORTS / "predict_horizon_agg.csv",
    )

    _run_sweep(
        large_cfg_factory,
        large_demo_grid,
        [30, 40, 50],
        seeds[:3],
        REPORTS / "large_scale_agg.csv",
    )

    _run_mode_compare(
        base_cfg_factory,
        demo_grid,
        ["distributed", "prioritized", "cooperative", "cbs"],
        [4, 8, 12],
        seeds[:3],
        REPORTS / "baseline_compare_agg.csv",
    )

    print("Aggregated experiments written to reports/.")


if __name__ == "__main__":
    main()
