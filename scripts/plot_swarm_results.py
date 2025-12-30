from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))


def _to_number(value: str) -> Any:
    raw = value.strip()
    if raw == "":
        return None
    try:
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)
    except ValueError:
        return value


def read_csv(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({k: _to_number(v) for k, v in row.items()})
    return rows


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _extract_metric(
    rows: List[dict],
    mean_key: str,
    ci_key: str,
    fallback_key: str,
) -> Tuple[List[float], List[float] | None]:
    if not rows:
        return [], None
    if mean_key in rows[0]:
        values = [_safe_float(row.get(mean_key)) for row in rows]
        errors = [_safe_float(row.get(ci_key)) for row in rows]
        return values, errors
    if fallback_key in rows[0]:
        values = [_safe_float(row.get(fallback_key)) for row in rows]
        return values, None
    return [], None


def _plot_line(
    ax: plt.Axes,
    xs: List[float],
    ys: List[float],
    errs: List[float] | None,
    **kwargs,
) -> None:
    if errs is not None and any(errs):
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, **kwargs)
    else:
        ax.plot(xs, ys, marker="o", **kwargs)


def plot_scale_sweep(rows: List[dict], out_path: Path) -> None:
    agents = [row["agents"] for row in rows]
    success, success_ci = _extract_metric(rows, "success_mean", "success_ci", "success")
    min_sep, min_sep_ci = _extract_metric(
        rows, "min_separation_mean", "min_separation_ci", "min_separation"
    )
    collisions, collisions_ci = _extract_metric(
        rows, "collision_steps_mean", "collision_steps_ci", "collision_steps"
    )

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    _plot_line(axes[0], agents, success, success_ci, color="#1f77b4")
    axes[0].set_title("Success Rate vs Agents")
    axes[0].set_xlabel("Agents")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3)

    _plot_line(axes[1], agents, min_sep, min_sep_ci, color="#ff7f0e")
    axes[1].set_title("Min Separation vs Agents")
    axes[1].set_xlabel("Agents")
    axes[1].set_ylabel("Min Separation")
    axes[1].grid(True, alpha=0.3)

    _plot_line(axes[2], agents, collisions, collisions_ci, color="#2ca02c")
    axes[2].set_title("Collision Steps vs Agents")
    axes[2].set_xlabel("Agents")
    axes[2].set_ylabel("Collision Steps")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_comm_sweep(
    rows: List[dict], x_key: str, x_label: str, out_path: Path
) -> None:
    rows = sorted(rows, key=lambda row: row[x_key])
    xs = [row[x_key] for row in rows]
    min_sep, min_sep_ci = _extract_metric(
        rows, "min_separation_mean", "min_separation_ci", "min_separation"
    )
    neighbors, neighbors_ci = _extract_metric(
        rows, "mean_neighbors_mean", "mean_neighbors_ci", "mean_neighbors"
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    _plot_line(axes[0], xs, min_sep, min_sep_ci, color="#1f77b4")
    axes[0].set_title("Min Separation")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Min Separation")
    axes[0].grid(True, alpha=0.3)

    _plot_line(axes[1], xs, neighbors, neighbors_ci, color="#d62728")
    axes[1].set_title("Mean Neighbors")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Mean Neighbors")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_dynamic_sweep(rows: List[dict], out_path: Path) -> None:
    agents = [row["agents"] for row in rows]
    dyn_collisions, dyn_ci = _extract_metric(
        rows,
        "dynamic_collision_steps_mean",
        "dynamic_collision_steps_ci",
        "dynamic_collision_steps",
    )
    min_sep, min_sep_ci = _extract_metric(
        rows, "min_separation_mean", "min_separation_ci", "min_separation"
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    _plot_line(axes[0], agents, dyn_collisions, dyn_ci, color="#9467bd")
    axes[0].set_title("Dynamic Collision Steps")
    axes[0].set_xlabel("Agents")
    axes[0].set_ylabel("Dynamic Collision Steps")
    axes[0].grid(True, alpha=0.3)

    _plot_line(axes[1], agents, min_sep, min_sep_ci, color="#8c564b")
    axes[1].set_title("Min Separation")
    axes[1].set_xlabel("Agents")
    axes[1].set_ylabel("Min Separation")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_dynamic_compare(
    base_rows: List[dict], pred_rows: List[dict], out_path: Path
) -> None:
    base_rows = sorted(base_rows, key=lambda row: row["agents"])
    pred_rows = sorted(pred_rows, key=lambda row: row["agents"])
    agents = [row["agents"] for row in base_rows]

    base_dyn, base_dyn_ci = _extract_metric(
        base_rows,
        "dynamic_collision_steps_mean",
        "dynamic_collision_steps_ci",
        "dynamic_collision_steps",
    )
    pred_dyn, pred_dyn_ci = _extract_metric(
        pred_rows,
        "dynamic_collision_steps_mean",
        "dynamic_collision_steps_ci",
        "dynamic_collision_steps",
    )
    base_sep, base_sep_ci = _extract_metric(
        base_rows, "min_separation_mean", "min_separation_ci", "min_separation"
    )
    pred_sep, pred_sep_ci = _extract_metric(
        pred_rows, "min_separation_mean", "min_separation_ci", "min_separation"
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    _plot_line(axes[0], agents, base_dyn, base_dyn_ci, color="#9467bd", label="Baseline")
    _plot_line(axes[0], agents, pred_dyn, pred_dyn_ci, color="#17becf", label="Predictive")
    axes[0].set_title("Dynamic Collision Steps")
    axes[0].set_xlabel("Agents")
    axes[0].set_ylabel("Dynamic Collision Steps")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    _plot_line(axes[1], agents, base_sep, base_sep_ci, color="#8c564b", label="Baseline")
    _plot_line(axes[1], agents, pred_sep, pred_sep_ci, color="#2ca02c", label="Predictive")
    axes[1].set_title("Min Separation")
    axes[1].set_xlabel("Agents")
    axes[1].set_ylabel("Min Separation")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_horizon_sweep(rows: List[dict], out_path: Path) -> None:
    rows = sorted(rows, key=lambda row: row["prediction_horizon"])
    horizons = [row["prediction_horizon"] for row in rows]
    success, success_ci = _extract_metric(rows, "success_mean", "success_ci", "success")
    dyn_collisions, dyn_ci = _extract_metric(
        rows,
        "dynamic_collision_steps_mean",
        "dynamic_collision_steps_ci",
        "dynamic_collision_steps",
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    _plot_line(axes[0], horizons, success, success_ci, color="#1f77b4")
    axes[0].set_title("Success Rate vs Horizon")
    axes[0].set_xlabel("Prediction Horizon")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3)

    _plot_line(axes[1], horizons, dyn_collisions, dyn_ci, color="#ff7f0e")
    axes[1].set_title("Dynamic Collision Steps")
    axes[1].set_xlabel("Prediction Horizon")
    axes[1].set_ylabel("Dynamic Collision Steps")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_mode_compare(rows: List[dict], out_path: Path) -> None:
    groups: dict[str, List[dict]] = {}
    for row in rows:
        mode = str(row.get("mode", "unknown"))
        groups.setdefault(mode, []).append(row)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for mode, mode_rows in groups.items():
        mode_rows = sorted(mode_rows, key=lambda row: row["agents"])
        agents = [row["agents"] for row in mode_rows]
        success, success_ci = _extract_metric(
            mode_rows, "success_mean", "success_ci", "success"
        )
        min_sep, min_sep_ci = _extract_metric(
            mode_rows, "min_separation_mean", "min_separation_ci", "min_separation"
        )
        steps_to_goal, steps_ci = _extract_metric(
            mode_rows,
            "mean_steps_to_goal_mean",
            "mean_steps_to_goal_ci",
            "mean_steps_to_goal",
        )
        label = mode.replace("_", " ").title()
        _plot_line(axes[0], agents, success, success_ci, label=label)
        _plot_line(axes[1], agents, min_sep, min_sep_ci, label=label)
        _plot_line(axes[2], agents, steps_to_goal, steps_ci, label=label)

    axes[0].set_title("Success Rate")
    axes[0].set_xlabel("Agents")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Min Separation")
    axes[1].set_xlabel("Agents")
    axes[1].set_ylabel("Min Separation")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_title("Mean Steps to Goal")
    axes[2].set_xlabel("Agents")
    axes[2].set_ylabel("Steps")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    scale_path = REPORTS / "scale_sweep_agg.csv"
    dropout_path = REPORTS / "comm_dropout_agg.csv"
    delay_path = REPORTS / "comm_delay_agg.csv"
    dynamic_path = REPORTS / "dynamic_sweep_agg.csv"
    predictive_path = REPORTS / "dynamic_sweep_predictive_agg.csv"
    horizon_path = REPORTS / "predict_horizon_agg.csv"
    large_scale_path = REPORTS / "large_scale_agg.csv"
    baseline_path = REPORTS / "baseline_compare_agg.csv"

    if not scale_path.exists():
        scale_path = REPORTS / "scale_sweep.csv"
    if not dropout_path.exists():
        dropout_path = REPORTS / "comm_dropout.csv"
    if not delay_path.exists():
        delay_path = REPORTS / "comm_delay.csv"
    if not dynamic_path.exists():
        dynamic_path = REPORTS / "dynamic_sweep.csv"
    if not predictive_path.exists():
        predictive_path = REPORTS / "dynamic_sweep_predictive.csv"

    if scale_path.exists():
        plot_scale_sweep(read_csv(scale_path), REPORTS / "scale_sweep.png")
    if dropout_path.exists():
        plot_comm_sweep(
            read_csv(dropout_path),
            "comm_dropout",
            "Dropout Rate",
            REPORTS / "comm_dropout.png",
        )
    if delay_path.exists():
        plot_comm_sweep(
            read_csv(delay_path),
            "comm_delay_steps",
            "Delay Steps",
            REPORTS / "comm_delay.png",
        )
    if dynamic_path.exists():
        plot_dynamic_sweep(read_csv(dynamic_path), REPORTS / "dynamic_sweep.png")
    if dynamic_path.exists() and predictive_path.exists():
        plot_dynamic_compare(
            read_csv(dynamic_path),
            read_csv(predictive_path),
            REPORTS / "dynamic_compare.png",
        )
    if horizon_path.exists():
        plot_horizon_sweep(read_csv(horizon_path), REPORTS / "predict_horizon.png")
    if large_scale_path.exists():
        plot_scale_sweep(read_csv(large_scale_path), REPORTS / "large_scale.png")
    if baseline_path.exists():
        plot_mode_compare(read_csv(baseline_path), REPORTS / "baseline_compare.png")


if __name__ == "__main__":
    main()
