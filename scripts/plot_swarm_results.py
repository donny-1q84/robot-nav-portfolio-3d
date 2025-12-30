from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


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


def plot_scale_sweep(rows: List[dict], out_path: Path) -> None:
    agents = [row["agents"] for row in rows]
    success = [float(row["success"]) for row in rows]
    min_sep = [row["min_separation"] for row in rows]
    collisions = [row["collision_steps"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    axes[0].plot(agents, success, marker="o", color="#1f77b4")
    axes[0].set_title("Success Rate vs Agents")
    axes[0].set_xlabel("Agents")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(agents, min_sep, marker="o", color="#ff7f0e")
    axes[1].set_title("Min Separation vs Agents")
    axes[1].set_xlabel("Agents")
    axes[1].set_ylabel("Min Separation")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(agents, collisions, marker="o", color="#2ca02c")
    axes[2].set_title("Collision Steps vs Agents")
    axes[2].set_xlabel("Agents")
    axes[2].set_ylabel("Collision Steps")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_comm_sweep(
    rows: List[dict], x_key: str, x_label: str, out_path: Path
) -> None:
    rows = sorted(rows, key=lambda row: row[x_key])
    xs = [row[x_key] for row in rows]
    min_sep = [row["min_separation"] for row in rows]
    neighbors = [row["mean_neighbors"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    axes[0].plot(xs, min_sep, marker="o", color="#1f77b4")
    axes[0].set_title("Min Separation")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Min Separation")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, neighbors, marker="o", color="#d62728")
    axes[1].set_title("Mean Neighbors")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Mean Neighbors")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_dynamic_sweep(rows: List[dict], out_path: Path) -> None:
    agents = [row["agents"] for row in rows]
    dyn_collisions = [row["dynamic_collision_steps"] for row in rows]
    min_sep = [row["min_separation"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    axes[0].plot(agents, dyn_collisions, marker="o", color="#9467bd")
    axes[0].set_title("Dynamic Collision Steps")
    axes[0].set_xlabel("Agents")
    axes[0].set_ylabel("Dynamic Collision Steps")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(agents, min_sep, marker="o", color="#8c564b")
    axes[1].set_title("Min Separation")
    axes[1].set_xlabel("Agents")
    axes[1].set_ylabel("Min Separation")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_dynamic_compare(
    base_rows: List[dict], pred_rows: List[dict], out_path: Path
) -> None:
    base_rows = sorted(base_rows, key=lambda row: row["agents"])
    pred_rows = sorted(pred_rows, key=lambda row: row["agents"])
    agents = [row["agents"] for row in base_rows]
    base_dyn = [row["dynamic_collision_steps"] for row in base_rows]
    pred_dyn = [row["dynamic_collision_steps"] for row in pred_rows]
    base_sep = [row["min_separation"] for row in base_rows]
    pred_sep = [row["min_separation"] for row in pred_rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    axes[0].plot(agents, base_dyn, marker="o", color="#9467bd", label="Baseline")
    axes[0].plot(agents, pred_dyn, marker="o", color="#17becf", label="Predictive")
    axes[0].set_title("Dynamic Collision Steps")
    axes[0].set_xlabel("Agents")
    axes[0].set_ylabel("Dynamic Collision Steps")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(agents, base_sep, marker="o", color="#8c564b", label="Baseline")
    axes[1].plot(agents, pred_sep, marker="o", color="#2ca02c", label="Predictive")
    axes[1].set_title("Min Separation")
    axes[1].set_xlabel("Agents")
    axes[1].set_ylabel("Min Separation")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    scale_path = REPORTS / "scale_sweep.csv"
    dropout_path = REPORTS / "comm_dropout.csv"
    delay_path = REPORTS / "comm_delay.csv"
    dynamic_path = REPORTS / "dynamic_sweep.csv"
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


if __name__ == "__main__":
    main()
