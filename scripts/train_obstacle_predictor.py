from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from navsim3d.dynamic3d import DynamicObstacle3D
from navsim3d.learned_pred import MLPModel, save_model
from navsim3d.map3d import demo_grid


def _sample_obstacle(rng: np.random.Generator, grid) -> DynamicObstacle3D:
    while True:
        x = rng.uniform(0.0, grid.width - 1)
        y = rng.uniform(0.0, grid.height - 1)
        z = rng.uniform(0.0, grid.depth - 1)
        cell = (int(round(x)), int(round(y)), int(round(z)))
        if grid.is_free(cell):
            break

    def _vel() -> float:
        val = rng.uniform(-0.8, 0.8)
        if abs(val) < 0.1:
            val = 0.2 if val >= 0 else -0.2
        return val

    return DynamicObstacle3D(x=x, y=y, z=z, vx=_vel(), vy=_vel(), vz=_vel())


def _generate_dataset(
    rng: np.random.Generator,
    grid,
    samples: int,
    history_steps: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.zeros((samples, history_steps * 3), dtype=float)
    targets = np.zeros((samples, 3), dtype=float)

    for i in range(samples):
        obstacle = _sample_obstacle(rng, grid)
        positions = []
        for _ in range(history_steps + 1):
            positions.append((obstacle.x, obstacle.y, obstacle.z))
            obstacle.step(dt, grid)
        inputs[i] = np.array(positions[:history_steps], dtype=float).reshape(-1)
        targets[i] = np.array(positions[history_steps], dtype=float)

    return inputs, targets


def _train_mlp(
    inputs: np.ndarray,
    targets: np.ndarray,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    in_dim = inputs.shape[1]
    w1 = rng.normal(scale=0.1, size=(in_dim, hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(scale=0.1, size=(hidden, 3))
    b2 = np.zeros(3)

    count = inputs.shape[0]
    for epoch in range(epochs):
        indices = rng.permutation(count)
        for start in range(0, count, batch_size):
            batch = indices[start : start + batch_size]
            xb = inputs[batch]
            yb = targets[batch]

            hidden_act = xb @ w1 + b1
            hidden_relu = np.maximum(0.0, hidden_act)
            outputs = hidden_relu @ w2 + b2
            error = outputs - yb
            grad_out = 2.0 * error / len(batch)

            dw2 = hidden_relu.T @ grad_out
            db2 = grad_out.sum(axis=0)
            dhidden = grad_out @ w2.T
            dhidden[hidden_act <= 0.0] = 0.0
            dw1 = xb.T @ dhidden
            db1 = dhidden.sum(axis=0)

            w1 -= lr * dw1
            b1 -= lr * db1
            w2 -= lr * dw2
            b2 -= lr * db2

        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            hidden_act = inputs @ w1 + b1
            hidden_relu = np.maximum(0.0, hidden_act)
            outputs = hidden_relu @ w2 + b2
            loss = np.mean((outputs - targets) ** 2)
            print(f"epoch {epoch + 1:03d} loss {loss:.6f}")

    return w1, b1, w2, b2


def main() -> None:
    parser = argparse.ArgumentParser(description="Train obstacle MLP predictor.")
    parser.add_argument("--out", type=Path, default=Path("models/obstacle_mlp.json"))
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--horizon-steps", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    grid = demo_grid()
    rng = np.random.default_rng(args.seed)
    inputs, targets = _generate_dataset(
        rng, grid, args.samples, args.history_steps, dt=0.1
    )

    scale = np.array(
        [max(1.0, grid.width - 1), max(1.0, grid.height - 1), max(1.0, grid.depth - 1)],
        dtype=float,
    )
    input_scale = np.tile(scale, args.history_steps)
    inputs = inputs / input_scale
    targets = targets / scale

    w1, b1, w2, b2 = _train_mlp(
        inputs,
        targets,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    model = MLPModel(
        history_steps=args.history_steps,
        scale=scale,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_model(args.out, model)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
