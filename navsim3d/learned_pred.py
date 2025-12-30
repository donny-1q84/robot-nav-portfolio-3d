from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

Point3D = Tuple[float, float, float]


@dataclass
class MLPModel:
    history_steps: int
    scale: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0.0, x @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2


class LearnedObstaclePredictor:
    def __init__(self, model: MLPModel, horizon_steps: int) -> None:
        self.model = model
        self.horizon_steps = max(1, int(horizon_steps))

    @property
    def history_steps(self) -> int:
        return self.model.history_steps

    def predict_next(self, history: List[Point3D]) -> Point3D:
        if len(history) < self.model.history_steps:
            raise ValueError("Insufficient history for prediction.")
        window = history[-self.model.history_steps :]
        inputs = np.array(window, dtype=float)
        inputs = inputs / self.model.scale
        flat = inputs.reshape(1, -1)
        prediction = self.model.predict(flat)[0] * self.model.scale
        prediction = np.clip(prediction, 0.0, self.model.scale)
        return float(prediction[0]), float(prediction[1]), float(prediction[2])

    def predict_horizon(
        self, history: List[Point3D], horizon_steps: int | None = None
    ) -> List[Point3D]:
        if len(history) < self.model.history_steps:
            return []
        horizon = self.horizon_steps if horizon_steps is None else max(1, horizon_steps)
        window = list(history[-self.model.history_steps :])
        predictions: List[Point3D] = []
        for _ in range(horizon):
            nxt = self.predict_next(window)
            predictions.append(nxt)
            window.append(nxt)
            window = window[-self.model.history_steps :]
        return predictions


def save_model(path: Path, model: MLPModel) -> None:
    data = {
        "history_steps": model.history_steps,
        "scale": model.scale.tolist(),
        "w1": model.w1.tolist(),
        "b1": model.b1.tolist(),
        "w2": model.w2.tolist(),
        "b2": model.b2.tolist(),
    }
    path.write_text(json.dumps(data, indent=2))


def load_model(path: Path) -> MLPModel:
    data = json.loads(path.read_text())
    scale = np.array(data["scale"], dtype=float)
    return MLPModel(
        history_steps=int(data["history_steps"]),
        scale=scale,
        w1=np.array(data["w1"], dtype=float),
        b1=np.array(data["b1"], dtype=float),
        w2=np.array(data["w2"], dtype=float),
        b2=np.array(data["b2"], dtype=float),
    )


def load_predictor(
    path: Path,
    horizon_steps: int,
    expected_history_steps: int | None = None,
) -> LearnedObstaclePredictor:
    model = load_model(path)
    if expected_history_steps is not None and model.history_steps != expected_history_steps:
        raise ValueError(
            f"Model history_steps={model.history_steps} does not match expected "
            f"{expected_history_steps}."
        )
    return LearnedObstaclePredictor(model, horizon_steps)
