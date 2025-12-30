from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

Point3D = Tuple[float, float, float]
Pose3D = Tuple[float, float, float, float, float]


@dataclass
class UAVParams:
    lookahead: float = 1.2
    speed: float = 1.0
    yaw_rate_max: float = 1.5
    pitch_rate_max: float = 1.0
    max_pitch: float = 0.9
    yaw_gain: float = 1.2
    pitch_gain: float = 1.0


def _distance(a: Point3D, b: Point3D) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def guidance_command(
    pose: Pose3D,
    path: List[Point3D],
    params: UAVParams,
    last_target_idx: int,
) -> Tuple[float, float, float, int]:
    if not path:
        return 0.0, 0.0, 0.0, last_target_idx

    x, y, z, yaw, pitch = pose
    target_idx = last_target_idx
    for i in range(last_target_idx, len(path)):
        if _distance((x, y, z), path[i]) >= params.lookahead:
            target_idx = i
            break
    else:
        target_idx = len(path) - 1

    tx, ty, tz = path[target_idx]
    dx = tx - x
    dy = ty - y
    dz = tz - z
    horiz = math.hypot(dx, dy)
    desired_yaw = math.atan2(dy, dx) if horiz > 1e-9 else yaw
    desired_pitch = math.atan2(dz, max(horiz, 1e-9))
    desired_pitch = _clamp(desired_pitch, -params.max_pitch, params.max_pitch)

    yaw_error = _wrap_angle(desired_yaw - yaw)
    pitch_error = desired_pitch - pitch

    yaw_rate = _clamp(
        params.yaw_gain * yaw_error, -params.yaw_rate_max, params.yaw_rate_max
    )
    pitch_rate = _clamp(
        params.pitch_gain * pitch_error, -params.pitch_rate_max, params.pitch_rate_max
    )

    alignment = max(abs(yaw_error), abs(pitch_error))
    speed_scale = max(0.2, math.cos(min(math.pi / 2.0, alignment)))
    speed = params.speed * speed_scale

    return speed, yaw_rate, pitch_rate, target_idx
