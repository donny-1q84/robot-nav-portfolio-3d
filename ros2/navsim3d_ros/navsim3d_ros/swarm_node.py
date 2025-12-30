from __future__ import annotations

import colorsys
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

try:
    from navsim3d.costmap3d import CostMap3D
    from navsim3d.map3d import demo_grid, large_demo_grid
    from navsim3d.swarm3d import generate_swarm_tasks
    from navsim3d import swarm_cli
except ImportError:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root))
    from navsim3d.costmap3d import CostMap3D
    from navsim3d.map3d import demo_grid, large_demo_grid
    from navsim3d.swarm3d import generate_swarm_tasks
    from navsim3d import swarm_cli


class SwarmVizNode(Node):
    def __init__(self) -> None:
        super().__init__("navsim3d_swarm_viz")
        self.declare_parameter("config", "configs/swarm.yaml")
        self.declare_parameter("map", "")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_rate", 1.0)

        config_value = self.get_parameter("config").get_parameter_value().string_value
        map_override = self.get_parameter("map").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        publish_rate = self.get_parameter("publish_rate").get_parameter_value().double_value
        publish_rate = publish_rate if publish_rate > 0.0 else 1.0

        self._markers = self._build_scene(Path(config_value), map_override)
        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._publisher = self.create_publisher(
            MarkerArray, "navsim3d/swarm_markers", qos
        )
        self._timer = self.create_timer(1.0 / publish_rate, self._publish)
        self._publish()

    def _build_scene(self, config_path: Path, map_override: str) -> MarkerArray:
        cfg = swarm_cli._load_config(config_path)
        if map_override:
            cfg.map_name = map_override

        grid = large_demo_grid() if cfg.map_name == "large" else demo_grid()
        costmap = CostMap3D.from_grid(grid, cfg.inflation_radius)

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

        paths, trajectories, metrics = swarm_cli._run_swarm(
            grid,
            costmap,
            starts,
            goals,
            cfg,
        )
        success_rate = metrics.reached / metrics.total if metrics.total else 0.0
        self.get_logger().info(
            "Swarm metrics: agents=%d success=%.2f min_sep=%.2f",
            metrics.total,
            success_rate,
            metrics.min_separation,
        )

        markers: List[Marker] = []
        marker_id = 0
        markers.append(self._make_obstacle_marker(grid, marker_id))
        marker_id += 1
        markers.append(
            self._make_point_marker(
                starts,
                marker_id,
                "starts",
                (0.1, 0.8, 0.1, 0.9),
                Marker.SPHERE_LIST,
            )
        )
        marker_id += 1
        markers.append(
            self._make_point_marker(
                goals,
                marker_id,
                "goals",
                (0.9, 0.1, 0.1, 0.9),
                Marker.SPHERE_LIST,
            )
        )
        marker_id += 1

        if cfg.dynamic_enabled and cfg.dynamic_obstacles:
            obstacles = [
                (obs.x, obs.y, obs.z) for obs in cfg.dynamic_obstacles
            ]
            markers.append(
                self._make_point_marker(
                    obstacles,
                    marker_id,
                    "dynamic_obstacles",
                    (0.5, 0.5, 0.5, 0.7),
                    Marker.SPHERE_LIST,
                )
            )
            marker_id += 1

        total_agents = len(paths)
        for idx, path in enumerate(paths):
            color = _agent_color(idx, total_agents)
            markers.append(
                self._make_line_marker(
                    path,
                    marker_id,
                    "paths",
                    (*color, 0.6),
                )
            )
            marker_id += 1

        for idx, traj in enumerate(trajectories):
            color = _agent_color(idx, total_agents)
            markers.append(
                self._make_line_marker(
                    traj,
                    marker_id,
                    "trajectories",
                    (*color, 0.95),
                    scale=0.06,
                )
            )
            marker_id += 1

        return MarkerArray(markers=markers)

    def _make_obstacle_marker(self, grid, marker_id: int) -> Marker:
        points = _points_from_cells(_occupied_cells(grid))
        marker = Marker()
        marker.ns = "obstacles"
        marker.id = marker_id
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.9
        marker.scale.y = 0.9
        marker.scale.z = 0.9
        marker.color.r = 0.4
        marker.color.g = 0.4
        marker.color.b = 0.4
        marker.color.a = 0.5
        marker.points = points
        return marker

    def _make_point_marker(
        self,
        points: Iterable[Tuple[float, float, float]],
        marker_id: int,
        namespace: str,
        rgba: Tuple[float, float, float, float],
        marker_type: int,
    ) -> Marker:
        marker = Marker()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]
        marker.points = _points_from_cells(points)
        return marker

    def _make_line_marker(
        self,
        points: Iterable[Tuple[float, float, float]],
        marker_id: int,
        namespace: str,
        rgba: Tuple[float, float, float, float],
        scale: float = 0.04,
    ) -> Marker:
        marker = Marker()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]
        marker.points = _points_from_cells(points)
        return marker

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for marker in self._markers.markers:
            marker.header.frame_id = self.frame_id
            marker.header.stamp = stamp
        self._publisher.publish(self._markers)


def _occupied_cells(grid) -> List[Tuple[int, int, int]]:
    occupied: List[Tuple[int, int, int]] = []
    for z in range(grid.depth):
        for y in range(grid.height):
            for x in range(grid.width):
                if not grid.is_free((x, y, z)):
                    occupied.append((x, y, z))
    return occupied


def _points_from_cells(
    cells: Iterable[Tuple[float, float, float]]
) -> List[Point]:
    points: List[Point] = []
    for x, y, z in cells:
        points.append(Point(x=float(x), y=float(y), z=float(z)))
    return points


def _agent_color(index: int, total: int) -> Tuple[float, float, float]:
    if total <= 0:
        return 0.2, 0.6, 0.9
    hue = (index / total) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
    return r, g, b


def main() -> None:
    rclpy.init()
    node = SwarmVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
