from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt

from .map3d import GridMap3D

Point3D = Tuple[float, float, float]


def _occupied_points(grid: List[List[List[int]]]) -> List[Point3D]:
    points: List[Point3D] = []
    for z, layer in enumerate(grid):
        for y, row in enumerate(layer):
            for x, cell in enumerate(row):
                if cell == 1:
                    points.append((float(x), float(y), float(z)))
    return points


def _draw_scene(
    ax,
    grid: GridMap3D,
    path: List[Point3D],
    poses: List[Point3D],
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    display_grid: List[List[List[int]]] | None = None,
    dynamic_positions: List[Point3D] | None = None,
) -> None:
    grid_data = display_grid if display_grid is not None else grid.grid
    obstacles = _occupied_points(grid_data)
    if obstacles:
        ox = [p[0] for p in obstacles]
        oy = [p[1] for p in obstacles]
        oz = [p[2] for p in obstacles]
        ax.scatter(ox, oy, oz, c="black", s=12, alpha=0.3, label="Obstacles")

    if dynamic_positions:
        dx = [p[0] for p in dynamic_positions]
        dy = [p[1] for p in dynamic_positions]
        dz = [p[2] for p in dynamic_positions]
        ax.scatter(dx, dy, dz, c="#17becf", s=50, alpha=0.9, label="Dynamic")

    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [p[2] for p in path]
        ax.plot(xs, ys, zs, color="#1f77b4", linewidth=2, label="Path")

    if poses:
        tx = [p[0] for p in poses]
        ty = [p[1] for p in poses]
        tz = [p[2] for p in poses]
        ax.plot(tx, ty, tz, color="#ff7f0e", linewidth=2, label="Trajectory")

    ax.scatter([start[0]], [start[1]], [start[2]], color="#2ca02c", s=80, label="Start")
    ax.scatter([goal[0]], [goal[1]], [goal[2]], color="#d62728", s=80, label="Goal")

    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(-0.5, grid.height - 0.5)
    ax.set_zlim(-0.5, grid.depth - 0.5)
    try:
        ax.set_box_aspect((grid.width, grid.height, grid.depth))
    except AttributeError:
        pass
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=25, azim=-60)
    ax.legend(loc="upper left")


def plot_scene(
    grid: GridMap3D,
    path: List[Point3D],
    poses: List[Point3D],
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    out_path: str,
    display_grid: List[List[List[int]]] | None = None,
    dynamic_positions: List[Point3D] | None = None,
) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    _draw_scene(
        ax,
        grid,
        path,
        poses,
        start,
        goal,
        display_grid,
        dynamic_positions,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_swarm_scene(
    grid: GridMap3D,
    paths: List[List[Point3D]],
    trajectories: List[List[Point3D]],
    starts: List[Tuple[int, int, int]],
    goals: List[Tuple[int, int, int]],
    out_path: str,
    display_grid: List[List[List[int]]] | None = None,
) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    _draw_swarm_scene(
        ax,
        grid,
        paths,
        trajectories,
        starts,
        goals,
        display_grid,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_swarm_gif(
    grid: GridMap3D,
    paths: List[List[Point3D]],
    trajectories: List[List[Point3D]],
    starts: List[Tuple[int, int, int]],
    goals: List[Tuple[int, int, int]],
    out_path: str,
    step: int = 4,
    display_grid: List[List[List[int]]] | None = None,
) -> None:
    import imageio.v2 as imageio

    frames = []
    max_len = max((len(traj) for traj in trajectories), default=0)
    for i in range(1, max_len + 1, step):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        partial_trajs = [traj[: min(i, len(traj))] for traj in trajectories]
        _draw_swarm_scene(
            ax,
            grid,
            paths,
            partial_trajs,
            starts,
            goals,
            display_grid,
        )
        fig.tight_layout()
        fig.canvas.draw()
        frame = imageio.imread(fig.canvas.buffer_rgba())
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(out_path, frames, fps=12)


def _draw_swarm_scene(
    ax,
    grid: GridMap3D,
    paths: List[List[Point3D]],
    trajectories: List[List[Point3D]],
    starts: List[Tuple[int, int, int]],
    goals: List[Tuple[int, int, int]],
    display_grid: List[List[List[int]]] | None = None,
) -> None:
    grid_data = display_grid if display_grid is not None else grid.grid
    obstacles = _occupied_points(grid_data)
    if obstacles:
        ox = [p[0] for p in obstacles]
        oy = [p[1] for p in obstacles]
        oz = [p[2] for p in obstacles]
        ax.scatter(ox, oy, oz, c="black", s=10, alpha=0.25, label="Obstacles")

    if starts:
        ax.scatter(
            [p[0] for p in starts],
            [p[1] for p in starts],
            [p[2] for p in starts],
            color="#2ca02c",
            s=40,
            label="Starts",
        )
    if goals:
        ax.scatter(
            [p[0] for p in goals],
            [p[1] for p in goals],
            [p[2] for p in goals],
            color="#d62728",
            s=40,
            label="Goals",
        )

    colors = plt.cm.tab20.colors
    for i, (path, poses) in enumerate(zip(paths, trajectories)):
        color = colors[i % len(colors)]
        if path:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            zs = [p[2] for p in path]
            ax.plot(
                xs,
                ys,
                zs,
                color=color,
                linewidth=1.2,
                alpha=0.6,
                linestyle="--",
                label="Path" if i == 0 else None,
            )
        if poses:
            tx = [p[0] for p in poses]
            ty = [p[1] for p in poses]
            tz = [p[2] for p in poses]
            ax.plot(
                tx,
                ty,
                tz,
                color=color,
                linewidth=1.8,
                label="Trajectory" if i == 0 else None,
            )

    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(-0.5, grid.height - 0.5)
    ax.set_zlim(-0.5, grid.depth - 0.5)
    try:
        ax.set_box_aspect((grid.width, grid.height, grid.depth))
    except AttributeError:
        pass
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=25, azim=-60)
    ax.legend(loc="upper left")


def render_gif(
    grid: GridMap3D,
    path: List[Point3D],
    poses: List[Point3D],
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    out_path: str,
    step: int = 4,
    display_grid: List[List[List[int]]] | None = None,
    dynamic_history: List[List[Point3D]] | None = None,
) -> None:
    import imageio.v2 as imageio

    frames = []
    for i in range(1, len(poses) + 1, step):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        dynamic_positions = None
        if dynamic_history:
            history_idx = min(i - 1, len(dynamic_history) - 1)
            dynamic_positions = dynamic_history[history_idx]
        _draw_scene(
            ax,
            grid,
            path,
            poses[:i],
            start,
            goal,
            display_grid,
            dynamic_positions,
        )
        fig.tight_layout()
        fig.canvas.draw()
        frame = imageio.imread(fig.canvas.buffer_rgba())
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(out_path, frames, fps=12)
