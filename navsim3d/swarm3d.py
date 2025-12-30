from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple, TYPE_CHECKING

from .collision3d import point_in_collision
from .control3d import UAVParams
from .costmap3d import CostMap3D
from .map3d import GridMap3D
from .planner3d import plan_path

if TYPE_CHECKING:
    from .learned_pred import LearnedObstaclePredictor
Point3D = Tuple[float, float, float]
Cell3D = Tuple[int, int, int]
PoseState = Tuple[float, float, float, float, float]


@dataclass
class SwarmParams:
    dt: float = 0.1
    max_steps: int = 700
    goal_tolerance: float = 0.4
    min_separation: float = 0.8
    neighbor_radius: float = 2.5
    avoidance_gain: float = 1.2
    obstacle_avoidance_gain: float = 0.9
    obstacle_avoidance_radius: float = 2.5
    start_delay_steps: int = 3
    respect_obstacles: bool = False
    goal_hold_steps: int = 30
    comm_range: float = 0.0
    comm_delay_steps: int = 0
    comm_dropout: float = 0.0
    comm_max_neighbors: int = 0
    comm_seed: int = 0


@dataclass
class SwarmMetrics:
    reached: int
    total: int
    min_separation: float
    collision_steps: int
    dynamic_collision_steps: int
    mean_path_length: float
    mean_traj_length: float
    mean_steps_to_goal: float | None
    mean_neighbors: float
    steps: int


@dataclass(frozen=True)
class CBSConstraint:
    agent: int
    time: int
    cell: Cell3D | None = None
    edge: tuple[Cell3D, Cell3D] | None = None


@dataclass(frozen=True)
class CBSConflict:
    kind: str
    time: int
    agent_a: int
    agent_b: int
    cell: Cell3D | None = None
    edge_a: tuple[Cell3D, Cell3D] | None = None
    edge_b: tuple[Cell3D, Cell3D] | None = None


def plan_swarm_paths(
    costmap: CostMap3D,
    starts: List[Cell3D],
    goals: List[Cell3D],
    connectivity: int = 26,
) -> List[List[Point3D]]:
    if len(starts) != len(goals):
        raise ValueError("Starts/goals must have the same length.")

    paths: List[List[Point3D]] = []
    for idx, (start, goal) in enumerate(zip(starts, goals)):
        plan = plan_path(costmap.inflated_map(), start, goal, connectivity=connectivity)
        if plan is None:
            raise ValueError(f"No path found for agent {idx}.")
        paths.append(_grid_to_path(plan.path))
    return paths


def simulate_swarm(
    grid: GridMap3D,
    costmap: CostMap3D,
    paths: List[List[Point3D]],
    starts: List[Cell3D],
    goals: List[Cell3D],
    params: SwarmParams,
    ctrl_params: UAVParams,
    dynamic_field: "DynamicObstacleField3D | None" = None,
    predictor: "LearnedObstaclePredictor | None" = None,
) -> Tuple[List[List[Point3D]], SwarmMetrics]:
    agents = len(paths)
    states: List[PoseState] = []
    target_indices: List[int] = []
    reached_flags = [False] * agents
    goal_steps: List[int | None] = [None] * agents
    trajectories: List[List[Point3D]] = []
    rng = random.Random(params.comm_seed)
    obstacle_history: List[List[Point3D]] | None = None
    if dynamic_field is not None and predictor is not None:
        obstacle_history = [[] for _ in dynamic_field.obstacles]

    for path, start in zip(paths, starts):
        start_pose = (float(start[0]), float(start[1]), float(start[2]))
        yaw, pitch = _initial_orientation(path, start_pose)
        states.append((start_pose[0], start_pose[1], start_pose[2], yaw, pitch))
        target_indices.append(0)
        trajectories.append([start_pose])

    min_pair_distance = float("inf")
    collision_steps = 0
    dynamic_collision_steps = 0
    steps = 0
    neighbor_samples = 0

    for step in range(params.max_steps):
        dynamic_cells: set[Cell3D] = set()
        obstacle_positions: List[Point3D] = []
        if dynamic_field is not None:
            dynamic_field.step(params.dt, grid)
            dynamic_cells = set(dynamic_field.cells(grid))
            if predictor is not None and obstacle_history is not None:
                dynamic_positions = dynamic_field.positions()
                for idx, pos in enumerate(dynamic_positions):
                    history = obstacle_history[idx]
                    history.append(pos)
                    if len(history) > predictor.history_steps:
                        history.pop(0)
                predicted_positions: List[Point3D] = []
                for history in obstacle_history:
                    if len(history) >= predictor.history_steps:
                        predicted_positions.extend(predictor.predict_horizon(history))
                obstacle_positions = dynamic_positions + predicted_positions
        positions = [(state[0], state[1], state[2]) for state in states]
        step_min = _min_pair_distance(positions)
        min_pair_distance = min(min_pair_distance, step_min)
        if step_min < params.min_separation:
            collision_steps += 1
        if dynamic_cells and _any_dynamic_collision(positions, dynamic_cells):
            dynamic_collision_steps += 1

        new_states: List[PoseState] = []
        new_indices: List[int] = []

        for i in range(agents):
            state = states[i]
            if step < i * params.start_delay_steps:
                new_states.append(state)
                new_indices.append(target_indices[i])
                continue
            if not reached_flags[i]:
                gx, gy, gz = goals[i]
                if _distance((state[0], state[1], state[2]), (gx, gy, gz)) <= params.goal_tolerance:
                    reached_flags[i] = True
                    goal_steps[i] = step

            if reached_flags[i]:
                new_states.append(state)
                new_indices.append(target_indices[i])
                continue

            neighbors = _visible_neighbors(i, positions, trajectories, params, rng)
            neighbor_samples += len(neighbors)
            speed, yaw_rate, pitch_rate, target_idx, min_dist, direction = _swarm_guidance(
                state,
                paths[i],
                ctrl_params,
                target_indices[i],
                neighbors,
                obstacle_positions,
                params,
            )
            yaw = _wrap_angle(state[3] + yaw_rate * params.dt)
            pitch = _clamp(
                state[4] + pitch_rate * params.dt,
                -ctrl_params.max_pitch,
                ctrl_params.max_pitch,
            )

            vx = speed * math.cos(pitch) * math.cos(yaw)
            vy = speed * math.cos(pitch) * math.sin(yaw)
            vz = speed * math.sin(pitch)
            x = state[0] + vx * params.dt
            y = state[1] + vy * params.dt
            z = state[2] + vz * params.dt

            x = _clamp(x, 0.0, grid.width - 1.0)
            y = _clamp(y, 0.0, grid.height - 1.0)
            z = _clamp(z, 0.0, grid.depth - 1.0)

            if params.respect_obstacles and (
                point_in_collision(costmap, (x, y, z))
                or _point_in_cells((x, y, z), dynamic_cells)
            ):
                speed, yaw_rate, pitch_rate = _command_from_direction(
                    state, direction, ctrl_params
                )
                speed = _apply_neighbor_speed_scale(speed, min_dist, params)
                yaw = _wrap_angle(state[3] + yaw_rate * params.dt)
                pitch = _clamp(
                    state[4] + pitch_rate * params.dt,
                    -ctrl_params.max_pitch,
                    ctrl_params.max_pitch,
                )
                vx = speed * math.cos(pitch) * math.cos(yaw)
                vy = speed * math.cos(pitch) * math.sin(yaw)
                vz = speed * math.sin(pitch)
                x = _clamp(state[0] + vx * params.dt, 0.0, grid.width - 1.0)
                y = _clamp(state[1] + vy * params.dt, 0.0, grid.height - 1.0)
                z = _clamp(state[2] + vz * params.dt, 0.0, grid.depth - 1.0)
                if point_in_collision(costmap, (x, y, z)) or _point_in_cells(
                    (x, y, z), dynamic_cells
                ):
                    x, y, z = state[0], state[1], state[2]

            new_states.append((x, y, z, yaw, pitch))
            new_indices.append(target_idx)

        states = new_states
        target_indices = new_indices
        for i, state in enumerate(states):
            trajectories[i].append((state[0], state[1], state[2]))

        steps = step + 1
        if all(reached_flags):
            break

    mean_path = _mean_length(paths)
    mean_traj = _mean_length(trajectories)
    reached = sum(reached_flags)
    reached_steps = [s for s in goal_steps if s is not None]
    mean_steps_to_goal = (
        sum(reached_steps) / len(reached_steps) if reached_steps else None
    )
    mean_neighbors = (
        neighbor_samples / (steps * agents) if steps > 0 and agents > 0 else 0.0
    )
    metrics = SwarmMetrics(
        reached=reached,
        total=agents,
        min_separation=min_pair_distance,
        collision_steps=collision_steps,
        dynamic_collision_steps=dynamic_collision_steps,
        mean_path_length=mean_path,
        mean_traj_length=mean_traj,
        mean_steps_to_goal=mean_steps_to_goal,
        mean_neighbors=mean_neighbors,
        steps=steps,
    )
    return trajectories, metrics


def simulate_swarm_prioritized(
    grid: GridMap3D,
    paths: List[List[Point3D]],
    goals: List[Cell3D],
    params: SwarmParams,
    dynamic_field: "DynamicObstacleField3D | None" = None,
) -> Tuple[List[List[Point3D]], SwarmMetrics]:
    schedules = _build_prioritized_schedule(paths, params)
    metrics = _schedule_metrics(
        schedules,
        goals,
        params,
        grid,
        dynamic_field,
        mean_path_length=_mean_length(paths),
    )
    return schedules, metrics


def simulate_swarm_cooperative(
    grid: GridMap3D,
    costmap: CostMap3D,
    starts: List[Cell3D],
    goals: List[Cell3D],
    params: SwarmParams,
    connectivity: int,
    dynamic_field: "DynamicObstacleField3D | None" = None,
) -> Tuple[List[List[Point3D]], SwarmMetrics]:
    schedules = _build_cooperative_schedule(
        costmap, starts, goals, params, connectivity
    )
    metrics = _schedule_metrics(
        schedules,
        goals,
        params,
        grid,
        dynamic_field,
        mean_path_length=_mean_length(schedules),
    )
    return schedules, metrics


def simulate_swarm_cbs(
    grid: GridMap3D,
    costmap: CostMap3D,
    starts: List[Cell3D],
    goals: List[Cell3D],
    params: SwarmParams,
    connectivity: int,
    dynamic_field: "DynamicObstacleField3D | None" = None,
    max_expansions: int = 800,
) -> Tuple[List[List[Point3D]], SwarmMetrics]:
    schedules = _build_cbs_schedule(
        costmap,
        starts,
        goals,
        params,
        connectivity,
        max_expansions,
    )
    metrics = _schedule_metrics(
        schedules,
        goals,
        params,
        grid,
        dynamic_field,
        mean_path_length=_mean_length(schedules),
    )
    return schedules, metrics


def _build_cbs_schedule(
    costmap: CostMap3D,
    starts: List[Cell3D],
    goals: List[Cell3D],
    params: SwarmParams,
    connectivity: int,
    max_expansions: int,
) -> List[List[Point3D]]:
    if len(starts) != len(goals):
        raise ValueError("Starts/goals must have the same length.")

    grid = costmap.inflated_map()
    constraints: List[CBSConstraint] = []
    schedules: List[List[Cell3D]] = []

    for agent_idx, (start, goal) in enumerate(zip(starts, goals)):
        schedule = _cbs_plan_for_agent(
            grid,
            start,
            goal,
            constraints,
            agent_idx,
            params,
            connectivity,
        )
        if schedule is None:
            raise ValueError(f"No CBS path found for agent {agent_idx}.")
        schedules.append(schedule)

    root_cost = _schedule_cost(schedules)
    open_heap: list[tuple[int, int, List[CBSConstraint], List[List[Cell3D]]]] = []
    counter = 0
    heapq.heappush(open_heap, (root_cost, counter, constraints, schedules))

    expansions = 0
    while open_heap:
        _, _, node_constraints, node_schedules = heapq.heappop(open_heap)
        conflict = _find_first_conflict(node_schedules)
        if conflict is None:
            return [_cells_to_points(schedule) for schedule in node_schedules]

        expansions += 1
        if expansions > max_expansions:
            raise ValueError("CBS exceeded expansion budget.")

        for constraint in _constraints_from_conflict(conflict):
            new_constraints = list(node_constraints)
            new_constraints.append(constraint)
            new_schedules = list(node_schedules)
            agent_idx = constraint.agent
            schedule = _cbs_plan_for_agent(
                grid,
                starts[agent_idx],
                goals[agent_idx],
                new_constraints,
                agent_idx,
                params,
                connectivity,
            )
            if schedule is None:
                continue
            new_schedules[agent_idx] = schedule
            counter += 1
            cost = _schedule_cost(new_schedules)
            heapq.heappush(open_heap, (cost, counter, new_constraints, new_schedules))

    raise ValueError("CBS failed to resolve conflicts.")


def _constraints_from_conflict(conflict: CBSConflict) -> List[CBSConstraint]:
    if conflict.kind == "vertex":
        return [
            CBSConstraint(
                agent=conflict.agent_a,
                time=conflict.time,
                cell=conflict.cell,
            ),
            CBSConstraint(
                agent=conflict.agent_b,
                time=conflict.time,
                cell=conflict.cell,
            ),
        ]
    if conflict.kind == "edge":
        return [
            CBSConstraint(
                agent=conflict.agent_a,
                time=conflict.time,
                edge=conflict.edge_a,
            ),
            CBSConstraint(
                agent=conflict.agent_b,
                time=conflict.time,
                edge=conflict.edge_b,
            ),
        ]
    raise ValueError(f"Unknown conflict kind: {conflict.kind}")


def _cbs_plan_for_agent(
    grid: GridMap3D,
    start: Cell3D,
    goal: Cell3D,
    constraints: List[CBSConstraint],
    agent_idx: int,
    params: SwarmParams,
    connectivity: int,
) -> List[Cell3D] | None:
    start_time = agent_idx * params.start_delay_steps
    if start_time >= params.max_steps:
        return None

    vertex_constraints, edge_constraints = _constraint_sets(constraints, agent_idx)
    for t in range(start_time):
        if (start, t) in vertex_constraints:
            return None

    path_cells = _time_astar_constrained(
        grid,
        start,
        goal,
        vertex_constraints,
        edge_constraints,
        params.max_steps,
        connectivity,
        params.goal_hold_steps,
        start_time,
    )
    if path_cells is None:
        return None

    schedule: List[Cell3D] = [start] * start_time + path_cells
    for _ in range(params.goal_hold_steps):
        if len(schedule) >= params.max_steps:
            break
        schedule.append(schedule[-1])
    return schedule


def _constraint_sets(
    constraints: List[CBSConstraint],
    agent: int,
) -> tuple[set[tuple[Cell3D, int]], set[tuple[Cell3D, Cell3D, int]]]:
    vertex_constraints: set[tuple[Cell3D, int]] = set()
    edge_constraints: set[tuple[Cell3D, Cell3D, int]] = set()
    for constraint in constraints:
        if constraint.agent != agent:
            continue
        if constraint.cell is not None:
            vertex_constraints.add((constraint.cell, constraint.time))
        if constraint.edge is not None:
            edge_constraints.add(
                (constraint.edge[0], constraint.edge[1], constraint.time)
            )
    return vertex_constraints, edge_constraints


def _find_first_conflict(
    schedules: List[List[Cell3D]],
) -> CBSConflict | None:
    if not schedules:
        return None
    agents = len(schedules)
    max_steps = max((len(schedule) for schedule in schedules), default=0)

    for time_idx in range(max_steps):
        positions: List[Cell3D] = []
        prev_positions: List[Cell3D] = []
        for schedule in schedules:
            pos = schedule[time_idx] if time_idx < len(schedule) else schedule[-1]
            positions.append(pos)
            if time_idx == 0:
                prev_positions.append(pos)
            else:
                prev = (
                    schedule[time_idx - 1]
                    if time_idx - 1 < len(schedule)
                    else schedule[-1]
                )
                prev_positions.append(prev)

        for i in range(agents):
            for j in range(i + 1, agents):
                if positions[i] == positions[j]:
                    return CBSConflict(
                        kind="vertex",
                        time=time_idx,
                        agent_a=i,
                        agent_b=j,
                        cell=positions[i],
                    )
        if time_idx == 0:
            continue
        for i in range(agents):
            for j in range(i + 1, agents):
                if (
                    positions[i] == prev_positions[j]
                    and positions[j] == prev_positions[i]
                ):
                    return CBSConflict(
                        kind="edge",
                        time=time_idx,
                        agent_a=i,
                        agent_b=j,
                        edge_a=(prev_positions[i], positions[i]),
                        edge_b=(prev_positions[j], positions[j]),
                    )
    return None


def _schedule_cost(schedules: List[List[Cell3D]]) -> int:
    return sum(len(schedule) for schedule in schedules)


def _time_astar_constrained(
    grid: GridMap3D,
    start: Cell3D,
    goal: Cell3D,
    vertex_constraints: set[tuple[Cell3D, int]],
    edge_constraints: set[tuple[Cell3D, Cell3D, int]],
    max_steps: int,
    connectivity: int,
    goal_hold_steps: int,
    start_time: int,
) -> List[Cell3D] | None:
    if start_time >= max_steps:
        return None
    if (start, start_time) in vertex_constraints:
        return None

    start_state = (start, start_time)
    open_heap: list[tuple[float, Cell3D, int]] = []
    heapq.heappush(
        open_heap, (start_time + _distance(start, goal), start, start_time)
    )
    came_from: dict[tuple[Cell3D, int], tuple[Cell3D, int]] = {}
    g_cost: dict[tuple[Cell3D, int], int] = {start_state: start_time}

    offsets = _neighbor_offsets(connectivity)
    offsets.append((0, 0, 0))

    while open_heap:
        _, current, time_idx = heapq.heappop(open_heap)
        state = (current, time_idx)
        if current == goal and _goal_hold_clear_constraints(
            vertex_constraints, goal, time_idx, goal_hold_steps, max_steps
        ):
            return _reconstruct_time_path(came_from, state)

        if time_idx >= max_steps - 1:
            continue

        for dx, dy, dz in offsets:
            nxt = (current[0] + dx, current[1] + dy, current[2] + dz)
            next_time = time_idx + 1
            if not grid.in_bounds(nxt) or not grid.is_free(nxt):
                continue
            if (nxt, next_time) in vertex_constraints:
                continue
            if (current, nxt, next_time) in edge_constraints:
                continue

            nxt_state = (nxt, next_time)
            if nxt_state in g_cost and next_time >= g_cost[nxt_state]:
                continue
            g_cost[nxt_state] = next_time
            came_from[nxt_state] = state
            f_cost = next_time + _distance(nxt, goal)
            heapq.heappush(open_heap, (f_cost, nxt, next_time))

    return None


def _goal_hold_clear_constraints(
    vertex_constraints: set[tuple[Cell3D, int]],
    goal: Cell3D,
    time_idx: int,
    hold_steps: int,
    max_steps: int,
) -> bool:
    end_time = min(max_steps - 1, time_idx + hold_steps)
    for t in range(time_idx + 1, end_time + 1):
        if (goal, t) in vertex_constraints:
            return False
    return True


def generate_swarm_tasks(
    grid: GridMap3D,
    count: int,
    seed: int,
    min_start_separation: float,
    min_goal_separation: float,
    min_goal_distance: float,
) -> Tuple[List[Cell3D], List[Cell3D]]:
    rng = random.Random(seed)
    free_cells = _free_cells(grid)
    start_candidates, goal_candidates = _face_candidates(grid, free_cells)
    if len(start_candidates) < count or len(goal_candidates) < count:
        start_candidates = free_cells
        goal_candidates = free_cells

    starts = _select_cells(rng, start_candidates, count, min_start_separation)
    free_without_starts = [cell for cell in goal_candidates if cell not in set(starts)]
    goals = _select_goal_cells(
        rng,
        free_without_starts,
        starts,
        count,
        min_goal_separation,
        min_goal_distance,
    )
    return starts, goals


def _select_goal_cells(
    rng: random.Random,
    candidates: List[Cell3D],
    starts: List[Cell3D],
    count: int,
    min_separation: float,
    min_distance: float,
) -> List[Cell3D]:
    goals: List[Cell3D] = []
    attempts = 0
    max_attempts = count * 400
    while len(goals) < count and attempts < max_attempts:
        attempts += 1
        cell = rng.choice(candidates)
        if cell in goals:
            continue
        if any(_distance(cell, goal) < min_separation for goal in goals):
            continue
        start = starts[len(goals)]
        if _distance(cell, start) < min_distance:
            continue
        goals.append(cell)
    if len(goals) < count:
        raise ValueError("Failed to sample goals with the requested constraints.")
    return goals


def _select_cells(
    rng: random.Random,
    candidates: List[Cell3D],
    count: int,
    min_separation: float,
) -> List[Cell3D]:
    selected: List[Cell3D] = []
    attempts = 0
    max_attempts = count * 400
    while len(selected) < count and attempts < max_attempts:
        attempts += 1
        cell = rng.choice(candidates)
        if cell in selected:
            continue
        if any(_distance(cell, other) < min_separation for other in selected):
            continue
        selected.append(cell)
    if len(selected) < count:
        raise ValueError("Failed to sample cells with the requested constraints.")
    return selected


def _free_cells(grid: GridMap3D) -> List[Cell3D]:
    free: List[Cell3D] = []
    for z in range(grid.depth):
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.is_free((x, y, z)):
                    free.append((x, y, z))
    return free


def _build_prioritized_schedule(
    paths: List[List[Point3D]], params: SwarmParams
) -> List[List[Point3D]]:
    reserved_cells: dict[int, set[Cell3D]] = {}
    reserved_edges: set[tuple[Cell3D, Cell3D, int]] = set()
    schedules: List[List[Point3D]] = []

    for agent_idx, path in enumerate(paths):
        cell_path = [_to_cell(point) for point in path]
        if not cell_path:
            schedules.append([])
            continue
        schedule: List[Cell3D] = []
        current = cell_path[0]
        time_idx = 0

        for _ in range(agent_idx * params.start_delay_steps):
            schedule.append(current)
            _reserve_cell(reserved_cells, time_idx, current)
            time_idx += 1

        if not schedule:
            schedule.append(current)
            _reserve_cell(reserved_cells, 0, current)

        for nxt in cell_path[1:]:
            while time_idx < params.max_steps - 1:
                next_time = time_idx + 1
                if _cell_reserved(reserved_cells, nxt, next_time) or _edge_conflict(
                    reserved_edges, current, nxt, next_time
                ):
                    time_idx += 1
                    schedule.append(current)
                    _reserve_cell(reserved_cells, time_idx, current)
                    continue
                time_idx += 1
                _reserve_edge(reserved_edges, current, nxt, time_idx)
                current = nxt
                schedule.append(current)
                _reserve_cell(reserved_cells, time_idx, current)
                break
            if time_idx >= params.max_steps - 1:
                break

        for _ in range(params.goal_hold_steps):
            if time_idx >= params.max_steps - 1:
                break
            time_idx += 1
            schedule.append(current)
            _reserve_cell(reserved_cells, time_idx, current)

        schedules.append(_cells_to_points(schedule))

    return schedules


def _build_cooperative_schedule(
    costmap: CostMap3D,
    starts: List[Cell3D],
    goals: List[Cell3D],
    params: SwarmParams,
    connectivity: int,
) -> List[List[Point3D]]:
    if len(starts) != len(goals):
        raise ValueError("Starts/goals must have the same length.")

    grid = costmap.inflated_map()
    reserved_cells: dict[int, set[Cell3D]] = {}
    reserved_edges: set[tuple[Cell3D, Cell3D, int]] = set()
    schedules: List[List[Point3D]] = []

    for agent_idx, (start, goal) in enumerate(zip(starts, goals)):
        start_time = agent_idx * params.start_delay_steps
        if start_time >= params.max_steps:
            raise ValueError("Start delay exceeds max steps.")
        for t in range(start_time + 1):
            if _cell_reserved(reserved_cells, start, t):
                raise ValueError("Start cell reserved by previous agent.")

        path_cells = _time_astar(
            grid,
            start,
            goal,
            reserved_cells,
            reserved_edges,
            params.max_steps,
            connectivity,
            params.goal_hold_steps,
            start_time,
        )
        if path_cells is None:
            raise ValueError(f"No cooperative path found for agent {agent_idx}.")

        schedule_cells: List[Cell3D] = [start] * start_time + path_cells
        for _ in range(params.goal_hold_steps):
            if len(schedule_cells) >= params.max_steps:
                break
            schedule_cells.append(schedule_cells[-1])

        _reserve_schedule(reserved_cells, reserved_edges, schedule_cells)
        schedules.append(_cells_to_points(schedule_cells))

    return schedules


def _time_astar(
    grid: GridMap3D,
    start: Cell3D,
    goal: Cell3D,
    reserved_cells: dict[int, set[Cell3D]],
    reserved_edges: set[tuple[Cell3D, Cell3D, int]],
    max_steps: int,
    connectivity: int,
    goal_hold_steps: int,
    start_time: int,
) -> List[Cell3D] | None:
    if start_time >= max_steps:
        return None

    start_state = (start, start_time)
    open_heap: list[tuple[float, Cell3D, int]] = []
    heapq.heappush(
        open_heap, (_distance(start, goal) + start_time, start, start_time)
    )
    came_from: dict[tuple[Cell3D, int], tuple[Cell3D, int]] = {}
    g_cost: dict[tuple[Cell3D, int], int] = {start_state: start_time}

    offsets = _neighbor_offsets(connectivity)
    offsets.append((0, 0, 0))

    while open_heap:
        _, current, time_idx = heapq.heappop(open_heap)
        state = (current, time_idx)
        if current == goal and _goal_hold_clear(
            reserved_cells, goal, time_idx, goal_hold_steps, max_steps
        ):
            return _reconstruct_time_path(came_from, state)

        if time_idx >= max_steps - 1:
            continue

        for dx, dy, dz in offsets:
            nxt = (current[0] + dx, current[1] + dy, current[2] + dz)
            next_time = time_idx + 1
            if not grid.in_bounds(nxt) or not grid.is_free(nxt):
                continue
            if _cell_reserved(reserved_cells, nxt, next_time):
                continue
            if _edge_conflict(reserved_edges, current, nxt, next_time):
                continue

            nxt_state = (nxt, next_time)
            if nxt_state in g_cost and next_time >= g_cost[nxt_state]:
                continue
            g_cost[nxt_state] = next_time
            came_from[nxt_state] = state
            f_cost = next_time + _distance(nxt, goal)
            heapq.heappush(open_heap, (f_cost, nxt, next_time))

    return None


def _reconstruct_time_path(
    came_from: dict[tuple[Cell3D, int], tuple[Cell3D, int]],
    current: tuple[Cell3D, int],
) -> List[Cell3D]:
    path = [current[0]]
    while current in came_from:
        current = came_from[current]
        path.append(current[0])
    path.reverse()
    return path


def _goal_hold_clear(
    reserved_cells: dict[int, set[Cell3D]],
    goal: Cell3D,
    time_idx: int,
    hold_steps: int,
    max_steps: int,
) -> bool:
    end_time = min(max_steps - 1, time_idx + hold_steps)
    for t in range(time_idx + 1, end_time + 1):
        if _cell_reserved(reserved_cells, goal, t):
            return False
    return True


def _reserve_schedule(
    reserved_cells: dict[int, set[Cell3D]],
    reserved_edges: set[tuple[Cell3D, Cell3D, int]],
    schedule: List[Cell3D],
) -> None:
    for time_idx, cell in enumerate(schedule):
        _reserve_cell(reserved_cells, time_idx, cell)
        if time_idx == 0:
            continue
        prev = schedule[time_idx - 1]
        _reserve_edge(reserved_edges, prev, cell, time_idx)


def _neighbor_offsets(connectivity: int) -> List[Tuple[int, int, int]]:
    if connectivity == 6:
        return [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]
    if connectivity == 26:
        offsets = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    offsets.append((dx, dy, dz))
        return offsets
    raise ValueError("Connectivity must be 6 or 26.")


def _schedule_metrics(
    trajectories: List[List[Point3D]],
    goals: List[Cell3D],
    params: SwarmParams,
    grid: GridMap3D,
    dynamic_field: "DynamicObstacleField3D | None",
    mean_path_length: float,
) -> SwarmMetrics:
    agents = len(trajectories)
    if agents == 0:
        return SwarmMetrics(
            reached=0,
            total=0,
            min_separation=float("inf"),
            collision_steps=0,
            dynamic_collision_steps=0,
            mean_path_length=0.0,
            mean_traj_length=0.0,
            mean_steps_to_goal=None,
            mean_neighbors=0.0,
            steps=0,
        )

    max_steps = max((len(traj) for traj in trajectories), default=0)
    histories: List[List[Point3D]] = [[] for _ in trajectories]
    reached_flags = [False] * agents
    goal_steps: List[int | None] = [None] * agents
    rng = random.Random(params.comm_seed)

    min_pair_distance = float("inf")
    collision_steps = 0
    dynamic_collision_steps = 0
    neighbor_samples = 0

    for step in range(max_steps):
        dynamic_cells: set[Cell3D] = set()
        if dynamic_field is not None:
            dynamic_field.step(params.dt, grid)
            dynamic_cells = set(dynamic_field.cells(grid))

        positions: List[Point3D] = []
        for i, traj in enumerate(trajectories):
            pos = traj[step] if step < len(traj) else traj[-1]
            positions.append(pos)
            histories[i].append(pos)

            if not reached_flags[i]:
                gx, gy, gz = goals[i]
                if _distance(pos, (gx, gy, gz)) <= params.goal_tolerance:
                    reached_flags[i] = True
                    goal_steps[i] = step

        step_min = _min_pair_distance(positions)
        min_pair_distance = min(min_pair_distance, step_min)
        if step_min < params.min_separation:
            collision_steps += 1
        if dynamic_cells and _any_dynamic_collision(positions, dynamic_cells):
            dynamic_collision_steps += 1

        for i in range(agents):
            neighbors = _visible_neighbors(i, positions, histories, params, rng)
            neighbor_samples += len(neighbors)

    reached = sum(reached_flags)
    reached_steps = [s for s in goal_steps if s is not None]
    mean_steps_to_goal = (
        sum(reached_steps) / len(reached_steps) if reached_steps else None
    )
    mean_neighbors = (
        neighbor_samples / (max_steps * agents) if max_steps > 0 else 0.0
    )
    metrics = SwarmMetrics(
        reached=reached,
        total=agents,
        min_separation=min_pair_distance,
        collision_steps=collision_steps,
        dynamic_collision_steps=dynamic_collision_steps,
        mean_path_length=mean_path_length,
        mean_traj_length=_mean_length(trajectories),
        mean_steps_to_goal=mean_steps_to_goal,
        mean_neighbors=mean_neighbors,
        steps=max_steps,
    )
    return metrics


def _point_in_cells(point: Point3D, cells: set[Cell3D]) -> bool:
    if not cells:
        return False
    cell = _to_cell(point)
    return cell in cells


def _any_dynamic_collision(positions: List[Point3D], dynamic_cells: set[Cell3D]) -> bool:
    if not dynamic_cells:
        return False
    return any(_to_cell(pos) in dynamic_cells for pos in positions)


def _to_cell(point: Point3D) -> Cell3D:
    return int(round(point[0])), int(round(point[1])), int(round(point[2]))


def _cells_to_points(cells: List[Cell3D]) -> List[Point3D]:
    return [(float(x), float(y), float(z)) for x, y, z in cells]


def _reserve_cell(
    reserved: dict[int, set[Cell3D]], time_idx: int, cell: Cell3D
) -> None:
    reserved.setdefault(time_idx, set()).add(cell)


def _cell_reserved(reserved: dict[int, set[Cell3D]], cell: Cell3D, time_idx: int) -> bool:
    return cell in reserved.get(time_idx, set())


def _reserve_edge(
    reserved_edges: set[tuple[Cell3D, Cell3D, int]],
    start: Cell3D,
    end: Cell3D,
    time_idx: int,
) -> None:
    reserved_edges.add((start, end, time_idx))


def _edge_conflict(
    reserved_edges: set[tuple[Cell3D, Cell3D, int]],
    start: Cell3D,
    end: Cell3D,
    time_idx: int,
) -> bool:
    return (end, start, time_idx) in reserved_edges


def _visible_neighbors(
    index: int,
    positions: List[Point3D],
    trajectories: List[List[Point3D]],
    params: SwarmParams,
    rng: random.Random,
) -> List[Point3D]:
    neighbors: List[Point3D] = []
    for j, pos in enumerate(positions):
        if j == index:
            continue
        if params.comm_range > 0.0:
            if _distance(positions[index], pos) > params.comm_range:
                continue
        if params.comm_dropout > 0.0 and rng.random() < params.comm_dropout:
            continue
        if params.comm_delay_steps > 0:
            history = trajectories[j]
            delay_idx = max(0, len(history) - 1 - params.comm_delay_steps)
            pos = history[delay_idx]
        neighbors.append(pos)

    if params.comm_max_neighbors > 0 and len(neighbors) > params.comm_max_neighbors:
        rng.shuffle(neighbors)
        neighbors = neighbors[: params.comm_max_neighbors]
    return neighbors


def _face_candidates(
    grid: GridMap3D, free_cells: List[Cell3D]
) -> Tuple[List[Cell3D], List[Cell3D]]:
    margin = max(1, grid.width // 6)
    start_candidates = [cell for cell in free_cells if cell[0] <= margin]
    goal_candidates = [
        cell for cell in free_cells if cell[0] >= grid.width - 1 - margin
    ]
    return start_candidates, goal_candidates


def _swarm_guidance(
    pose: PoseState,
    path: List[Point3D],
    params: UAVParams,
    last_target_idx: int,
    neighbors: Iterable[Point3D],
    obstacle_positions: Iterable[Point3D],
    swarm_params: SwarmParams,
) -> Tuple[float, float, float, int, float, Point3D]:
    if not path:
        return 0.0, 0.0, 0.0, last_target_idx, float("inf"), (0.0, 0.0, 0.0)

    x, y, z, yaw, pitch = pose
    direction, target_idx = _target_direction(
        (x, y, z), path, params.lookahead, last_target_idx
    )
    repulsion, min_dist = _repulsion_vector(
        (x, y, z), neighbors, swarm_params.neighbor_radius
    )
    obstacle_repulsion, obstacle_min_dist = _repulsion_vector(
        (x, y, z), obstacle_positions, swarm_params.obstacle_avoidance_radius
    )

    combined = direction
    if min_dist < swarm_params.min_separation:
        combined = (
            combined[0] + swarm_params.avoidance_gain * repulsion[0],
            combined[1] + swarm_params.avoidance_gain * repulsion[1],
            combined[2] + swarm_params.avoidance_gain * repulsion[2],
        )
    if obstacle_repulsion != (0.0, 0.0, 0.0):
        combined = (
            combined[0] + swarm_params.obstacle_avoidance_gain * obstacle_repulsion[0],
            combined[1] + swarm_params.obstacle_avoidance_gain * obstacle_repulsion[1],
            combined[2] + swarm_params.obstacle_avoidance_gain * obstacle_repulsion[2],
        )
    if _norm(combined) < 1e-6:
        combined = direction if _norm(direction) > 1e-6 else repulsion

    speed, yaw_rate, pitch_rate = _command_from_direction(pose, combined, params)
    speed = _apply_neighbor_speed_scale(speed, min_dist, swarm_params)
    speed = _apply_obstacle_speed_scale(speed, obstacle_min_dist, swarm_params)

    return speed, yaw_rate, pitch_rate, target_idx, min_dist, direction


def _target_direction(
    position: Point3D,
    path: List[Point3D],
    lookahead: float,
    last_target_idx: int,
) -> Tuple[Point3D, int]:
    x, y, z = position
    target_idx = last_target_idx
    for i in range(last_target_idx, len(path)):
        if _distance((x, y, z), path[i]) >= lookahead:
            target_idx = i
            break
    else:
        target_idx = len(path) - 1

    tx, ty, tz = path[target_idx]
    return (tx - x, ty - y, tz - z), target_idx


def _repulsion_vector(
    position: Point3D,
    neighbors: Iterable[Point3D],
    radius: float,
) -> Tuple[Point3D, float]:
    rx, ry, rz = 0.0, 0.0, 0.0
    min_dist = float("inf")
    for nx, ny, nz in neighbors:
        dx = position[0] - nx
        dy = position[1] - ny
        dz = position[2] - nz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < min_dist:
            min_dist = dist
        if dist < 1e-6 or dist > radius:
            continue
        weight = (radius - dist) / radius
        rx += (dx / dist) * weight
        ry += (dy / dist) * weight
        rz += (dz / dist) * weight
    return (rx, ry, rz), min_dist


def _direction_to_angles(direction: Point3D, fallback_yaw: float) -> Tuple[float, float]:
    dx, dy, dz = direction
    horiz = math.hypot(dx, dy)
    desired_yaw = math.atan2(dy, dx) if horiz > 1e-9 else fallback_yaw
    desired_pitch = math.atan2(dz, max(horiz, 1e-9))
    return desired_yaw, desired_pitch


def _command_from_direction(
    pose: PoseState, direction: Point3D, params: UAVParams
) -> Tuple[float, float, float]:
    yaw = pose[3]
    pitch = pose[4]
    if _norm(direction) < 1e-6:
        return 0.0, 0.0, 0.0

    desired_yaw, desired_pitch = _direction_to_angles(direction, yaw)
    desired_pitch = _clamp(desired_pitch, -params.max_pitch, params.max_pitch)

    yaw_error = _wrap_angle(desired_yaw - yaw)
    pitch_error = desired_pitch - pitch

    yaw_rate = _clamp(params.yaw_gain * yaw_error, -params.yaw_rate_max, params.yaw_rate_max)
    pitch_rate = _clamp(
        params.pitch_gain * pitch_error, -params.pitch_rate_max, params.pitch_rate_max
    )

    alignment = max(abs(yaw_error), abs(pitch_error))
    speed_scale = max(0.3, math.cos(min(math.pi / 2.0, alignment)))
    speed = params.speed * speed_scale
    return speed, yaw_rate, pitch_rate


def _apply_neighbor_speed_scale(
    speed: float, min_dist: float, swarm_params: SwarmParams
) -> float:
    if min_dist < swarm_params.min_separation:
        return speed * 0.3
    if min_dist < swarm_params.neighbor_radius:
        return speed * max(0.5, min_dist / swarm_params.neighbor_radius)
    return speed


def _apply_obstacle_speed_scale(
    speed: float, min_dist: float, swarm_params: SwarmParams
) -> float:
    if min_dist == float("inf"):
        return speed
    if min_dist < 1e-6:
        return speed * 0.2
    if min_dist < swarm_params.obstacle_avoidance_radius:
        return speed * max(0.4, min_dist / swarm_params.obstacle_avoidance_radius)
    return speed


def _initial_orientation(path: List[Point3D], start_pose: Point3D) -> Tuple[float, float]:
    for point in path:
        dx = point[0] - start_pose[0]
        dy = point[1] - start_pose[1]
        dz = point[2] - start_pose[2]
        if abs(dx) + abs(dy) + abs(dz) > 1e-6:
            yaw = math.atan2(dy, dx)
            pitch = math.atan2(dz, max(math.hypot(dx, dy), 1e-9))
            return yaw, pitch
    return 0.0, 0.0


def _grid_to_path(plan: List[Cell3D]) -> List[Point3D]:
    return [(float(x), float(y), float(z)) for x, y, z in plan]


def _min_pair_distance(positions: List[Point3D]) -> float:
    if len(positions) < 2:
        return float("inf")
    min_dist = float("inf")
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = _distance(positions[i], positions[j])
            if dist < min_dist:
                min_dist = dist
    return min_dist


def _mean_length(paths: List[List[Point3D]]) -> float:
    if not paths:
        return 0.0
    lengths = []
    for path in paths:
        total = 0.0
        for i in range(1, len(path)):
            total += _distance(path[i - 1], path[i])
        lengths.append(total)
    return sum(lengths) / len(lengths)


def _distance(a: Point3D, b: Point3D) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _norm(vec: Point3D) -> float:
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
