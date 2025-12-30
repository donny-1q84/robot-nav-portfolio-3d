# Robot Navigation 3D Demo

This project is a minimal closed-loop 3D navigation demo:
- 3D grid (voxel) map with obstacles
- 3D A* path planning (6 or 26 connectivity)
- 3D yaw/pitch kinematics with lookahead path guidance
- Matplotlib 3D visualization (PNG, optional GIF)

## Quick Start
```bash
python -m venv .venv
.venv/bin/pip install -e .
navsim3d-demo --start 0,0,0 --goal 11,11,5 --png output.png
```

Optional GIF:
```bash
navsim3d-demo --gif output.gif
```

## Swarm Demo
Multi-agent planning + distributed collision avoidance:
```bash
navsim3d-swarm --png swarm.png
```

Override agent count or seed:
```bash
navsim3d-swarm --agents 20 --seed 7 --png swarm.png
```

Prioritized baseline (time-coordinated scheduling):
```bash
navsim3d-swarm --mode prioritized --png swarm_prioritized.png
```

Cooperative A* baseline (time-extended planning with reservations):
```bash
navsim3d-swarm --mode cooperative --png swarm_cooperative.png
```

Sweep multiple swarm sizes (writes metrics to CSV):
```bash
navsim3d-swarm --sweep 4,8,12 --csv swarm_metrics.csv
```

Sweep communication dropout or delay:
```bash
navsim3d-swarm --comm-dropout-sweep 0,0.2,0.4 --csv comm_dropout.csv
navsim3d-swarm --comm-delay-sweep 0,3,6 --csv comm_delay.csv
```

Enable dynamic obstacles:
```bash
navsim3d-swarm --dynamic --png swarm_dynamic.png
```

Communication limits are configurable in `configs/swarm.yaml`:
`comm_range`, `comm_delay_steps`, `comm_dropout`, `comm_max_neighbors`.
Dynamic obstacles are configurable in `configs/swarm.yaml` under `dynamic_obstacles`.

Learning-based prediction (tiny MLP) for dynamic obstacles:
```bash
python scripts/train_obstacle_predictor.py --out models/obstacle_mlp.json
navsim3d-swarm --dynamic --predictive --respect-obstacles --png swarm_predictive.png
```

To regenerate plots from CSV sweeps, run:
```bash
python scripts/plot_swarm_results.py
```
This script writes both PNG and PDF plots into `reports/`.

## Parameters
```
--config path  Config file (default: configs/default.yaml)
--start x,y,z  Start voxel
--goal x,y,z   Goal voxel
--png path     Output PNG path (default: output.png)
--gif path     Optional GIF path
--inflation-radius  Obstacle inflation radius (voxel units)
--connectivity      6 or 26 neighbor connectivity for A*
--lookahead         Lookahead distance (default: 1.2)
--speed             Speed (default: 1.0)
--yaw-rate-max      Max yaw rate (default: 1.5)
--pitch-rate-max    Max pitch rate (default: 1.0)
--max-pitch         Max pitch angle in radians (default: 0.9)
--yaw-gain          Yaw error gain (default: 1.2)
--pitch-gain        Pitch error gain (default: 1.0)
--dynamic           Enable dynamic obstacles with replanning
--no-dynamic        Disable dynamic obstacles
--replan-interval   Steps between replans (default from config)
--max-replans       Safety cap on replans (default from config)
```

## Design Notes
- **Map**: hard-coded 3D demo map in `navsim3d/map3d.py`.
- **Planner**: 3D A* on a voxel grid.
- **Controller**: yaw/pitch kinematic model with rate limits and lookahead guidance.
- **Costmap**: 3D obstacle inflation.
- **Dynamic obstacles**: moving voxels with periodic replanning.
- **Visualization**: 3D obstacles, planned path, executed trajectory.
- **Swarm**: per-agent A* with local avoidance or prioritized/cooperative scheduling, plus optional comm limits.
- **Learning**: MLP-based dynamic obstacle predictor for predictive avoidance.
