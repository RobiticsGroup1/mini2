# Doosan E0509 Isaac Lab Project

This project focuses on reinforcement learning for the Doosan E0509 6-DOF robot arm using NVIDIA Isaac Lab and Stable Baselines 3 (SB3). It includes tasks for reaching, picking, and pick-and-place operations.

## Project Overview

- **Core Technologies:** NVIDIA Isaac Sim, Isaac Lab, Stable Baselines 3 (PPO), Gymnasium.
- **Robot Asset:** Doosan E0509 (URDF-based).
- **Architecture:** The project contains multiple task definitions under `doosan_tasks/`, currently transitioning from `DirectRLEnv` to the more modular `ManagerBasedRLEnv`.

## Project Structure

- `train_e0509.py`: The entry point for training RL agents using Stable Baselines 3.
- `doosan_tasks/`:
  - `e0509_reach/`: Task where the robot arm learns to move its end-effector to a target.
  - `e0509_pick/`: Task for picking up objects (Manager-based transition in progress).
  - `e0509_pick_place/`: Task for picking and placing objects.
  - Each task contains:
    - `env.py`: Environment logic.
    - `env_cfg.py`: Environment and scene configurations.
    - `agents.py`: RL agent (SB3) configurations.
- `asset/`: Contains the URDF, USD, and mesh files for the Doosan E0509 robot.
- `summary.txt`: Contains project roadmap and architectural design notes.

## Building and Running

### Prerequisites
- NVIDIA Isaac Sim 4.2.0 or later.
- NVIDIA Isaac Lab installation.
- CUDA-compatible GPU.

### Installation
Typically, this project is used within an Isaac Lab environment.
```bash
# Example if using the Isaac Lab helper script
./isaaclab.sh --install
```

### Running Training
Use `train_e0509.py` with the desired task ID.
```bash
# Reach Task
python train_e0509.py --task DoosanE0509-Reach-v0 --num_envs 64 --device cuda:0

# Pick Task
python train_e0509.py --task DoosanE0509-Pick-v0 --num_envs 64 --device cuda:0
```
Note: If using the `isaaclab.sh` wrapper:
```bash
./isaaclab.sh -p train_e0509.py --task DoosanE0509-Reach-v0 --num_envs 64 --device cuda:0
```

## Development Conventions

- **Task Registration:** All tasks are registered as Gymnasium environments in their respective `__init__.py` files using `gym.register`.
- **Environment Design:**
  - `DirectRLEnv` is used for simpler tasks (legacy).
  - `ManagerBasedRLEnv` is preferred for new or complex tasks (modularity for rewards, observations, and events).
- **Configurations:** Use the `@configclass` decorator from `isaaclab.utils` for environment and scene configurations.
- **Observations/Actions:** Standardized through `gym.spaces`.

## Roadmap
- [x] Complete transition of `e0509_pick` to Manager-based environment.
- [x] Implement specialized reward functions for robust picking in `rewards.py`.
- [x] Configure Action Manager for Inverse Kinematics (IK) control.
- [ ] Extend tasks to include more complex manipulation scenarios.
