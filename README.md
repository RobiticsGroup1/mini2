# Doosan E0509 Reach Project

This repository contains a reinforcement learning project for the Doosan E0509 robot using Isaac Lab and Stable Baselines3 (SB3). The task is a "Reach" task where the robot arm learns to move its end-effector to a target position.

## Project Structure

- `train_e0509.py`: The main training script using Stable Baselines3.
- `doosan_tasks/`: Contains the task environment definitions and SB3 agent configurations.
- `asset/`: Contains the URDF and mesh files for the Doosan E0509 robot and its gripper.
- `tools/`: Utility scripts for the project.

## Requirements

- NVIDIA GPU (CUDA compatible)
- Ubuntu 22.04 or 20.04
- Isaac Sim 4.2.0 or later
- Isaac Lab

## Installation

### 1. Isaac Lab Setup
Ensure you have Isaac Lab installed. Follow the official [Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html).

### 2. Virtual Environment
Isaac Lab provides a script to set up a virtual environment automatically. From the `IsaacLab` root directory, run:

```bash
# Create the virtual environment
./isaaclab.sh --install
```

This will create a `.venv` folder (or equivalent) within the `IsaacLab` directory containing all necessary dependencies.

## Usage

To run the training for the Doosan E0509 reach task, use the following command from the `IsaacLab` root directory:

```bash
cd ~/IsaacLab
./isaaclab.sh -p doosan_isaaclab_sb3/train_e0509.py --task DoosanE0509-Reach-v0 --num_envs 64 --device cuda:0
```

### Parameters:
- `-p`: Specifies the python script to run.
- `--task`: The registered Gym task ID.
- `--num_envs`: Number of parallel environments to simulate.
- `--device`: The device to run the simulation and training on (e.g., `cuda:0`).

## Acknowledgments
This project is built using [Isaac Lab](https://github.com/isaac-sim/IsaacLab).
