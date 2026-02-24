# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from Stable-Baselines3 for Doosan tasks."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--use_last_checkpoint", action="store_true", help="Use the last saved model.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

# IMPORTANT: Import your custom tasks to register them
import doosan_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    # override configurations
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # determine checkpoint path
    task_name = args_cli.task.split(":")[-1]
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", task_name))
    
    if args_cli.checkpoint is None:
        checkpoint = "model_.*.zip" if args_cli.use_last_checkpoint else "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)
    else:
        checkpoint_path = args_cli.checkpoint

    if checkpoint_path is None:
        raise ValueError(f"No checkpoint found in {log_root_path}. Please provide one with --checkpoint.")

    print(f"Loading checkpoint from: {checkpoint_path}")

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    # load normalization if it exists
    vec_norm_path = checkpoint_path.replace("/model", "/model_vecnormalize").replace(".zip", ".pkl")
    if os.path.exists(vec_norm_path):
        print(f"Loading saved normalization: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False

    # load agent
    agent = PPO.load(checkpoint_path, env)

    # simulate
    obs = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(actions)
        
        if args_cli.real_time:
            time.sleep(1.0 / 60.0) # Approx real-time

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
