# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from Stable-Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import random
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path

# ---------------
# E0509 추가
# ---------------
import doosan_tasks
import importlib

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with stable-baselines agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # ---------------
    # E0509 추가
    # ---------------
    # Get agent config from task spec if not provided by hydra or to override
    spec = gym.spec(args_cli.task)
    if hasattr(spec, "kwargs") and "sb3_cfg_entry_point" in spec.kwargs:
        entry_point = spec.kwargs["sb3_cfg_entry_point"]
        module_name, func_name = entry_point.split(":")
        module = importlib.import_module(module_name)
        agent_cfg = getattr(module, func_name)()
    elif hasattr(spec, "_kwargs") and "sb3_cfg_entry_point" in spec._kwargs:
        entry_point = spec._kwargs["sb3_cfg_entry_point"]
        module_name, func_name = entry_point.split(":")
        module = importlib.import_module(module_name)
        agent_cfg = getattr(module, func_name)()
    else:
        # Fallback for old tasks or if not found
        from doosan_tasks.e0509_reach.agents import sb3_ppo_e0509_cfg
        agent_cfg = sb3_ppo_e0509_cfg()

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # set the environment seed
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", train_task_name))
    
    # checkpoint path
    if args_cli.checkpoint is None:
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint, sort_alpha=False)
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(checkpoint_path)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    vec_norm_path = checkpoint_path.replace("/model", "/model_vecnormalize").replace(".zip", ".pkl")
    vec_norm_path = Path(vec_norm_path)

    # normalize environment (if needed)
    if vec_norm_path.exists():
        print(f"Loading saved normalization: {vec_norm_path}")
        env = VecNormalize.load(str(vec_norm_path), env)
        env.training = False
        env.norm_reward = False
    elif agent_cfg.get("normalize_input"):
        env = VecNormalize(
            env,
            training=True,
            norm_obs=agent_cfg.get("normalize_input", True),
            clip_obs=agent_cfg.get("clip_obs", 10.0),
        )

    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent = PPO.load(checkpoint_path, env, print_system_info=True)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions, _ = agent.predict(obs, deterministic=True)
            # env stepping
            obs, _, _, _ = env.step(actions)
        
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
