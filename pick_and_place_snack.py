# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import torch
import numpy as np
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Scripted pick and place for Doosan E0509.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, quat_rotate, subtract_frame_transforms

from doosan_tasks.e0509_pick_place.env_cfg import DoosanE0509PickPlaceEnvCfg
from doosan_tasks.e0509_pick_place.env import DoosanE0509PickPlaceEnv

class ScriptedPickPlaceEnv(DoosanE0509PickPlaceEnv):
    def __init__(self, cfg=None, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # IK Controller for scripted movement
        ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.ik_controller = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        # State machine
        self.state = "APPROACH_SNACK"
        self.state_timer = 0
        
        # Positions
        self.init_ee_pos = None
        self.init_ee_quat = None
        self.grasp_ee_pos_w = None
        
        # Basket goal position (local)
        # Basket center is at (0.0, -0.45, -0.215)
        # Top of basket is at ~0.035
        self.basket_goal_pos = torch.tensor([0.0, -0.45, 0.15], device=self.device)
        self.basket_goal_quat = quat_from_euler_xyz(
            torch.tensor(0.0, device=self.device),
            torch.tensor(np.pi, device=self.device), # Pointing down
            torch.tensor(0.0, device=self.device)
        ).repeat(self.num_envs, 1)

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self.state = "APPROACH_SNACK"
        self.state_timer = 0
        
        # Capture initial EE pose
        if self.init_ee_pos is None:
            # Step once to get initial data
            self.sim.step()
            self.init_ee_pos = self.robot.data.body_pos_w[0, self.ee_body_id, :].clone() - self.scene.env_origins[0]
            self.init_ee_quat = self.robot.data.body_quat_w[0, self.ee_body_id, :].clone()

    def _apply_action(self):
        # This is where we implement the scripted logic
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_id, :]
        ee_pos_l = ee_pos_w - self.scene.env_origins
        ee_quat_w = self.robot.data.body_quat_w[:, self.ee_body_id, :]
        
        snack_pos_w = self.snack.data.root_pos_w[:, :3]
        snack_pos_l = snack_pos_w - self.scene.env_origins
        
        # Target in WORLD frame first
        target_pos_w = torch.zeros_like(ee_pos_w)
        target_quat_w = torch.zeros_like(ee_quat_w)
        gripper_command = torch.zeros((self.num_envs, 1), device=self.device) # 0: open, 1: closed
        
        if self.state == "APPROACH_SNACK":
            # Move above snack
            target_pos_w = snack_pos_w.clone()
            target_pos_w[:, 2] += 0.25 # Sufficient clearance
            target_quat_w = quat_from_euler_xyz(
                torch.tensor(0.0, device=self.device),
                torch.tensor(np.pi, device=self.device),
                torch.tensor(0.0, device=self.device)
            ).repeat(self.num_envs, 1)
            
            dist = torch.norm(ee_pos_w - target_pos_w, dim=-1)
            if torch.all(dist < 0.02):
                self.state = "DESCEND_SNACK"
        
        elif self.state == "DESCEND_SNACK":
            # Move to snack, accounting for gripper length (~0.15m offset to avoid desk)
            target_pos_w = snack_pos_w.clone()
            target_pos_w[:, 2] += 0.15 
            target_quat_w = quat_from_euler_xyz(
                torch.tensor(0.0, device=self.device),
                torch.tensor(np.pi, device=self.device),
                torch.tensor(0.0, device=self.device)
            ).repeat(self.num_envs, 1)
            
            dist = torch.norm(ee_pos_w - target_pos_w, dim=-1)
            if torch.all(dist < 0.02): # Relaxed tolerance
                self.state = "GRASP_SNACK"
                self.state_timer = 0
                self.grasp_ee_pos_w = target_pos_w.clone() # Capture where we actually grasped
        
        elif self.state == "GRASP_SNACK":
            # Stay at the descent target while closing
            target_pos_w = self.grasp_ee_pos_w.clone()
            target_quat_w = quat_from_euler_xyz(
                torch.tensor(0.0, device=self.device),
                torch.tensor(np.pi, device=self.device),
                torch.tensor(0.0, device=self.device)
            ).repeat(self.num_envs, 1)
            gripper_command[:] = 1.0 # Close
            self.state_timer += 1
            if self.state_timer > 60: # Longer wait for grasp
                self.state = "LIFT_SNACK"
        
        elif self.state == "LIFT_SNACK":
            target_pos_w = self.grasp_ee_pos_w.clone()
            target_pos_w[:, 2] += 0.20 # Lift relative to grasp pos
            target_quat_w = quat_from_euler_xyz(
                torch.tensor(0.0, device=self.device),
                torch.tensor(np.pi, device=self.device),
                torch.tensor(0.0, device=self.device)
            ).repeat(self.num_envs, 1)
            gripper_command[:] = 1.0
            dist = torch.norm(ee_pos_w - target_pos_w, dim=-1)
            if torch.all(dist < 0.02):
                self.state = "RETURN_INIT"
        
        elif self.state == "RETURN_INIT":
            target_pos_w = self.init_ee_pos.repeat(self.num_envs, 1) + self.scene.env_origins
            target_quat_w = self.init_ee_quat.repeat(self.num_envs, 1)
            gripper_command[:] = 1.0
            dist = torch.norm(ee_pos_w - target_pos_w, dim=-1)
            if torch.all(dist < 0.02):
                self.state = "MOVE_CIRCULAR_BASKET"
                self.state_timer = 0
                self.start_pos_w = ee_pos_w.clone()
                self.start_quat_w = ee_quat_w.clone()
        
        elif self.state == "MOVE_CIRCULAR_BASKET":
            # Circular motion (movec)
            duration = 60.0
            t = min(self.state_timer / duration, 1.0)
            self.state_timer += 1
            
            basket_goal_pos_w = self.basket_goal_pos + self.scene.env_origins
            
            # Interpolate pos
            target_pos_w = self.start_pos_w * (1 - t) + basket_goal_pos_w * t
            # Add arc height
            arc_height = 0.2 * np.sin(np.pi * t)
            target_pos_w[:, 2] += arc_height
            
            target_quat_w = self.basket_goal_quat.clone()
            gripper_command[:] = 1.0
            
            if t >= 1.0:
                self.state = "RELEASE_SNACK"
                self.state_timer = 0
        
        elif self.state == "RELEASE_SNACK":
            target_pos_w = self.basket_goal_pos.repeat(self.num_envs, 1) + self.scene.env_origins
            target_quat_w = self.basket_goal_quat.clone()
            gripper_command[:] = 0.0 # Open
            self.state_timer += 1
            if self.state_timer > 30:
                self.state = "WAIT_AFTER_RELEASE"
                self.state_timer = 0
        
        elif self.state == "WAIT_AFTER_RELEASE":
            target_pos_w = self.basket_goal_pos.repeat(self.num_envs, 1) + self.scene.env_origins
            target_quat_w = self.basket_goal_quat.clone()
            gripper_command[:] = 0.0
            self.state_timer += 1
            if self.state_timer > 60:
                self.state = "RETURN_INIT_CIRCULAR"
                self.state_timer = 0
                self.start_pos_w = ee_pos_w.clone()
                self.start_quat_w = ee_quat_w.clone()

        elif self.state == "RETURN_INIT_CIRCULAR":
            duration = 60.0
            t = min(self.state_timer / duration, 1.0)
            self.state_timer += 1
            
            init_ee_pos_w = self.init_ee_pos + self.scene.env_origins
            
            target_pos_w = self.start_pos_w * (1 - t) + init_ee_pos_w * t
            arc_height = 0.2 * np.sin(np.pi * t)
            target_pos_w[:, 2] += arc_height
            
            target_quat_w = self.init_ee_quat.repeat(self.num_envs, 1)
            gripper_command[:] = 0.0
            
            if t >= 1.0:
                print("Task Done!")
                self.state = "DONE"
        
        elif self.state == "DONE":
            target_pos_w = self.init_ee_pos.repeat(self.num_envs, 1) + self.scene.env_origins
            target_quat_w = self.init_ee_quat.repeat(self.num_envs, 1)
            gripper_command[:] = 0.0

        # --- Transform everything to ROBOT ROOT frame for IK ---
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        
        # EE pose in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        # Target pose in root frame
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w, target_quat_w
        )

        # Apply IK
        self.ik_controller.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))
        
        arm_actions = self.ik_controller.compute(
            ee_pos_b,
            ee_quat_b,
            self.robot.root_physx_view.get_jacobians()[:, self.ee_body_id - 1, :, self.arm_joint_ids],
            self.robot.data.joint_pos[:, self.arm_joint_ids]
        )
        
        # Combined actions
        q = self.robot.data.joint_pos.clone()
        q[:, self.arm_joint_ids] = arm_actions
        
        # Gripper: 0 to 1.1
        gripper_pos = gripper_command * 1.1
        q[:, self.gripper_joint_ids] = gripper_pos
        
        self.robot.set_joint_position_target(q)

def main():
    env_cfg = DoosanE0509PickPlaceEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ScriptedPickPlaceEnv(cfg=env_cfg)
    
    obs, _ = env.reset()
    print("Starting Scripted Pick and Place...")
    while simulation_app.is_running():
        env.step(torch.zeros((env.num_envs, 7), device=env.device))
        if env.state == "DONE":
            # Wait a bit then exit
            for _ in range(50):
                if not simulation_app.is_running(): break
                env.step(torch.zeros((env.num_envs, 7), device=env.device))
            break

if __name__ == "__main__":
    main()
    simulation_app.close()
