from __future__ import annotations

import torch
import gymnasium as gym
import numpy as np

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_rotate, quat_from_euler_xyz, quat_mul
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from .env_cfg import DoosanE0509PickPlaceEnvCfg


class DoosanE0509PickPlaceEnv(DirectRLEnv):
    def __init__(self, cfg=None, render_mode: str | None = None, **kwargs):
        if cfg is None:
            cfg = DoosanE0509PickPlaceEnvCfg()
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # 로봇 핸들
        self.robot: Articulation = self.scene["robot"]
        self.snack: RigidObject = self.scene["snack"]

        # 조인트 인덱스 (e0509: joint_1 ... joint_6)
        self.arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.gripper_joint_names = ["rh_l1", "rh_l2", "rh_r1", "rh_r2"]
        
        all_joint_names = list(self.robot.data.joint_names)
        name_to_id = {name: i for i, name in enumerate(all_joint_names)}

        self.arm_joint_ids = torch.tensor([name_to_id[n] for n in self.arm_joint_names],
                                         device=self.device, dtype=torch.long)
        self.gripper_joint_ids = torch.tensor([name_to_id[n] for n in self.gripper_joint_names],
                                             device=self.device, dtype=torch.long)

        # EE 바디 인덱스
        all_body_names = list(self.robot.data.body_names)
        body_to_id = {name: i for i, name in enumerate(all_body_names)}

        if self.cfg.ee_link_name not in body_to_id:
            raise RuntimeError(
                f"EE link '{self.cfg.ee_link_name}' not found.\n"
                f"Available bodies: {all_body_names}"
            )
        self.ee_body_id = body_to_id[self.cfg.ee_link_name]

        # Home position (Initial joint state)
        self.home_q = self.robot.data.default_joint_pos.clone()
        self.home_ee_pos_l = torch.zeros((self.num_envs, 3), device=self.device) # Will be set in first reset

        # Basket goal (Top center)
        self.basket_goal_l = torch.tensor([0.0, -0.45, 0.035], device=self.device).repeat(self.num_envs, 1)

        # PD 타겟 추적용 변수
        self.current_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        # Sequential task tracking
        # Stages: 0: Approach, 1: Grip, 2: Return Home with Snack, 3: Move to Basket, 4: Release, 5: Return Home empty
        self.task_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.stage_timer = torch.zeros(self.num_envs, device=self.device)

        # 관측/액션 스페이스
        self.observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(self.cfg.obs_dim,), dtype=float
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.action_dim,), dtype=float
        )

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        # Snack 물체 초기화
        snack_pos_w = torch.tensor(self.cfg.scene.snack.init_state.pos, device=self.device).repeat(len(env_ids), 1) + self.scene.env_origins[env_ids]
        snack_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        
        self.snack.write_root_pose_to_sim(torch.cat([snack_pos_w, snack_quat_w], dim=-1), env_ids=env_ids)
        self.snack.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)

        # 로봇 초기화
        q_init = self.robot.data.default_joint_pos[env_ids]
        self.current_targets[env_ids] = q_init
        self.robot.set_joint_position_target(q_init, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q_init, torch.zeros_like(q_init), env_ids=env_ids)

        # Reset task tracking
        self.task_stage[env_ids] = 0
        self.stage_timer[env_ids] = 0.0
        
        # Capture home EE pos if not set
        if torch.all(self.home_ee_pos_l == 0):
            self.home_ee_pos_l = self.robot.data.body_pos_w[:, self.ee_body_id, :] - self.scene.env_origins

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = torch.clamp(actions, -1.0, 1.0)
        self._apply_action()

    def _apply_action(self):
        # Arm delta
        self.current_targets[:, self.arm_joint_ids] += self._actions[:, :6] * self.cfg.action_scale
        
        # Gripper absolute
        # Gripper range [0, 1.1]
        gripper_pos = (self._actions[:, 6:7] + 1.0) / 2.0 * 1.1 
        self.current_targets[:, self.gripper_joint_ids] = gripper_pos
        
        self.robot.set_joint_position_target(self.current_targets)

    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel
        ee_pos_l = self.robot.data.body_pos_w[:, self.ee_body_id, :] - self.scene.env_origins
        snack_pos_l = self.snack.data.root_pos_w[:, :3] - self.scene.env_origins
        ee_to_snack_pos_l = snack_pos_l - ee_pos_l

        obs = torch.cat([
            q, qd, ee_pos_l, snack_pos_l, self.basket_goal_l,
            ee_to_snack_pos_l,
            self.task_stage.float().unsqueeze(-1), 
            self.stage_timer.unsqueeze(-1)
        ], dim=-1)
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        # Update timer (dt * decimation)
        self.stage_timer += self.cfg.sim.dt * self.cfg.decimation

        ee_pos_l = self.robot.data.body_pos_w[:, self.ee_body_id, :] - self.scene.env_origins
        snack_pos_l = self.snack.data.root_pos_w[:, :3] - self.scene.env_origins
        ee_quat_w = self.robot.data.body_quat_w[:, self.ee_body_id, :]
        
        # Distances
        dist_ee_snack = torch.norm(ee_pos_l - snack_pos_l, dim=-1)
        dist_ee_home = torch.norm(ee_pos_l - self.home_ee_pos_l, dim=-1)
        dist_ee_basket = torch.norm(ee_pos_l - self.basket_goal_l, dim=-1)
        dist_snack_basket = torch.norm(snack_pos_l - self.basket_goal_l, dim=-1)

        # Orientation reward (Gripper pointing down)
        down_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        ee_z_axis = quat_rotate(ee_quat_w, torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1))
        align_dot = torch.clamp(torch.sum(ee_z_axis * down_vec, dim=-1), min=0.0)

        # Stage management
        # 0: Approach Snack
        # 1: Grip Snack
        # 2: Return Home with Snack
        # 3: Move to Basket
        # 4: Release & Wait
        # 5: Return Home empty
        
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Stage 0: Approach
        stage_0_mask = (self.task_stage == 0)
        # Use exponential for dense reward at distance
        rewards[stage_0_mask] = torch.exp(-5.0 * dist_ee_snack[stage_0_mask]) * self.cfg.reach_reward_scale
        rewards[stage_0_mask] += align_dot[stage_0_mask] * 0.5
        # Keeping gripper high bonus
        rewards[stage_0_mask] += (ee_pos_l[stage_0_mask, 2] > 0.05).float() * 0.5
        
        # Transition 0 -> 1
        reached_snack = (dist_ee_snack < self.cfg.reach_success_dist) & stage_0_mask
        self.task_stage[reached_snack] = 1
        self.stage_timer[reached_snack] = 0.0

        # Stage 1: Grip
        stage_1_mask = (self.task_stage == 1)
        # Reward closing the gripper (actions[6] > 0)
        rewards[stage_1_mask] = self._actions[stage_1_mask, 6] * self.cfg.grasp_reward_scale
        
        # Transition 1 -> 2 (Grip for 0.5s)
        gripped = (self.stage_timer > 0.5) & stage_1_mask
        self.task_stage[gripped] = 2
        self.stage_timer[gripped] = 0.0

        # Stage 2: Return Home with Snack
        stage_2_mask = (self.task_stage == 2)
        rewards[stage_2_mask] = (1.0 - torch.tanh(dist_ee_home[stage_2_mask] / 0.2)) * self.cfg.home_reward_scale
        # Check if snack is still lifted
        lifted = (snack_pos_l[:, 2] > 0.05)
        rewards[stage_2_mask] += lifted[stage_2_mask].float() * 2.0
        
        # Transition 2 -> 3
        at_home = (dist_ee_home < self.cfg.home_success_dist) & stage_2_mask
        self.task_stage[at_home] = 3
        self.stage_timer[at_home] = 0.0

        # Stage 3: Move to Basket
        stage_3_mask = (self.task_stage == 3)
        rewards[stage_3_mask] = (1.0 - torch.tanh(dist_ee_basket[stage_3_mask] / 0.2)) * self.cfg.place_reward_scale
        # Movec-like behavior: Reward height being maintained or specific curve (simplified as height > basket)
        rewards[stage_3_mask] += (ee_pos_l[stage_3_mask, 2] > 0.1).float() * 1.0
        
        # Transition 3 -> 4
        at_basket = (dist_ee_basket < self.cfg.place_success_dist) & stage_3_mask
        self.task_stage[at_basket] = 4
        self.stage_timer[at_basket] = 0.0

        # Stage 4: Release & Wait
        stage_4_mask = (self.task_stage == 4)
        # Reward opening the gripper (actions[6] < 0)
        rewards[stage_4_mask] = -self._actions[stage_4_mask, 6] * self.cfg.grasp_reward_scale
        
        # Transition 4 -> 5 (Wait 2s)
        waited = (self.stage_timer > 2.0) & stage_4_mask
        self.task_stage[waited] = 5
        self.stage_timer[waited] = 0.0

        # Stage 5: Return Home empty
        stage_5_mask = (self.task_stage == 5)
        rewards[stage_5_mask] = (1.0 - torch.tanh(dist_ee_home[stage_5_mask] / 0.2)) * self.cfg.home_reward_scale

        # Penalties
        r_act = -torch.sum(torch.square(self._actions), dim=-1) * self.cfg.action_penalty_scale
        joint_vel = self.robot.data.joint_vel
        r_vel = -torch.sum(torch.square(joint_vel), dim=-1) * self.cfg.joint_vel_penalty_scale
        
        return rewards + r_act + r_vel
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        # Success: Snack in basket AND robot back at home (Stage 5 complete)
        snack_pos_l = self.snack.data.root_pos_w[:, :3] - self.scene.env_origins
        dist_snack_basket = torch.norm(snack_pos_l - self.basket_goal_l, dim=-1)
        ee_pos_l = self.robot.data.body_pos_w[:, self.ee_body_id, :] - self.scene.env_origins
        dist_ee_home = torch.norm(ee_pos_l - self.home_ee_pos_l, dim=-1)
        
        success = (dist_snack_basket < 0.15) & (dist_ee_home < 0.1) & (self.task_stage == 5)
        
        terminated = success
        return terminated, truncated
