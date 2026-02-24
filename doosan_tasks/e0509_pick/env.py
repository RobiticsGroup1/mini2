from __future__ import annotations

import torch
import gymnasium as gym

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_rotate

from .env_cfg import DoosanE0509PickEnvCfg


class DoosanE0509PickEnv(DirectRLEnv):
    def __init__(self, cfg=None, render_mode: str | None = None, **kwargs):
        if cfg is None:
            cfg = DoosanE0509PickEnvCfg()
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # Assets
        self.robot: Articulation = self.scene["robot"]
        self.snack: RigidObject = self.scene["snack"]

        # Joint Indices
        self.arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.gripper_joint_names = ["rh_l1", "rh_l2", "rh_r1", "rh_r2"]
        
        all_joint_names = list(self.robot.data.joint_names)
        name_to_id = {name: i for i, name in enumerate(all_joint_names)}

        self.arm_joint_ids = torch.tensor([name_to_id[n] for n in self.arm_joint_names],
                                         device=self.device, dtype=torch.long)
        self.gripper_joint_ids = torch.tensor([name_to_id[n] for n in self.gripper_joint_names],
                                             device=self.device, dtype=torch.long)

        # EE Link Index
        all_body_names = list(self.robot.data.body_names)
        body_to_id = {name: i for i, name in enumerate(all_body_names)}
        self.ee_body_id = body_to_id[self.cfg.ee_link_name]
        
        # EE Offset (from flange to tip)
        self.ee_offset = torch.tensor(self.cfg.ee_offset, device=self.device).repeat(self.num_envs, 1)

        # Targets and States
        self.home_ee_pos_l = torch.zeros((self.num_envs, 3), device=self.device)
        self.default_arm_q = self.robot.data.default_joint_pos[:, self.arm_joint_ids].clone()
        
        self.current_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.task_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.stage_timer = torch.zeros(self.num_envs, device=self.device)

        # Observation/Action Space
        self.observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(self.cfg.obs_dim,), dtype=float
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.action_dim,), dtype=float
        )

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        # Snack Reset
        snack_pos_w = torch.tensor(self.cfg.scene.snack.init_state.pos, device=self.device).repeat(len(env_ids), 1) + self.scene.env_origins[env_ids]
        snack_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        self.snack.write_root_pose_to_sim(torch.cat([snack_pos_w, snack_quat_w], dim=-1), env_ids=env_ids)
        self.snack.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)

        # 로봇 초기화
        q_init = self.robot.data.default_joint_pos[env_ids]
        self.current_targets[env_ids] = q_init
        self.robot.set_joint_position_target(q_init, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q_init, torch.zeros_like(q_init), env_ids=env_ids)

        # Task State Reset
        self.task_stage[env_ids] = 0
        self.stage_timer[env_ids] = 0.0
        
        # Capture Initial EE Position (Home)
        if torch.all(self.home_ee_pos_l == 0):
            # Apply offset to home pos as well
            flange_pos_w = self.robot.data.body_pos_w[:, self.ee_body_id, :]
            flange_quat_w = self.robot.data.body_quat_w[:, self.ee_body_id, :]
            tip_pos_w = flange_pos_w + quat_rotate(flange_quat_w, self.ee_offset)
            self.home_ee_pos_l = tip_pos_w - self.scene.env_origins

    def _get_ee_tip_pos_l(self):
        flange_pos_w = self.robot.data.body_pos_w[:, self.ee_body_id, :]
        flange_quat_w = self.robot.data.body_quat_w[:, self.ee_body_id, :]
        tip_pos_w = flange_pos_w + quat_rotate(flange_quat_w, self.ee_offset)
        return tip_pos_w - self.scene.env_origins

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = torch.clamp(actions, -1.0, 1.0)
        self._apply_action()

    def _apply_action(self):
        # Arm delta control
        self.current_targets[:, self.arm_joint_ids] += self._actions[:, :6] * self.cfg.action_scale
        # Gripper absolute control [0, 1.1]
        # Inverted mapping: 1.0 (Action) -> 0.0 (Closed), -1.0 (Action) -> 1.1 (Open)
        gripper_pos = (1.0 - self._actions[:, 6:7]) / 2.0 * 1.1 
        self.current_targets[:, self.gripper_joint_ids] = gripper_pos
        self.robot.set_joint_position_target(self.current_targets)

    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel
        ee_pos_l = self._get_ee_tip_pos_l()
        snack_pos_l = self.snack.data.root_pos_w[:, :3] - self.scene.env_origins
        ee_to_snack_l = snack_pos_l - ee_pos_l

        obs = torch.cat([
            q, qd, ee_pos_l, snack_pos_l, self.home_ee_pos_l, ee_to_snack_l,
            self.task_stage.float().unsqueeze(-1), 
            self.stage_timer.unsqueeze(-1)
        ], dim=-1)
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        self.stage_timer += self.cfg.sim.dt * self.cfg.decimation

        ee_pos_l = self._get_ee_tip_pos_l()
        snack_pos_l = self.snack.data.root_pos_w[:, :3] - self.scene.env_origins
        ee_quat_w = self.robot.data.body_quat_w[:, self.ee_body_id, :]
        
        dist_ee_snack = torch.norm(ee_pos_l - snack_pos_l, dim=-1)
        dist_ee_home = torch.norm(ee_pos_l - self.home_ee_pos_l, dim=-1)
        
        # Joint distance to home (arm joints only)
        arm_q = self.robot.data.joint_pos[:, self.arm_joint_ids]
        dist_joint_home = torch.norm(arm_q - self.default_arm_q, dim=-1)

        # Orientation: Gripper pointing down
        down_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        ee_z_axis = quat_rotate(ee_quat_w, torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1))
        align_dot = torch.clamp(torch.sum(ee_z_axis * down_vec, dim=-1), min=0.0)

        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # --- Stage 0: Approach ---
        stage_0_mask = (self.task_stage == 0)
        rewards[stage_0_mask] = torch.exp(-20.0 * dist_ee_snack[stage_0_mask]) * self.cfg.reach_reward_scale
        rewards[stage_0_mask] += align_dot[stage_0_mask] * 0.5
        # NEW: Reward keeping gripper open (action near -1.0)
        rewards[stage_0_mask] += -self._actions[stage_0_mask, 6] * 2.0
        
        reached_snack = (dist_ee_snack < self.cfg.reach_success_dist) & stage_0_mask
        self.task_stage[reached_snack] = 1
        self.stage_timer[reached_snack] = 0.0
        rewards[reached_snack] += self.cfg.grasp_success_reward

        # --- Stage 1: Grip ---
        stage_1_mask = (self.task_stage == 1)
        # Reward closing: action[6] pushed to 1.0 (Closed)
        rewards[stage_1_mask] = self._actions[stage_1_mask, 6] * self.cfg.grasp_reward_scale
        rewards[stage_1_mask] += torch.exp(-20.0 * dist_ee_snack[stage_1_mask]) * 5.0
        
        # Transition condition: Wait 0.5s to ensure contact physics
        gripped = (self.stage_timer > 0.5) & (self._actions[:, 6] > 0.8) & stage_1_mask
        self.task_stage[gripped] = 2
        self.stage_timer[gripped] = 0.0
        rewards[gripped] += self.cfg.lift_success_reward

        # --- Stage 2: Lift ---
        stage_2_mask = (self.task_stage == 2)
        # Reward vertical height specifically
        rewards[stage_2_mask] = (snack_pos_l[stage_2_mask, 2] - 0.059) * 100.0 
        rewards[stage_2_mask] += torch.exp(-20.0 * dist_ee_snack[stage_2_mask]) * 10.0
        # Reward keeping gripper closed
        rewards[stage_2_mask] += self._actions[stage_2_mask, 6] * 5.0
        
        # Penalty for horizontal dragging while low
        drag_dist = torch.norm(snack_pos_l[stage_2_mask, :2] - torch.tensor([0.10, 0.0], device=self.device), dim=-1)
        dragging = (snack_pos_l[stage_2_mask, 2] < 0.08) & (drag_dist > 0.02)
        rewards[stage_2_mask] -= dragging.float() * 20.0
        
        # Transition: Lifted to 0.12m
        is_lifted = (snack_pos_l[:, 2] > 0.12) & stage_2_mask
        self.task_stage[is_lifted] = 3
        self.stage_timer[is_lifted] = 0.0

        # --- Stage 3: Return Home ---
        stage_3_mask = (self.task_stage == 3)
        rewards[stage_3_mask] = (1.0 - torch.tanh(dist_ee_home[stage_3_mask] / 0.2)) * self.cfg.home_reward_scale
        rewards[stage_3_mask] += (1.0 - torch.tanh(dist_joint_home[stage_3_mask] / 0.5)) * 10.0
        rewards[stage_3_mask] += torch.exp(-20.0 * dist_ee_snack[stage_3_mask]) * 10.0 # KEEP GRIP
        rewards[stage_3_mask] += self._actions[stage_3_mask, 6] * 5.0 # KEEP GRIPPER CLOSED
        
        # Strong penalty for dragging/dropping
        dropped = (snack_pos_l[stage_3_mask, 2] < 0.08)
        rewards[stage_3_mask] -= dropped.float() * 50.0
        
        # Transition: Reached home (Position AND Pose)
        reached_home = (dist_ee_home < self.cfg.home_success_dist) & \
                       (dist_joint_home < 0.15) & \
                       (snack_pos_l[:, 2] > 0.10) & stage_3_mask
        self.task_stage[reached_home] = 4
        self.stage_timer[reached_home] = 0.0

        # --- Stage 4: Hold ---
        stage_4_mask = (self.task_stage == 4)
        rewards[stage_4_mask] = self.cfg.home_reward_scale 
        rewards[stage_4_mask] += (1.0 - torch.tanh(dist_joint_home[stage_4_mask] / 0.2)) * 15.0
        rewards[stage_4_mask] += torch.exp(-20.0 * dist_ee_snack[stage_4_mask]) * 10.0
        rewards[stage_4_mask] += self._actions[stage_4_mask, 6] * 5.0 # KEEP GRIPPER CLOSED
        
        # Strong drop penalty
        dropped_penalty = ((self.task_stage >= 2) & (snack_pos_l[:, 2] < 0.05)).float() * 100.0
        
        # Action/Vel penalties
        r_act = -torch.sum(torch.square(self._actions), dim=-1) * self.cfg.action_penalty_scale
        r_vel = -torch.sum(torch.square(self.robot.data.joint_vel), dim=-1) * self.cfg.joint_vel_penalty_scale
        
        return rewards + r_act + r_vel - dropped_penalty
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        success = (self.task_stage == 4) & (self.stage_timer > 3.0)
        terminated = success
        return terminated, truncated
