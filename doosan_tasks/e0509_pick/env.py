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

        self.arm_joint_ids = torch.tensor(
            [name_to_id[n] for n in self.arm_joint_names],
            device=self.device,
            dtype=torch.long,
        )
        self.gripper_joint_ids = torch.tensor(
            [name_to_id[n] for n in self.gripper_joint_names],
            device=self.device,
            dtype=torch.long,
        )

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

        # Initialize extras for logging
        self.extras["log"] = dict()

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        # Snack Reset
        snack_pos_w = (
            torch.tensor(self.cfg.scene.snack.init_state.pos, device=self.device).repeat(len(env_ids), 1)
            + self.scene.env_origins[env_ids]
        )
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

    def _get_ee_tip_pos_l(self):
        flange_pos_w = self.robot.data.body_pos_w[:, self.ee_body_id, :]
        flange_quat_w = self.robot.data.body_quat_w[:, self.ee_body_id, :]
        tip_pos_w = flange_pos_w + quat_rotate(flange_quat_w, self.ee_offset)
        return tip_pos_w - self.scene.env_origins

    def _pre_physics_step(self, actions: torch.Tensor):
        # Clamp actions to valid range.
        self._actions = torch.clamp(actions, -1.0, 1.0)

        # ---------------------------------------------------------------------
        # Gripper gating by stage (demo-friendly)
        # action[6] mapping (see _apply_action):
        #   -1.0 -> Open  (0.0 joint target)
        #   +1.0 -> Close (1.1 joint target)
        # ---------------------------------------------------------------------
        stage0 = (self.task_stage == 0)  # Approach
        stage1 = (self.task_stage == 1)  # Grip
        stage2_3 = (self.task_stage >= 2) & (self.task_stage <= 3)  # Lift + Return Home

        # Stage 0: always approach with OPEN gripper.
        self._actions[stage0, 6] = -1.0

        # Stage 2~3: keep CLOSED to avoid accidental release while lifting/returning.
        self._actions[stage2_3, 6] = 1.0

        # Stage 1: allow policy to close, but only in the closing direction (reduces jitter).
        self._actions[stage1, 6] = torch.clamp(self._actions[stage1, 6], 0.0, 1.0)

        self._apply_action()

    def _apply_action(self):
        # Arm delta control
        self.current_targets[:, self.arm_joint_ids] += self._actions[:, :6] * self.cfg.action_scale

        # Gripper absolute control [0, 1.1]
        # Mapping: 1.0 (Action) -> 1.1 (Closed), -1.0 (Action) -> 0.0 (Open)
        gripper_pos = (self._actions[:, 6:7] + 1.0) / 2.0 * 1.1
        self.current_targets[:, self.gripper_joint_ids] = gripper_pos

        self.robot.set_joint_position_target(self.current_targets)

    def _get_observations(self) -> dict:
        # Capture home EE position once after first physics step.
        # _get_observations is always called post-sim.step(), so body_pos_w is valid here.
        if torch.all(self.home_ee_pos_l == 0):
            flange_pos_w = self.robot.data.body_pos_w[:, self.ee_body_id, :]
            flange_quat_w = self.robot.data.body_quat_w[:, self.ee_body_id, :]
            tip_pos_w = flange_pos_w + quat_rotate(flange_quat_w, self.ee_offset)
            self.home_ee_pos_l = tip_pos_w - self.scene.env_origins

        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel
        ee_pos_l = self._get_ee_tip_pos_l()
        snack_pos_l = self.snack.data.root_pos_w[:, :3] - self.scene.env_origins
        ee_to_snack_l = snack_pos_l - ee_pos_l

        obs = torch.cat(
            [
                q,
                qd,
                ee_pos_l,
                snack_pos_l,
                self.home_ee_pos_l,
                ee_to_snack_l,
                self.task_stage.float().unsqueeze(-1),
                (self.stage_timer / self.cfg.episode_length_s).unsqueeze(-1),
            ],
            dim=-1,
        )
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
        ee_z_axis = quat_rotate(
            ee_quat_w,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
        )
        align_dot = torch.clamp(torch.sum(ee_z_axis * down_vec, dim=-1), min=0.0)

        # Velocities for "slow approach"
        ee_vel_w = self.robot.data.body_lin_vel_w[:, self.ee_body_id, :]
        ee_speed = torch.norm(ee_vel_w, dim=-1)

        # Gripper state
        gripper_q = self.robot.data.joint_pos[:, self.gripper_joint_ids]
        gripper_closed = gripper_q.mean(dim=-1) > 0.7

        rewards = torch.zeros(self.num_envs, device=self.device)

        # ── 전환 조건을 현재 stage 기준으로 미리 계산 (아직 task_stage 변경 안 함) ──
        stage_0_mask = (self.task_stage == 0)
        stage_1_mask = (self.task_stage == 1)
        stage_2_mask = (self.task_stage == 2)
        stage_3_mask = (self.task_stage == 3)
        stage_4_mask = (self.task_stage == 4)

        reached_snack = (dist_ee_snack < self.cfg.reach_success_dist) & stage_0_mask
        gripped = (self.stage_timer > 0.5) & gripper_closed & stage_1_mask
        is_lifted = (snack_pos_l[:, 2] > 0.12) & stage_2_mask
        reached_home = (
            (dist_ee_home < self.cfg.home_success_dist)
            & (dist_joint_home < 0.15)
            & (snack_pos_l[:, 2] > 0.10)
            & stage_3_mask
        )

        # --- Stage 0: Approach ---
        # exp(-4 * dist): 시작 거리(~0.35m)에서도 0.25의 gradient가 존재
        # 기존 exp(-20 * dist)는 0.35m에서 0.001로 사실상 gradient 없음
        rewards[stage_0_mask] = torch.exp(-4.0 * dist_ee_snack[stage_0_mask]) * self.cfg.reach_reward_scale

        # Alignment reward (그리퍼가 아래를 향하도록, 최종 접근 자세 유도)
        rewards[stage_0_mask] += align_dot[stage_0_mask] * 2.0

        # Penalty for high speed during approach
        rewards[stage_0_mask] -= torch.clamp(ee_speed[stage_0_mask] - 0.5, min=0.0) * 2.0
        # Extra penalty for high speed when very close
        near_snack = (dist_ee_snack < 0.1) & stage_0_mask
        rewards[near_snack] -= torch.clamp(ee_speed[near_snack] - 0.1, min=0.0) * 10.0

        # Penalty for horizontal offset (수직 접근 유도, 3D distance와 겹치므로 약하게)
        horiz_dist = torch.norm(ee_pos_l[stage_0_mask, :2] - snack_pos_l[stage_0_mask, :2], dim=-1)
        rewards[stage_0_mask] -= horiz_dist * 2.0

        # --- Stage 1: Grip ---
        rewards[stage_1_mask] = self._actions[stage_1_mask, 6] * self.cfg.grasp_reward_scale
        rewards[stage_1_mask] += torch.exp(-20.0 * dist_ee_snack[stage_1_mask]) * 5.0
        rewards[stage_1_mask] += align_dot[stage_1_mask] * 2.0

        # --- Stage 2: Lift ---
        rewards[stage_2_mask] = torch.clamp(snack_pos_l[stage_2_mask, 2] - 0.059, min=0.0) * self.cfg.lift_reward_scale
        rewards[stage_2_mask] += self._actions[stage_2_mask, 6] * 10.0

        # --- Stage 3: Return Home ---
        rewards[stage_3_mask] = (1.0 - torch.tanh(dist_ee_home[stage_3_mask] / 0.2)) * self.cfg.home_reward_scale
        rewards[stage_3_mask] += (1.0 - torch.tanh(dist_joint_home[stage_3_mask] / 0.5)) * 10.0
        rewards[stage_3_mask] += self._actions[stage_3_mask, 6] * 20.0  # keep closed

        # --- Stage 4: Release (open gripper at home) ---
        rewards[stage_4_mask] = self.cfg.home_reward_scale * 0.5
        open_score = (1.0 - self._actions[stage_4_mask, 6]) * 0.5  # open=1, close=0
        rewards[stage_4_mask] += open_score * 10.0

        # ── 모든 stage 보상 적용 후 전환 실행 + 보너스 추가 ──
        # 이 시점에 += 하므로 위의 = 할당에 덮어써지지 않음
        self.task_stage[reached_snack] = 1
        self.stage_timer[reached_snack] = 0.0
        rewards[reached_snack] += self.cfg.grasp_success_reward

        self.task_stage[gripped] = 2
        self.stage_timer[gripped] = 0.0
        rewards[gripped] += self.cfg.lift_success_reward

        self.task_stage[is_lifted] = 3
        self.stage_timer[is_lifted] = 0.0

        self.task_stage[reached_home] = 4
        self.stage_timer[reached_home] = 0.0

        # Drop penalty
        drop = ((self.task_stage >= 2) & (snack_pos_l[:, 2] < 0.05))
        rewards[drop] -= 50.0

        # Action/Vel penalties
        a = self._actions.clone()
        a[:, 6] = 0.0  # do not penalize gripper command (we gate it by stage)
        r_act = -torch.sum(torch.square(a), dim=-1) * self.cfg.action_penalty_scale
        r_vel = -torch.sum(torch.square(self.robot.data.joint_vel), dim=-1) * self.cfg.joint_vel_penalty_scale

        # Log success rate (scalar): Stage 4 + gripper actually open + held for a bit
        gripper_open = gripper_q.mean(dim=-1) < 0.2
        success = ((self.task_stage == 4) & (self.stage_timer > 0.5) & gripper_open).float()
        self.extras["log"]["success_rate"] = success.mean()

        return rewards + r_act + r_vel

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        snack_pos_l = self.snack.data.root_pos_w[:, :3] - self.scene.env_origins
        drop = ((self.task_stage >= 2) & (snack_pos_l[:, 2] < 0.05))

        gripper_q = self.robot.data.joint_pos[:, self.gripper_joint_ids]
        gripper_open = gripper_q.mean(dim=-1) < 0.2
        success = (self.task_stage == 4) & (self.stage_timer > 0.5) & gripper_open

        terminated = success | drop
        return terminated, truncated