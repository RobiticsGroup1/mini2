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
        eps = 1e-8 # NaN 방지를 위한 미소값

        # 1. 좌표 및 상태 업데이트 (벡터화 유지)
        ee_pos_l = self._get_ee_tip_pos_l()
        snack_pos_l = self.snack.data.root_pos_w[:, :3] - self.scene.env_origins
        ee_quat_w = self.robot.data.body_quat_w[:, self.ee_body_id, :]

        # 거리 계산 (eps 추가로 NaN 방지)
        dist_ee_snack = torch.norm(ee_pos_l - snack_pos_l, dim=-1) + eps
        dist_ee_home = torch.norm(ee_pos_l - self.home_ee_pos_l, dim=-1) + eps

        arm_q = self.robot.data.joint_pos[:, self.arm_joint_ids]
        dist_joint_home = torch.norm(arm_q - self.default_arm_q, dim=-1) + eps

        # 그리퍼 방향 (아래를 향하도록)
        down_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        ee_z_axis = quat_rotate(
            ee_quat_w,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
        )
        align_dot = torch.clamp(torch.sum(ee_z_axis * down_vec, dim=-1), min=0.0, max=1.0)

        # 속도 제어
        ee_vel_w = self.robot.data.body_lin_vel_w[:, self.ee_body_id, :]
        ee_speed = torch.norm(ee_vel_w, dim=-1)

        # 그리퍼 상태
        gripper_q = self.robot.data.joint_pos[:, self.gripper_joint_ids]
        gripper_closed = gripper_q.mean(dim=-1) > 0.7 

        rewards = torch.zeros(self.num_envs, device=self.device)

        # 마스크 설정
        stage_masks = [self.task_stage == i for i in range(5)]

        # --- Stage 0: Approach ---
        # 수평 거리를 더 엄격하게 벌점 주어 위에서 아래로 접근하게 유도
        horiz_dist = torch.norm(ee_pos_l[:, :2] - snack_pos_l[:, :2], dim=-1)
        rewards[stage_masks[0]] = torch.exp(-5.0 * dist_ee_snack[stage_masks[0]]) * self.cfg.reach_reward_scale
        rewards[stage_masks[0]] += align_dot[stage_masks[0]] * 5.0
        rewards[stage_masks[0]] -= horiz_dist[stage_masks[0]] * 3.0 # 수평 오차 벌점 강화

        # --- Stage 1: Grip (강화됨) ---
        # 단순히 닫는 명령뿐 아니라, 닫았을 때 물체와 손의 거리가 가까워야 보상
        grip_command = self._actions[:, 6]
        rewards[stage_masks[1]] = grip_command[stage_masks[1]] * self.cfg.grasp_reward_scale
        rewards[stage_masks[1]] += (dist_ee_snack[stage_masks[1]] < 0.05).float() * 10.0 # 근접 보너스

        # --- Stage 2: Lift (강화됨) ---
        # 물체가 들렸더라도 손에서 멀어지면(튕겨나가면) 보상 삭제
        is_holding = dist_ee_snack < 0.08
        lift_height = torch.clamp(snack_pos_l[:, 2] - 0.059, min=0.0)
        rewards[stage_masks[2]] = lift_height[stage_masks[2]] * self.cfg.lift_reward_scale * is_holding[stage_masks[2]].float()
        rewards[stage_masks[2]] -= (~is_holding[stage_masks[2]]).float() * 20.0 # 놓치면 벌점

        # --- Stage 3: Return Home ---
        # 물체를 유지한 채로 복귀할 때만 높은 보상
        rewards[stage_masks[3]] = (1.0 - torch.tanh(dist_ee_home[stage_masks[3]] / 0.3)) * self.cfg.home_reward_scale
        rewards[stage_masks[3]] += is_holding[stage_masks[3]].float() * 20.0 # 유지 보상

        # --- Stage 4: Release ---
        rewards[stage_masks[4]] = self.cfg.home_reward_scale
        open_score = (1.0 - grip_command[stage_masks[4]]) # 1에 가까울수록 잘 연 것
        rewards[stage_masks[4]] += open_score * 15.0

        # ── 전환 로직 (안전 장치 추가) ──
        # 0 -> 1: 충분히 접근했을 때
        reached_snack = (dist_ee_snack < self.cfg.reach_success_dist) & stage_masks[0]
        self.task_stage[reached_snack] = 1
        self.stage_timer[reached_snack] = 0.0
        rewards[reached_snack] += self.cfg.grasp_success_reward

        # 1 -> 2: 닫기 시작한 후 약간의 시간 대기 (물리 안정화)
        gripped = (self.stage_timer > 0.3) & gripper_closed & stage_masks[1]
        self.task_stage[gripped] = 2
        self.stage_timer[gripped] = 0.0
        rewards[gripped] += self.cfg.lift_success_reward

        # 2 -> 3: 실제로 물체가 들렸고 + 여전히 손에 있을 때만 전환
        is_lifted = (snack_pos_l[:, 2] > 0.15) & is_holding & stage_masks[2]
        self.task_stage[is_lifted] = 3
        self.stage_timer[is_lifted] = 0.0

        # 3 -> 4: 홈 포지션 근처 도달
        reached_home = (dist_ee_home < self.cfg.home_success_dist) & stage_masks[3]
        self.task_stage[reached_home] = 4
        self.stage_timer[reached_home] = 0.0

        # 공통 벌점 (추락 및 행동 제약)
        drop = ((self.task_stage >= 2) & (snack_pos_l[:, 2] < 0.06))
        rewards[drop] -= 100.0 # 추락 벌점 강화

        r_act = -torch.sum(torch.square(self._actions), dim=-1) * self.cfg.action_penalty_scale
        r_vel = -torch.sum(torch.square(self.robot.data.joint_vel), dim=-1) * self.cfg.joint_vel_penalty_scale

        # 성공 판정 로그
        success = ((self.task_stage == 4) & (self.stage_timer > 0.5)).float()
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