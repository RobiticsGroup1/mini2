from __future__ import annotations

import math
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

        # Doosan E0509 joint limits: all axes ±360° = ±2π rad
        self.joint_limit_rad = math.pi * 2.0

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
        # Gripper gating by stage (2-stage)
        # action[6] mapping: -1.0 -> Open (0.0 rad), +1.0 -> Close (1.1 rad)
        # Stage 0: Approach → always OPEN
        # Stage 1: Lift     → always CLOSED
        # ---------------------------------------------------------------------
        self._actions[self.task_stage == 0, 6] = -1.0
        self._actions[self.task_stage == 1, 6] = 1.0

        self._apply_action()

    def _apply_action(self):
        # Arm delta control
        self.current_targets[:, self.arm_joint_ids] += self._actions[:, :6] * self.cfg.action_scale

        # Hard clamp to ±360° joint limits (Doosan E0509 spec)
        self.current_targets[:, self.arm_joint_ids] = torch.clamp(
            self.current_targets[:, self.arm_joint_ids],
            -self.joint_limit_rad,
            self.joint_limit_rad,
        )

        # Gripper absolute control [0, 1.1]
        # Mapping: +1.0 (Action) -> 1.1 rad (Closed), -1.0 (Action) -> 0.0 rad (Open)
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

        # EE z-axis in world frame: (0,0,-1) = gripper pointing straight down
        ee_quat_w = self.robot.data.body_quat_w[:, self.ee_body_id, :]
        ee_z_axis = quat_rotate(
            ee_quat_w,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
        )

        obs = torch.cat(
            [
                q,                                                    # 10
                qd,                                                   # 10
                ee_pos_l,                                             #  3
                snack_pos_l,                                          #  3
                self.home_ee_pos_l,                                   #  3
                ee_to_snack_l,                                        #  3
                ee_z_axis,                                            #  3  ← 그리퍼 방향
                self.task_stage.float().unsqueeze(-1),                #  1
                (self.stage_timer / self.cfg.episode_length_s).unsqueeze(-1),  #  1
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self.stage_timer += self.cfg.sim.dt * self.cfg.decimation
        eps = 1e-8

        # ── 공통 상태 계산 ──
        ee_pos_l = self._get_ee_tip_pos_l()
        snack_pos_l = self.snack.data.root_pos_w[:, :3] - self.scene.env_origins
        dist_ee_snack = torch.norm(ee_pos_l - snack_pos_l, dim=-1) + eps

        arm_q = self.robot.data.joint_pos[:, self.arm_joint_ids]

        # 그리퍼 하향 정렬 (위에서 접근 유도)
        ee_quat_w = self.robot.data.body_quat_w[:, self.ee_body_id, :]
        ee_z_axis = quat_rotate(
            ee_quat_w,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
        )
        down_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        align_dot = torch.clamp(torch.sum(ee_z_axis * down_vec, dim=-1), min=0.0, max=1.0)

        # 속도
        ee_vel_w = self.robot.data.body_lin_vel_w[:, self.ee_body_id, :]
        ee_speed = torch.norm(ee_vel_w, dim=-1)

        # 물체 hold 여부 및 lift
        is_holding = dist_ee_snack < 0.08
        lift_height = torch.clamp(snack_pos_l[:, 2] - 0.059, min=0.0)

        # 거리 가중 align: 멀 때(>0.2m)는 0, 가까울수록 최대 3.0
        align_near = align_dot * torch.clamp(1.0 - dist_ee_snack / 0.2, min=0.0) * 3.0

        # 스낵 정위에서 접근 유도: xy 정렬 + EE가 스낵보다 위에 있을 때 보상
        xy_dist_to_snack = torch.norm(ee_pos_l[:, :2] - snack_pos_l[:, :2], dim=-1) + eps
        above_snack = (ee_pos_l[:, 2] > snack_pos_l[:, 2]).float()
        above_reward = torch.exp(-6.0 * xy_dist_to_snack) * above_snack * 5.0

        rewards = torch.zeros(self.num_envs, device=self.device)
        stage_masks = [self.task_stage == i for i in range(2)]

        # ── Stage 0: Approach (그리퍼 열린 채 접근) ──
        # 거리 보상에 align_dot을 곱함: 수평 접근(align_dot≈0)이면 보상 자체가 0
        # above_reward: 스낵 바로 위에서 내려오는 접근 유도
        rewards[stage_masks[0]] = (
            torch.exp(-4.0 * dist_ee_snack[stage_masks[0]]) * self.cfg.reach_reward_scale * align_dot[stage_masks[0]]
            + align_near[stage_masks[0]]
            + above_reward[stage_masks[0]]
            - torch.clamp(ee_speed[stage_masks[0]] - 0.5, min=0.0) * 2.0
        )

        # ── Stage 1: Lift (그리퍼 강제 닫힘, 들어올리기) ──
        # 접근 보상 제거: exp(-4*dist)*10 은 "snack 위에 누우면 최대" 라는 exploit 유발
        # 대신 is_holding 여부로만 구분: 잡고 있으면 소정 보상 + lift, 못 잡으면 소정 패널티
        rewards[stage_masks[1]] = (
            lift_height[stage_masks[1]] * self.cfg.lift_reward_scale * is_holding[stage_masks[1]].float()
            + is_holding[stage_masks[1]].float() * 2.0
            - (~is_holding[stage_masks[1]]).float() * 2.0
        )

        # ── 전환 로직 ──
        # 자발 전환: dist < 0.06m + 그리퍼가 스낵 위에서 내려오는 자세일 때만 전환
        # (align_dot > 0.5: 그리퍼 하향, ee_above: EE가 스낵보다 위)
        ee_above = ee_pos_l[:, 2] > snack_pos_l[:, 2] - 0.01
        reached_snack = (dist_ee_snack < self.cfg.reach_success_dist) & stage_masks[0] & (align_dot > 0.5) & ee_above
        self.task_stage[reached_snack] = 1
        self.stage_timer[reached_snack] = 0.0
        rewards[reached_snack] += self.cfg.grasp_success_reward

        # 강제 전환: 3초(≈90스텝) 후 자세 무관하게 Stage 1 진입 → Stage 0 회피 불가
        force_stage1 = (self.stage_timer > 3.0) & (self.task_stage == 0)
        self.task_stage[force_stage1] = 1
        self.stage_timer[force_stage1] = 0.0

        # ── 공통 벌점 ──
        # Stage 1 중 snack 추락
        drop = (self.task_stage == 1) & (snack_pos_l[:, 2] < 0.045)
        rewards[drop] -= 100.0

        # 관절 한계 소프트 벌점 (±360° 도달 전 경고 구역: soft limit ≈ ±331°)
        soft_limit = self.joint_limit_rad - 0.5  # ≈ 5.78 rad
        joint_excess = torch.clamp(torch.abs(arm_q) - soft_limit, min=0.0)
        r_joint_limit = -torch.sum(joint_excess, dim=-1) * self.cfg.joint_limit_penalty_scale

        r_act = -torch.sum(torch.square(self._actions), dim=-1) * self.cfg.action_penalty_scale
        r_vel = -torch.sum(torch.square(self.robot.data.joint_vel), dim=-1) * self.cfg.joint_vel_penalty_scale

        # 성공 보너스: 성공 시 즉시 큰 보너스 지급 (호버링보다 성공이 유리하게)
        success = (self.task_stage == 1) & (snack_pos_l[:, 2] > self.cfg.lift_success_height) & is_holding
        rewards[success] += self.cfg.lift_success_reward

        # 로그
        self.extras["log"]["success_rate"] = success.float().mean()
        self.extras["log"]["stage_mean"] = self.task_stage.float().mean()

        # 진단용 콘솔 출력 (5000 스텝마다)
        if not hasattr(self, "_diag_step"):
            self._diag_step = 0
        self._diag_step += 1
        if self._diag_step % 5000 == 0:
            stage_counts = [(self.task_stage == i).sum().item() for i in range(2)]
            total = self.num_envs
            print(
                f"[diag step={self._diag_step}] "
                f"stage0={stage_counts[0]/total:.1%} "
                f"stage1={stage_counts[1]/total:.1%} | "
                f"dist_mean={dist_ee_snack.mean():.3f}m | "
                f"align_dot={align_dot.mean():.3f} | "
                f"success={success.float().mean():.3%}"
            )

        return rewards + r_act + r_vel + r_joint_limit

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        snack_pos_l = self.snack.data.root_pos_w[:, :3] - self.scene.env_origins

        # Stage 1 중 snack 추락 → 실패 종료
        drop = (self.task_stage == 1) & (snack_pos_l[:, 2] < 0.045)

        # Stage 1: snack을 target 높이까지 들어올리고 + 여전히 잡고 있으면 성공 종료
        ee_pos_l = self._get_ee_tip_pos_l()
        dist_ee_snack = torch.norm(ee_pos_l - snack_pos_l, dim=-1)
        is_holding = dist_ee_snack < 0.08
        success = (self.task_stage == 1) & (snack_pos_l[:, 2] > self.cfg.lift_success_height) & is_holding

        terminated = success | drop
        return terminated, truncated