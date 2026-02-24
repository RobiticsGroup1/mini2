from __future__ import annotations

import torch
import gymnasium as gym

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_rotate

from .env_cfg import DoosanE0509ReachEnvCfg


class DoosanE0509ReachEnv(DirectRLEnv):
    def __init__(self, cfg=None, render_mode: str | None = None, **kwargs):
        if cfg is None:
            cfg = DoosanE0509ReachEnvCfg()
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # 로봇 핸들
        self.robot: Articulation = self.scene["robot"]
        self.snack: RigidObject = self.scene["snack"]

        # 조인트 인덱스 (e0509: joint_1 ... joint_6)
        self.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        all_joint_names = list(self.robot.data.joint_names)
        name_to_id = {name: i for i, name in enumerate(all_joint_names)}

        missing = [n for n in self.joint_names if n not in name_to_id]
        if missing:
            raise RuntimeError(
                f"Requested joints not found: {missing}\n"
                f"Available joints: {all_joint_names}"
            )
        
        self.joint_ids = torch.tensor([name_to_id[n] for n in self.joint_names],
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

        # 타겟(각 env 당 1개, 로컬 좌표계)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # PD 타겟 추적용 변수
        self.current_targets = torch.zeros((self.num_envs, len(self.joint_ids)), device=self.device)

        # 관측/액션 스페이스 (SB3용)
        self.observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(self.cfg.obs_dim,), dtype=float
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.action_dim,), dtype=float
        )

    # ---------------
    # RL hooks
    # ---------------
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        # 타깃 설정 (고정 위치: stand x + 0.35 = 0.10)
        low = torch.tensor(self.cfg.target_min, device=self.device, dtype=torch.float32)
        high = torch.tensor(self.cfg.target_max, device=self.device, dtype=torch.float32)

        u = torch.rand((len(env_ids), 3), device=self.device, dtype=torch.float32)
        self.target_pos[env_ids] = low + (high - low) * u

        # Snack 물체를 타겟 위치로 이동
        snack_pos_w = self.target_pos[env_ids] + self.scene.env_origins[env_ids]
        snack_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        
        self.snack.write_root_pose_to_sim(torch.cat([snack_pos_w, snack_quat_w], dim=-1), env_ids=env_ids)
        self.snack.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)

        # 로봇 초기화: PD 타겟을 현재(초기) 위치로 설정하여 추락 방지
        q_init = self.robot.data.default_joint_pos[env_ids][:, self.joint_ids]
        self.current_targets[env_ids] = q_init
        self.robot.set_joint_position_target(q_init, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q_init, torch.zeros_like(q_init), joint_ids=self.joint_ids, env_ids=env_ids)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = torch.clamp(actions, -1.0, 1.0)
        self._apply_action()

    def _apply_action(self):
        # 현재 타겟 + 델타 액션
        self.current_targets += self._actions * self.cfg.action_scale
        self.robot.set_joint_position_target(self.current_targets, joint_ids=self.joint_ids)

    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos[:, self.joint_ids]
        qd = self.robot.data.joint_vel[:, self.joint_ids]
        
        # EE 위치(Local로 변환: World - Origin)
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_id, :]
        ee_pos_l = ee_pos_w - self.scene.env_origins

        obs = torch.cat([q, qd, ee_pos_l, self.target_pos], dim=-1)
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        # EE 상태(Local)
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_id, :]
        ee_pos_l = ee_pos_w - self.scene.env_origins
        ee_quat_w = self.robot.data.body_quat_w[:, self.ee_body_id, :]
        
        # 1. Reach Reward (Local Distance)
        dist = torch.linalg.norm(ee_pos_l - self.target_pos, dim=-1)
        r_reach = torch.exp(-5.0 * dist) * self.cfg.reach_reward_scale

        # 2. Align Reward (Gripper pointing down)
        down_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        ee_z_axis = quat_rotate(ee_quat_w, torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1))
        align_dot = torch.sum(ee_z_axis * down_vec, dim=-1)
        r_align = torch.clamp(align_dot, min=0.0) * self.cfg.align_reward_scale

        # 3. Penalties
        r_act = -torch.sum(torch.square(self._actions), dim=-1) * self.cfg.action_penalty_scale
        joint_vel = self.robot.data.joint_vel[:, self.joint_ids]
        r_vel = -torch.sum(torch.square(joint_vel), dim=-1) * self.cfg.joint_vel_penalty_scale

        # 4. Success bonus
        success = dist < self.cfg.reach_success_dist
        r_success = success.float() * self.cfg.success_bonus

        return r_reach + r_align + r_act + r_vel + r_success
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_id, :]
        ee_pos_l = ee_pos_w - self.scene.env_origins
        dist = torch.linalg.norm(ee_pos_l - self.target_pos, dim=-1)
        success = dist < self.cfg.reach_success_dist

        terminated = success
        return terminated, truncated
