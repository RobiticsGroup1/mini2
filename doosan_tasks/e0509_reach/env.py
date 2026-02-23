from __future__ import annotations

import math
import torch
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv
from isaaclab.utils.math import sample_uniform
from isaaclab.assets import Articulation

from .env_cfg import DoosanE0509ReachEnvCfg


class DoosanE0509ReachEnv(DirectRLEnv):
    def __init__(self, cfg=None, render_mode: str | None = None, **kwargs):
        if cfg is None:
            cfg = DoosanE0509ReachEnvCfg()
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # 로봇 핸들
        self.robot: Articulation = self.scene["robot"]

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

        # 타겟(각 env 당 1개)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # 관측/액션 스페이스 (SB3용)
        self.observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(self.cfg.obs_dim,), dtype=float
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.action_dim,), dtype=float
        )

        # debug
        # print("=== robot cfg.prim_path ===", self.robot.cfg.prim_path)
        # print("=== joint names ===", self.robot.data.joint_names)
        # print("=== body names ===", self.robot.data.body_names)
        """
        === robot cfg.prim_path === /World/envs/env_.*/e0509
        === joint names === ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        === body names === ['base_link', 'base', 'link_1', 'link_2', 'link_3', 'link_4', 'link_5', 'link_6', 'tool0']
        """

    # ---------------
    # RL hooks
    # ---------------
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        # 타깃 샘플
        low = torch.tensor(self.cfg.target_min, device=self.device, dtype=torch.float32)
        high = torch.tensor(self.cfg.target_max, device=self.device, dtype=torch.float32)

        u = torch.rand((len(env_ids), 3), device=self.device, dtype=torch.float32)
        self.target_pos[env_ids] = low + (high - low) * u

        # 로봇 초기화(조인트는 cfg.init_state 기반으로 이미 세팅되지만, 리셋마다 조금 랜덤도 가능)
        # 간단히 속도 0으로
        # zeros = torch.zeros((len(env_ids), len(self.joint_ids)), device=self.device)
        # self.robot.set_joint_velocities(zeros, joint_ids=self.joint_ids, env_ids=env_ids)

    def _pre_physics_step(self, actions: torch.Tensor):
        # actions: [-1, 1] -> joint position target delta
        self._actions = torch.clamp(actions, -1.0, 1.0)

        # # 현재 조인트 포지션
        # # q = self.robot.data.joint_pos[:, self.joint_ids]
        # q = self.robot.data.joint_pos[env_ids][:, self.joint_ids]
        # q_des = q + actions * self.cfg.action_scale

        # # position target으로 구동
        # # self.robot.set_joint_position_target(q_des, joint_ids=self.joint_ids)
        # self.robot.set_joint_position_target(q, joint_ids=self.joint_ids, env_ids=env_ids)

    def _apply_action(self):
        # 현재 조인트각
        q = self.robot.data.joint_pos[:, self.joint_ids]  # (num_envs, dof)
        q_des = q + self._actions * self.cfg.action_scale

        self.robot.set_joint_position_target(q_des, joint_ids=self.joint_ids)

        # # Isaac Lab v2.3.2에서 set_joint_position_target이 없을 수 있으니 안전하게 처리
        # if hasattr(self.robot, "set_joint_position_target"):
        #     self.robot.set_joint_position_target(q_des, joint_ids=self.joint_ids)
        # elif hasattr(self.robot, "write_joint_position_target_to_sim"):
        #     self.robot.write_joint_position_target_to_sim(q_des, self.joint_ids=self.joint_ids)
        # else:
        #     raise AttributeError("No API found to write joint position targets to simulation.")

    def _get_observations(self) -> dict:
        # q, qd
        q = self.robot.data.joint_pos[:, self.joint_ids]
        qd = self.robot.data.joint_vel[:, self.joint_ids]

        # EE 위치(World)
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_id, :]

        obs = torch.cat([q, qd, ee_pos_w, self.target_pos], dim=-1)
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_id, :]
        dist = torch.linalg.norm(ee_pos_w - self.target_pos, dim=-1)

        # 거리 보상(가까울수록 큼)
        r_dist = -dist * self.cfg.w_dist

        # 액션 패널티(부드럽게)
        r_act = -(self._actions**2).sum(dim=-1) * self.cfg.w_action

        # 성공 보너스
        success = dist < self.cfg.success_dist
        r_success = success.float() * self.cfg.success_bonus

        return r_dist + r_act + r_success
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # time-out
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 성공 종료
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_id, :]
        dist = torch.linalg.norm(ee_pos_w - self.target_pos, dim=-1)
        success = dist < self.cfg.success_dist

        terminated = success
        truncated = time_out
        return terminated, truncated
