from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

def get_tcp_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Calculate the Tool Center Point (TCP) pose.
    
    The TCP is located approximately 10cm from link_6 along the local Z-axis.
    """
    from isaaclab.utils.math import quat_apply
    
    robot = env.scene[asset_cfg.name]
    # link_6 position and orientation
    ee_pos = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    ee_quat = robot.data.body_quat_w[:, asset_cfg.body_ids[0]]
    
    # TCP offset in link_6 local frame: Corrected to 10cm
    tcp_offset = torch.tensor([0.0, 0.0, 0.10], device=env.device).repeat(env.num_envs, 1)
    
    # Transform offset to world frame
    tcp_pos = ee_pos + quat_apply(ee_quat, tcp_offset)
    
    return tcp_pos

def ee_distance_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for end-effector (TCP) distance to target."""
    tcp_pos = get_tcp_pose(env, asset_cfg)
    
    snack = env.scene["snack"]
    target_pos = snack.data.root_pos_w[:, :3].clone()
    target_pos[:, 2] += 0.048 / 2.0 + 0.03
    
    distance = torch.norm(tcp_pos - target_pos, dim=-1)
    
    # Stricter Gaussian reward
    return torch.exp(-10.0 * distance)

def ee_orientation_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for keeping the gripper perfectly vertical (pointing downward)."""
    robot = env.scene[asset_cfg.name]
    ee_quat = robot.data.body_quat_w[:, asset_cfg.body_ids[0]]
    
    from isaaclab.utils.math import quat_apply
    
    # Current pointing vector
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    ee_pointing_axis = quat_apply(ee_quat, z_axis)
    
    # Desired vector (Downward)
    down_vector = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    dot_product = torch.sum(ee_pointing_axis * down_vector, dim=-1)
    
    # Balanced strictness (Reduced from 100.0 to 50.0 for stability)
    return torch.exp(50.0 * (dot_product - 1.0))

def reached_target_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.02) -> torch.Tensor:
    """Bonus for TCP reaching the target."""
    tcp_pos = get_tcp_pose(env, asset_cfg)
    
    snack = env.scene["snack"]
    target_pos = snack.data.root_pos_w[:, :3].clone()
    target_pos[:, 2] += 0.048 / 2.0 + 0.03
    
    distance = torch.norm(tcp_pos - target_pos, dim=-1)
    return (distance < threshold).float()

def halt_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for TCP staying still when reached the target."""
    tcp_pos = get_tcp_pose(env, asset_cfg)
    robot = env.scene[asset_cfg.name]
    ee_vel = robot.data.body_vel_w[:, asset_cfg.body_ids[0], :3]
    
    snack = env.scene["snack"]
    target_pos = snack.data.root_pos_w[:, :3].clone()
    target_pos[:, 2] += 0.048 / 2.0 + 0.03
    
    distance = torch.norm(tcp_pos - target_pos, dim=-1)
    velocity_norm = torch.norm(ee_vel, dim=-1)
    
    close_to_target = (distance < 0.05).float()
    return close_to_target * torch.exp(-2.0 * velocity_norm)

def gripper_open_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for keeping the gripper open."""
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    mean_pos = torch.mean(torch.abs(joint_pos), dim=-1)
    return torch.exp(-5.0 * mean_pos)

def dummy_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """A dummy reward function."""
    return torch.zeros(env.num_envs, device=env.device)
