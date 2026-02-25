from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

def reaching_ee_snack_l2(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for the distance between the end-effector and the object."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    
    ee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] 
    obj_pos_w = obj.data.root_pos_w[:, :3]
    
    distance = torch.norm(obj_pos_w - ee_pos_w, dim=-1)
    return torch.exp(-20.0 * distance)

def object_is_lifted(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, threshold: float = 0.05) -> torch.Tensor:
    """Reward for lifting the object above a certain height."""
    obj = env.scene[object_cfg.name]
    obj_pos_z = obj.data.root_pos_w[:, 2]
    return (obj_pos_z > (0.059 + threshold)).float()

def gripper_is_closed(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, open_threshold: float = 0.5) -> torch.Tensor:
    """Reward for keeping the gripper closed (joint values near 1.1)."""
    robot = env.scene[robot_cfg.name]
    gripper_q = robot.data.joint_pos[:, robot_cfg.joint_ids] 
    return torch.mean((gripper_q > open_threshold).float(), dim=-1)

def ee_alignment_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for pointing the gripper tip (Z-axis) downwards."""
    robot = env.scene[robot_cfg.name]
    ee_quat = robot.data.body_quat_w[:, robot_cfg.body_ids[0], :]
    
    from isaaclab.utils.math import quat_rotate
    ee_z_axis = quat_rotate(ee_quat, torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1))
    down_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    return torch.clamp(torch.sum(ee_z_axis * down_vec, dim=-1), min=0.0)

def object_carrying_reward(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for the object being near the robot's home position."""
    obj = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w[:, :3]
    home_pos_w = env.scene.env_origins + torch.tensor([-0.25, 0.0, 0.5], device=env.device)
    dist = torch.norm(home_pos_w - obj_pos, dim=-1)
    return torch.exp(-2.0 * dist)

# --- Observation Terms ---

def ee_pos_rel(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector position in environment frame."""
    robot = env.scene[robot_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    return ee_pos_w - env.scene.env_origins

def snack_pos_rel(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Snack position in environment frame."""
    obj = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, :3] - env.scene.env_origins

def ee_to_snack_rel(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Vector from end-effector to snack."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    return obj.data.root_pos_w[:, :3] - ee_pos_w
