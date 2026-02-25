from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

def reaching_ee_snack_l2(env: ManagerBasedRLEnv, robot_cfg: str, object_cfg: str) -> torch.Tensor:
    """Reward for the distance between the end-effector and the object."""
    robot = env.scene[robot_cfg]
    obj = env.scene[object_cfg]
    
    # We'll use the 7th body (link_6) as the EE for distance calculation
    # In practice, SceneEntityCfg provides body_ids, but here we simplify for the example
    ee_pos_w = robot.data.body_pos_w[:, -1, :] 
    obj_pos_w = obj.data.root_pos_w[:, :3]
    
    distance = torch.norm(obj_pos_w - ee_pos_w, dim=-1)
    return torch.exp(-5.0 * distance)

def object_is_lifted(env: ManagerBasedRLEnv, object_cfg: str, threshold: float = 0.05) -> torch.Tensor:
    """Reward for lifting the object above a certain height."""
    obj = env.scene[object_cfg]
    obj_pos_z = obj.data.root_pos_w[:, 2]
    # Initial height was 0.059
    return (obj_pos_z > (0.059 + threshold)).float()

def gripper_is_closed(env: ManagerBasedRLEnv, robot_cfg: str, open_threshold: float = 0.5) -> torch.Tensor:
    """Reward for keeping the gripper closed (joint values near 0)."""
    robot = env.scene[robot_cfg]
    # rh_l1, rh_l2, rh_r1, rh_r2 are the last 4 joints
    gripper_q = robot.data.joint_pos[:, -4:] 
    # Open is 1.1, Closed is 0.0. We reward values below threshold.
    return torch.mean((gripper_q < open_threshold).float(), dim=-1)

def ee_alignment_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for pointing the gripper tip (Z-axis) downwards."""
    robot = env.scene[robot_cfg.name]
    # Get EE orientation (using the body index provided by SceneEntityCfg or assumed)
    ee_quat = robot.data.body_quat_w[:, robot_cfg.body_ids[0], :]
    
    # Current Z-axis vector of the EE (pointing out of the flange)
    from isaaclab.utils.math import quat_rotate
    ee_z_axis = quat_rotate(ee_quat, torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1))
    
    # Target vector (World Down)
    down_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    return torch.clamp(torch.sum(ee_z_axis * down_vec, dim=-1), min=0.0)

def object_carrying_reward(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for the object being near the robot's home position."""
    obj = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w[:, :3]
    
    # Home position is the origin of the robot: (-0.25, 0.0, 0.066) in local env
    # We add the environment origins to get world coordinates
    home_pos_w = env.scene.env_origins + torch.tensor([-0.25, 0.0, 0.066], device=env.device)
    
    dist = torch.norm(home_pos_w - obj_pos, dim=-1)
    return torch.exp(-2.0 * dist)
