from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

def reaching_ee_snack_l2(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Strong linear reward to pull the arm toward the target grasp point."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] 
    
    # Target is 15cm above the snack center to account for gripper length
    target_pos_w = obj.data.root_pos_w[:, :3].clone()
    target_pos_w[:, 2] += 0.15
    
    distance = torch.norm(target_pos_w - ee_pos_w, dim=-1)
    
    # Linear pull: 1.0 when distance is 0, decreasing linearly.
    # We use a broad inverse to ensure the gradient never vanishes.
    return 1.0 / (1.0 + 10.0 * distance)

def ee_snack_xy_dist_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for alignment in XY plane between the end-effector and the object.
    Increased exponent from -20.0 to -50.0 for extreme precision.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] 
    obj_pos_w = obj.data.root_pos_w[:, :3]
    xy_distance = torch.norm(obj_pos_w[:, :2] - ee_pos_w[:, :2], dim=-1)
    return torch.exp(-50.0 * xy_distance)

def ee_snack_z_dist_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, target_z_offset: float = 0.0) -> torch.Tensor:
    """Reward for alignment in Z axis. We want gripper to surround the snack."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] 
    obj_pos_w = obj.data.root_pos_w[:, :3]
    # target_z_offset should be roughly half of snack height + gripper finger length
    z_distance = torch.abs(obj_pos_w[:, 2] + target_z_offset - ee_pos_w[:, 2])
    return torch.exp(-20.0 * z_distance)

def ee_alignment_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Vertical alignment reward using a smooth cosine similarity.
    Provides a constant gradient for the policy to learn alignment.
    """
    robot = env.scene[robot_cfg.name]
    ee_quat = robot.data.body_quat_w[:, robot_cfg.body_ids[0], :]
    from isaaclab.utils.math import quat_rotate
    ee_z_axis = quat_rotate(ee_quat, torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1))
    down_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    dot_product = torch.sum(ee_z_axis * down_vec, dim=-1)
    
    # Scale from 0 to 1 (0 when pointing up, 1 when pointing down)
    return 0.5 * (1.0 + dot_product)

def gripper_open_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for keeping the gripper open ONLY during the approach phase.
    If the robot is close to the snack, this reward drops to zero.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    gripper_q = robot.data.joint_pos[:, robot_cfg.joint_ids] 
    
    ee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] 
    
    # Check distance to the grasp point, not the snack center!
    target_pos_w = obj.data.root_pos_w[:, :3].clone()
    target_pos_w[:, 2] += 0.15
    distance = torch.norm(target_pos_w - ee_pos_w, dim=-1)
    
    # Only reward being open if distance > 10cm from the ideal grasp point
    is_far = distance > 0.10
    return is_far.float() * torch.exp(-5.0 * torch.mean(torch.abs(gripper_q), dim=-1))

def pre_grasp_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Active Closing Reward: Encourages the robot to actually close fingers when arrived.
    Uses a smooth arrival weight to provide a continuous gradient.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] 
    obj_pos_w = obj.data.root_pos_w[:, :3]
    gripper_q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    
    # Target height (15cm wrist offset)
    dist_xy = torch.norm(obj_pos_w[:, :2] - ee_pos_w[:, :2], dim=-1)
    dist_z = torch.abs(obj_pos_w[:, 2] + 0.15 - ee_pos_w[:, 2])
    
    # Smooth arrival weight (Gaussian-like)
    # Becomes significant when within ~10cm and peaks at 0cm distance
    arrival_weight = torch.exp(-15.0 * (dist_xy + dist_z))
    
    # Reward for HIGH joint values (closed) weighted by how 'arrived' we are
    return arrival_weight * torch.mean(gripper_q, dim=-1)

def object_is_grasped_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward if the object is lifted AND the gripper is not fully closed."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    gripper_q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    obj_pos_z = obj.data.root_pos_w[:, 2]
    
    # Fully closed is roughly 1.1. If 0.3 < q < 1.0, it's likely holding the snack.
    is_holding = (torch.mean(gripper_q, dim=-1) > 0.2) & (torch.mean(gripper_q, dim=-1) < 1.0)
    is_lifted = obj_pos_z > 0.08 # Snack starting height is 0.059
    
    return (is_holding & is_lifted).float()

def object_carrying_reward(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for returning the lifted object to home position."""
    obj = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w[:, :3]
    home_pos_w = env.scene.env_origins + torch.tensor([-0.25, 0.0, 0.4], device=env.device)
    
    # Only reward if object is actually lifted
    is_lifted = obj_pos[:, 2] > 0.08
    dist = torch.norm(home_pos_w - obj_pos, dim=-1)
    
    return is_lifted.float() * torch.exp(-3.0 * dist)

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

def ee_quat(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector orientation (quaternion) in world frame."""
    robot = env.scene[robot_cfg.name]
    return robot.data.body_quat_w[:, robot_cfg.body_ids[0], :]
