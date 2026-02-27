from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def target_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The target position (3cm above snack top)."""
    snack = env.scene["snack"]
    target_pos = snack.data.root_pos_w[:, :3].clone()
    target_pos[:, 2] += 0.048 / 2.0 + 0.03
    return target_pos

def relative_tcp_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The relative position of TCP to target."""
    from isaaclab.utils.math import quat_apply
    
    robot = env.scene["robot"]
    
    # We need to find the index of link_6 dynamically
    # Since this is a manager-based env, we can usually find it in the data
    # For now, we assume the reward manager has already correctly mapped the body_ids
    # But to be safe in observations, we use the first body_id from the robot's data 
    # that matches our target (link_6). 
    # In Isaac Lab articulation data, body_pos_w is [num_envs, num_bodies, 3]
    
    # Let's get the index from the articulation itself
    body_names = robot.body_names
    link_6_idx = body_names.index("link_6")
    
    ee_pos = robot.data.body_pos_w[:, link_6_idx]
    ee_quat = robot.data.body_quat_w[:, link_6_idx]
    
    # TCP offset in link_6 local frame: Corrected to 10cm
    tcp_offset = torch.tensor([0.0, 0.0, 0.10], device=env.device).repeat(env.num_envs, 1)
    tcp_pos = ee_pos + quat_apply(ee_quat, tcp_offset)
    
    # Target
    goal_pos = target_pos(env)
    
    return goal_pos - tcp_pos
