import math
import numpy as np
import os
from dataclasses import dataclass, field

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.lights import DomeLightCfg


E0509_URDF = "/home/user/IsaacLab/doosan_isaaclab_sb3/asset/doosan_e0509/e0509_with_gripper.urdf"


@configclass
class DoosanE0509SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = None
    desk: AssetBaseCfg = None
    stand: AssetBaseCfg = None
    snack: RigidObjectCfg = None

    # lights
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=DomeLightCfg(
            intensity=3000.0, 
            color=(0.05, 0.05, 0.05), # ✅ Deep Charcoal Grey Background
            visible_in_primary_ray=True 
        )
    )


@configclass
class DoosanE0509PickEnvCfg(DirectRLEnvCfg):
    # Time/Episode settings
    decimation = 4
    episode_length_s = 20.0

    # Simulation settings
    sim = sim_utils.SimulationCfg(
        dt=1.0 / 120.0, 
        render_interval=decimation,
        device="cuda:0",
        # Increase GPU buffer capacities for extreme high env counts (65k)
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_contact_count=2**24,      # Increased to 16M
            gpu_max_rigid_patch_count=2**18,        # Increased to 256k
            gpu_found_lost_aggregate_pairs_capacity=2**21,
            gpu_found_lost_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
            gpu_max_soft_body_contacts=2**21,
            gpu_max_particle_contacts=2**21,
            gpu_heap_capacity=2**27,               # Increased to 128M
            gpu_temp_buffer_capacity=2**25,        # Increased to 32M
            gpu_max_num_partitions=8,
        ),
    )

    # Observation/Action dimensions
    action_dim: int = 7
    obs_dim: int = 34  

    observation_space: gym.Space = field(
        default_factory=lambda: gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32
        )
    )
    action_space: gym.Space = field(
        default_factory=lambda: gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
    )

    state_space: gym.Space = field(
        default_factory=lambda: gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
    )

    # Global Physics Material (Defined once, used everywhere)
    snack_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.8,
        dynamic_friction=0.8,
        restitution=0.0,
    )

    # Robot configuration
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/e0509",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=E0509_URDF,
            fix_base=True,
            make_instanceable=True,
            merge_fixed_joints=True,
            force_usd_conversion=True, 
            collision_from_visuals=True, 
            collider_type="convex_hull",
            self_collision=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=5000.0,
                    damping=500.0,
                ),
            ),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4, 
                solver_velocity_iteration_count=0, 
                fix_root_link=True,
            ),
        ),
        articulation_root_prim_path="/base_link",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.25, 0.0, 0.066), 
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "joint_1": 0.0,
                "joint_2": 0.0,
                "joint_3": 1.5708,
                "joint_4": 0.0,
                "joint_5": 1.5708,
                "joint_6": 0.0,
                "rh_l1": 0.0,
                "rh_l2": 0.0,
                "rh_r1": 0.0,
                "rh_r2": 0.0,
            }
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
                stiffness=5000.0,
                damping=500.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["rh_.*"],
                stiffness=1000.0,
                damping=100.0,
            ),
        },
    )

    scene: DoosanE0509SceneCfg = DoosanE0509SceneCfg(
        num_envs=64,
        env_spacing=2.5,
        replicate_physics=True,
        lazy_sensor_update=True,
        robot=robot,
        desk=AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Desk",
            spawn=sim_utils.CuboidCfg(
                size=(1.2, 0.6, 0.035),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.4)),
                physics_material=None,
                collision_props=sim_utils.CollisionPropertiesCfg(), # Added collision
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0175)),
        ),
        stand=AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Stand",
            spawn=sim_utils.CuboidCfg(
                size=(0.22, 0.18, 0.03),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
                physics_material=None,
                collision_props=sim_utils.CollisionPropertiesCfg(), # Added collision
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.25, 0.0, 0.05)),
        ),
        snack=RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Snack",
            spawn=sim_utils.CuboidCfg(
                size=(0.08, 0.044, 0.048),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.733, 0.063)),
                physics_material=snack_material,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.10, 0.0, 0.059)),
        ),
    )

    ee_link_name: str = "link_6"
    ee_offset = (0.0, 0.0, 0.13)
    
    # Reward scales
    reach_reward_scale = 5.0
    grasp_reward_scale = 10.0
    lift_reward_scale = 20.0
    home_reward_scale = 20.0
    
    # Transition bonuses
    grasp_success_reward = 20.0
    lift_success_reward = 50.0

    action_penalty_scale = 0.01
    joint_vel_penalty_scale = 0.005

    # Thresholds
    reach_success_dist = 0.04
    home_success_dist = 0.05

    # Action properties
    action_scale: float = 0.02
