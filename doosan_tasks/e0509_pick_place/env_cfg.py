import math
import numpy as np
import os
from dataclasses import dataclass, field

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils import configclass

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UrdfFileCfg
from isaaclab.sim.spawners.lights import DomeLightCfg
from isaaclab.assets import AssetBaseCfg


E0509_URDF = "/home/user/IsaacLab/doosan_isaaclab_sb3/asset/doosan_e0509/e0509_with_gripper.urdf"


@configclass
class DoosanE0509SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = None

    # Each robot arm is mounted on its own desk/stand
    desk: AssetBaseCfg = None
    stand: AssetBaseCfg = None
    snack: RigidObjectCfg = None
    
    # Basket components (to make it open)
    basket_bottom: AssetBaseCfg = None
    basket_wall_nx: AssetBaseCfg = None
    basket_wall_px: AssetBaseCfg = None
    basket_wall_ny: AssetBaseCfg = None
    basket_wall_py: AssetBaseCfg = None

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
class DoosanE0509PickPlaceEnvCfg(DirectRLEnvCfg):
    # 기본 타임/에피소드
    decimation = 4
    episode_length_s = 15.0 # Increased for multi-stage task

    # Basket tuning parameters
    basket_size = (0.6, 0.3, 0.5)
    basket_pos = (0.0, -0.45, -0.215) 

    # 시뮬레이션
    sim = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    # 관측/액션 차원 (6축 + 1 그리퍼)
    # q(10) + qd(10) + ee_pos(3) + snack_pos(3) + basket_pos(3) + rel_pos(3) + stage(1) + timer(1) = 34
    action_dim: int = 7
    obs_dim: int = 34  

    # Hydra가 serialize할 수 있도록 CFG에 Space를 명시
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

    # 로봇 설정
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
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
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
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.4)), # ✅ Slate Blue
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.005,
                    rest_offset=0.0
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0175)),
        ),
        stand=AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Stand",
            spawn=sim_utils.CuboidCfg(
                size=(0.22, 0.18, 0.03),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)), # ✅ Dark Grey
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.005,
                    rest_offset=0.0
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.25, 0.0, 0.05)),
        ),
        snack=RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Snack",
            spawn=sim_utils.CuboidCfg(
                size=(0.08, 0.044, 0.024),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.733, 0.063)), # ffbb10
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.005,
                    rest_offset=0.0
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.10, 0.0, 0.047)),
        ),
        # Basket - Composed of 5 walls to make it "open"
        basket_bottom=AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/BasketBottom",
            spawn=sim_utils.CuboidCfg(
                size=(0.6, 0.3, 0.02),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.1)),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.45, 0.035 - 0.5 + 0.02/2)),
        ),
        basket_wall_nx=AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/BasketWallNX",
            spawn=sim_utils.CuboidCfg(
                size=(0.02, 0.3, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.1)),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.3 + 0.02/2, -0.45, 0.035 - 0.25)),
        ),
        basket_wall_px=AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/BasketWallPX",
            spawn=sim_utils.CuboidCfg(
                size=(0.02, 0.3, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.1)),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3 - 0.02/2, -0.45, 0.035 - 0.25)),
        ),
        basket_wall_ny=AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/BasketWallNY",
            spawn=sim_utils.CuboidCfg(
                size=(0.6, 0.02, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.1)),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.45 - 0.15 + 0.02/2, 0.035 - 0.25)),
        ),
        basket_wall_py=AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/BasketWallPY",
            spawn=sim_utils.CuboidCfg(
                size=(0.6, 0.02, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.1)),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.45 + 0.15 - 0.02/2, 0.035 - 0.25)),
        ),
    )

    ee_link_name: str = "link_6"
    
    # target bounds (fixed on the desk based on stand position)
    target_min = (0.10, 0.0, 0.047)
    target_max = (0.10, 0.0, 0.047)
    
    # reward scales (Perception + Pick)
    reach_reward_scale = 2.0
    grasp_reward_scale = 5.0
    lift_reward_scale = 5.0
    home_reward_scale = 5.0
    place_reward_scale = 10.0

    action_penalty_scale = 0.01
    joint_vel_penalty_scale = 0.005
    collision_penalty_scale = 2.0

    # thresholds
    reach_success_dist = 0.04
    grasp_success_dist = 0.03
    home_success_dist = 0.05
    place_success_dist = 0.05

    # action properties
    action_scale: float = 0.03
