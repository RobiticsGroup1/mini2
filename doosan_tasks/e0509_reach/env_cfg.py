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


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E0509_URDF = os.path.join(_PROJECT_ROOT, "asset", "doosan_e0509", "e0509_with_gripper.urdf")


@configclass
class DoosanE0509SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = None

    # Each robot arm is mounted on its own desk/stand
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
class DoosanE0509ReachEnvCfg(DirectRLEnvCfg):
    # 기본 타임/에피소드
    decimation = 4
    episode_length_s = 10.0

    # 시뮬레이션
    sim = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    # 관측/액션 차원 (6축)
    action_dim: int = 6
    obs_dim: int = 6 * 2 + 3 + 3  # q(6) + qd(6) + ee_pos(3) + target(3)

    # Hydra가 serialize할 수 있도록 CFG에 Space를 명시
    observation_space: gym.Space = field(
        default_factory=lambda: gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )
    )
    action_space: gym.Space = field(
        default_factory=lambda: gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
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
            force_usd_conversion=True, # ✅ Force refresh to apply new colors
            collision_from_visuals=True, 
            collider_type="convex_hull",
            self_collision=True,
            joint_drive=None,
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
                stiffness=200.0,
                damping=40.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["rh_.*"],
                stiffness=200.0,
                damping=40.0,
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
                size=(0.16, 0.088, 0.024),
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.0, 0.047)),
        ),
    )

    ee_link_name: str = "link_6"
    target_min = (-0.3, -0.3, 0.2)
    target_max = (0.5, 0.3, 0.6)
    success_dist: float = 0.04
    w_dist: float = 1.0
    w_action: float = 0.01
    success_bonus: float = 5.0
    action_scale: float = 0.15
