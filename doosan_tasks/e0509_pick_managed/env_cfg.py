import os
from dataclasses import dataclass, field

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg, JointPositionActionCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from . import rewards as custom_rewards

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E0509_URDF = os.path.join(_PROJECT_ROOT, "asset", "doosan_e0509", "e0509_with_gripper.urdf")

@configclass
class DoosanE0509SceneCfg(InteractiveSceneCfg):
    # Assets
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

    desk: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.6, 0.035),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.4)),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            physics_material_path="/World/Materials/DeskMaterial",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0175)),
    )

    stand: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stand",
        spawn=sim_utils.CuboidCfg(
            size=(0.22, 0.18, 0.03),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            physics_material_path="/World/Materials/StandMaterial",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.25, 0.0, 0.05)),
    )

    snack: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Snack",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.044, 0.048),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.733, 0.063)),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            physics_material_path="/World/Materials/SnackMaterial",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.10, 0.0, 0.059)),
    )

    # Lights
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.05, 0.05, 0.05))
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 1. Robot Proprioception (10 joints)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # 2. End-effector (EE) State
        ee_pos = ObsTerm(
            func=custom_rewards.ee_pos_rel, 
            params={"robot_cfg": SceneEntityCfg("robot", body_names="link_6")}
        )
        
        # 3. Target Position: Snack
        obj_pos = ObsTerm(
            func=custom_rewards.snack_pos_rel, 
            params={"object_cfg": SceneEntityCfg("snack")}
        )
        
        # 4. Relational Vectors
        rel_ee_to_snack = ObsTerm(
            func=custom_rewards.ee_to_snack_rel, 
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names="link_6"), 
                "object_cfg": SceneEntityCfg("snack")
            }
        )

    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:
    # Inverse Kinematics Control for Arm
    arm_action: RelativeJointPositionActionCfg = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        scale=0.015,
        use_zero_offset=True,
    )
    
    # Direct Control for Gripper
    gripper_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["rh_.*"],
        # JointPositionAction is: action = offset + scale * input_action
        # If input is [-1, 1], and we want [0, 1.1]:
        # scale = 0.55, offset = 0.55
        scale=0.55,
        offset=0.55,
        use_default_offset=False,
    )

@configclass
class RewardsCfg:
    reaching_reward = RewTerm(
        func=custom_rewards.reaching_ee_snack_l2,
        weight=10.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names="link_6"), 
            "object_cfg": SceneEntityCfg("snack")
        }
    )
    alignment_reward = RewTerm(
        func=custom_rewards.ee_alignment_reward,
        weight=2.0,
        params={"robot_cfg": SceneEntityCfg("robot", body_names="link_6")}
    )
    grasping_reward = RewTerm(
        func=custom_rewards.gripper_is_closed,
        weight=5.0,
        params={"robot_cfg": SceneEntityCfg("robot", joint_names=["rh_.*"])}
    )
    carrying_reward = RewTerm(
        func=custom_rewards.object_carrying_reward,
        weight=25.0,
        params={"object_cfg": SceneEntityCfg("snack")}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.005)

@configclass
class TerminationsCfg:
    time_out = TermTerm(func=mdp.time_out, time_out=True)
    object_dropped = TermTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("snack"), "minimum_height": 0.01}
    )

@configclass
class EventsCfg:
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        }
    )
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("snack")
        }
    )

@configclass
class DoosanE0509PickEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DoosanE0509SceneCfg = DoosanE0509SceneCfg(num_envs=64, env_spacing=2.5)
    
    # Manager settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # Time/Episode settings
    decimation = 4
    episode_length_s = 20.0 

    # Simulation settings
    sim = sim_utils.SimulationCfg(
        dt=1.0 / 120.0, 
        render_interval=decimation,
        device="cuda:0",
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_contact_count=2**24,
            gpu_max_rigid_patch_count=2**18,
            gpu_found_lost_aggregate_pairs_capacity=2**21,
            gpu_found_lost_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
            gpu_max_soft_body_contacts=2**21,
            gpu_max_particle_contacts=2**21,
            gpu_heap_capacity=2**27,
            gpu_temp_buffer_capacity=2**25,
            gpu_max_num_partitions=8,
        ),
    )
