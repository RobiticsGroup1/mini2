from __future__ import annotations

import os
from dataclasses import dataclass, field

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import rewards
from . import observations

##
# Path to Robot URDF
##
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E0509_URDF = os.path.join(_PROJECT_ROOT, "asset", "doosan_e0509", "e0509_with_gripper.urdf")

@configclass
class DoosanE0509SceneCfg(InteractiveSceneCfg):
    """Configuration for the scene."""
    
    # Ground plane
    ground_plane = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=E0509_URDF,
            fix_base=True,
            make_instanceable=True,
            self_collision=True,
            activate_contact_sensors=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=5000.0, damping=500.0),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.25, 0.0, 0.066),
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

    # Contact Sensor
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
    )

    # Desk
    desk = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.6, 0.035),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.4)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            activate_contact_sensors=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0175)),
    )

    # Stand
    stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stand",
        spawn=sim_utils.CuboidCfg(
            size=(0.22, 0.18, 0.03),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.25, 0.0, 0.05)),
    )

    # Snack
    snack = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Snack",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.044, 0.048),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.733, 0.063)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.0, 0.059)),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

@configclass
class ObservationsCfg:
    """Configuration for observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Configuration for policy observations."""
        
        # Joint state
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # EE Pose (x,y,z, qw,qx,qy,qz)
        ee_pose = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["link_6"])},
        )
        
        # Relative TCP position
        rel_tcp_pos = ObsTerm(func=observations.relative_tcp_pos)

    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:
    """Configuration for actions."""
    
    # Arm control: Absolute joint positions
    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        scale=0.1,                # Reduced from 0.5 to make movement much slower and smoother
        use_default_offset=True,
    )
    
    # Gripper control: Absolute joint positions
    gripper_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["rh_.*"],
        scale=0.1,                # Reduced for smoother gripper control
        use_default_offset=True,
    )

@configclass
class RewardsCfg:
    """Configuration for rewards."""
    
    # 1. Reaching reward
    ee_distance = RewTerm(
        func=rewards.ee_distance_reward,
        weight=30.0,              # Increased from 10.0 for more aggressive reaching
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["link_6"])},
    )
    
    # 2. Orientation reward
    ee_orientation = RewTerm(
        func=rewards.ee_orientation_reward,
        weight=20.0,              # Increased from 10.0 for extreme vertical precision
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["link_6"])},
    )
    
    # 3. Bonus for reaching target
    reached_bonus = RewTerm(
        func=rewards.reached_target_bonus,
        weight=100.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["link_6"]), "threshold": 0.02},
    )
    
    # 4. Stay at target reward
    halt_reward = RewTerm(
        func=rewards.halt_reward,
        weight=20.0,              # Increased from 2.0 to strongly penalize shivering at target
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["link_6"])},
    )

    # 5. Gripper open reward
    gripper_open = RewTerm(
        func=rewards.gripper_open_reward,
        weight=5.0,               # Increased from 2.0
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["rh_.*"])},
    )
    
    # 6. Undesired contact penalty (includes self-collision)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces")},
    )
    
    # Action penalties - Specifically targeting shivering
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)  # Increased from -0.01 to penalize jitter
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.01)     # Increased from -0.001 to penalize fast vibration

@configclass
class TerminationsCfg:
    """Configuration for terminations."""
    
    # Episode timeout
    time_out = TermTerm(func=mdp.time_out, time_out=True)
    
    # Terminate if gripper closes too much (enforce stay open)
    gripper_closed = TermTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "bounds": (-0.1, 0.5), # 0.0 is open, 1.1 is closed. Let's allow some movement.
            "asset_cfg": SceneEntityCfg("robot", joint_names=["rh_.*"]),
        },
    )

    # Terminate if robot hits the floor/desk/snack
    # We use a threshold of 1.0 Newton
    illegal_contact = TermTerm(
        func=mdp.illegal_contact,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces")},
    )

@configclass
class EventsCfg:
    """Configuration for events."""
    
    # Reset robot joints to initial position
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class DoosanE0509PickEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the environment."""
    
    # Scene settings
    scene: DoosanE0509SceneCfg = DoosanE0509SceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Manager settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # Episode settings
    decimation = 2
    episode_length_s = 10.0

    # Simulation settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=decimation,
        device="cuda:0",
    )
