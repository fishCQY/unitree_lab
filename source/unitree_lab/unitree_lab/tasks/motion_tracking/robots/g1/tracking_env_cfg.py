"""G1 Motion Tracking Environment Configuration.

Hierarchy:
    TrackingEnvCfg (config/envs/base_tracking_env_cfg.py)
    └── UnitreeG1TrackingEnvCfg (this file)
        └── UnitreeG1TrackingEnvCfg_PLAY
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_lab.tasks.motion_tracking.config.envs.base_tracking_env_cfg import (
    TrackingEnvCfg,
    BaseTrackingSceneCfg,
    BaseTrackingCommandsCfg,
    BaseTrackingObservationsCfg,
    BaseTrackingEventCfg,
    BaseTrackingRewardsCfg,
    BaseTrackingTerminationsCfg,
    BaseTrackingCurriculumCfg,
)
from unitree_lab.tasks.motion_tracking import mdp
from unitree_lab.assets.robots.unitree_beyondmimic import UNITREE_G1_CFG, G1_ACTION_SCALE
from unitree_lab.terrain import RANDOM_TERRAINS_CFG


G1_TRACKING_BODIES = [
    "pelvis", "torso_link",
    ".*_knee_link", ".*_ankle_roll_link",
    ".*_elbow_link", ".*_wrist_yaw_link",
]


# =============================================================================
# G1-specific MDP overrides
# =============================================================================


@configclass
class G1TrackingCommandsCfg(BaseTrackingCommandsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion_tracking.dataset_path = ["source/unitree_lab/unitree_lab/dataset/g1_lafan_50fps_tracking/*.npz"]
        self.motion_tracking.root_link_name = "pelvis"
        self.motion_tracking.tracking_body_names = G1_TRACKING_BODIES


@configclass
class G1TrackingObservationsCfg:

    @configclass
    class CommandCfg(ObsGroup):
        motion_tracking_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion_tracking"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class FutureCurrentNextCommandCfg(ObsGroup):
        """Future/current/next command set for FSQ/RFSQ interpolation."""

        future_commands = ObsTerm(func=mdp.future_command, params={"command_name": "motion_tracking"})
        current_command = ObsTerm(func=mdp.current_command, params={"command_name": "motion_tracking"})
        next_command = ObsTerm(func=mdp.next_command, params={"command_name": "motion_tracking"})
        interpolation_alpha = ObsTerm(func=mdp.interpolation_alpha, params={"command_name": "motion_tracking"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.imu_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.imu_projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        root_link_repr_6d = ObsTerm(func=mdp.imu_repr_6d, params={"add_noise": False})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.imu_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.imu_projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        joint_torques = ObsTerm(func=mdp.joint_torques)
        joint_accs = ObsTerm(func=mdp.joint_accs)
        root_link_repr_6d = ObsTerm(func=mdp.imu_repr_6d, params={"add_noise": False})
        key_points_pos_b = ObsTerm(func=mdp.key_points_pos_b, params={"command_name": "motion_tracking", "asset_cfg": SceneEntityCfg("robot", body_names=G1_TRACKING_BODIES)})
        body_link_lin_vel_b = ObsTerm(func=mdp.body_link_lin_vel_b, params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", ".*_knee_link", ".*_ankle_roll_link", ".*_elbow_link", ".*_wrist_yaw_link"])})
        feet_lin_vel = ObsTerm(func=mdp.feet_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")})
        feet_contact_force = ObsTerm(func=mdp.feet_contact_force, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")})
        base_mass_rel = ObsTerm(func=mdp.rigid_body_masses, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})
        rigid_body_material = ObsTerm(func=mdp.rigid_body_material, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")})
        base_com = ObsTerm(func=mdp.base_com, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})
        action_delay_left_leg = ObsTerm(func=mdp.action_delay, params={"actuators_names": "left_leg"})
        action_delay_left_foot = ObsTerm(func=mdp.action_delay, params={"actuators_names": "left_foot"})
        action_delay_right_leg = ObsTerm(func=mdp.action_delay, params={"actuators_names": "right_leg"})
        action_delay_right_foot = ObsTerm(func=mdp.action_delay, params={"actuators_names": "right_foot"})
        action_delay_waist = ObsTerm(func=mdp.action_delay, params={"actuators_names": "waist"})
        action_delay_left_shoulder = ObsTerm(func=mdp.action_delay, params={"actuators_names": "left_shoulder"})
        action_delay_left_arm = ObsTerm(func=mdp.action_delay, params={"actuators_names": "left_arm"})
        action_delay_left_wrist = ObsTerm(func=mdp.action_delay, params={"actuators_names": "left_wrist"})
        action_delay_right_shoulder = ObsTerm(func=mdp.action_delay, params={"actuators_names": "right_shoulder"})
        action_delay_right_arm = ObsTerm(func=mdp.action_delay, params={"actuators_names": "right_arm"})
        action_delay_right_wrist = ObsTerm(func=mdp.action_delay, params={"actuators_names": "right_wrist"})
        push_force = ObsTerm(func=mdp.push_force, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})
        push_torque = ObsTerm(func=mdp.push_torque, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})
        contact_information = ObsTerm(func=mdp.contact_information, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
            'pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
            'left_knee_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
            'right_knee_link', 'waist_yaw_link', 'waist_roll_link', 'torso_link',
            'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link',
            'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
            'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link',
            'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link',
        ])})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    command: CommandCfg = CommandCfg()
    fsq_command: FutureCurrentNextCommandCfg = FutureCurrentNextCommandCfg()
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1TrackingEventCfg(BaseTrackingEventCfg):
    scale_link_mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names=["(?!.*torso.*).*"]), "mass_distribution_params": (0.8, 1.2), "operation": "scale"})
    scale_actuator_gains = EventTerm(func=mdp.randomize_actuator_gains, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"), "stiffness_distribution_params": (0.8, 1.2), "damping_distribution_params": (0.8, 1.2), "operation": "scale"})
    add_joint_default_pos = EventTerm(func=mdp.randomize_joint_default_pos, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "pos_distribution_params": (-0.01, 0.01), "operation": "add"})


@configclass
class G1TrackingRewardsCfg(BaseTrackingRewardsCfg):
    tracking_body_lin_vel = RewTerm(func=mdp.tracking_body_vel_exp, weight=0.5, params={"asset_cfg": SceneEntityCfg("robot", body_names=G1_TRACKING_BODIES), "command_name": "motion_tracking", "std": math.sqrt(0.25)})
    tracking_body_ang_vel = RewTerm(func=mdp.tracking_body_ang_vel_exp, weight=0.5, params={"asset_cfg": SceneEntityCfg("robot", body_names=["pelvis"]), "command_name": "motion_tracking", "std": math.sqrt(0.5)})
    tracking_key_points_w_exp = RewTerm(func=mdp.tracking_key_points_w_exp, weight=1.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=["pelvis"]), "command_name": "motion_tracking", "std": math.sqrt(0.01)})
    tracking_key_points_exp = RewTerm(func=mdp.tracking_key_points_exp, weight=1.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", ".*_knee_link", ".*_ankle_roll_link", ".*_elbow_link", ".*_wrist_yaw_link"]), "command_name": "motion_tracking", "std": math.sqrt(0.01)})
    torso_contacts = RewTerm(func=mdp.select_undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0, "command_name": "motion_tracking", "height_threshold": 0.6})
    feet_height_l2 = RewTerm(func=mdp.tracking_body_height_l2, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll.*"]), "command_name": "motion_tracking"})


# =============================================================================
# Environment configuration
# =============================================================================


@configclass
class UnitreeG1TrackingEnvCfg(TrackingEnvCfg):
    """G1 motion tracking environment."""

    observations: G1TrackingObservationsCfg = G1TrackingObservationsCfg()
    commands: G1TrackingCommandsCfg = G1TrackingCommandsCfg()
    rewards: G1TrackingRewardsCfg = G1TrackingRewardsCfg()
    events: G1TrackingEventCfg = G1TrackingEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = RANDOM_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 10
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.actions.joint_pos.clip = {".*": [-100.0, 100]}


@configclass
class UnitreeG1TrackingEnvCfg_PLAY(UnitreeG1TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 200.0
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
