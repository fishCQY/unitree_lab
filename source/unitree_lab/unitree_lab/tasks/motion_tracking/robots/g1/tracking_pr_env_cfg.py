"""G1 PR/AB Parallel Robot Motion Tracking Environment Configurations.

Hierarchy:
    UnitreeG1TrackingEnvCfg (tracking_env_cfg.py)
    ├── UnitreeG1TrackingPREnvCfg (this file)
    │   └── UnitreeG1TrackingPREnvCfg_PLAY
    └── UnitreeG1TrackingABEnvCfg (this file)
        └── UnitreeG1TrackingABEnvCfg_PLAY
"""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_lab.assets.robots.unitree_parallel import (
    UNITREE_G1_PR_CFG, UNITREE_G1_AB_CFG,
    G1_PR_ACTION_SCALE, G1_AB_ACTION_SCALE,
)

from .tracking_env_cfg import UnitreeG1TrackingEnvCfg


PR_JOINT_NAMES = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
    'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint',
    'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint',
    'left_knee_joint', 'right_knee_joint',
    'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
    'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
    'left_ankle_roll_joint', 'right_ankle_roll_joint',
    'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
    'left_elbow_joint', 'right_elbow_joint',
    'left_wrist_roll_joint', 'right_wrist_roll_joint',
    'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
    'left_wrist_yaw_joint', 'right_wrist_yaw_joint',
]

AB_JOINT_NAMES = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
    'left_hip_roll_joint', 'right_hip_roll_joint', 'torso_constraint_L_joint',
    'left_hip_yaw_joint', 'right_hip_yaw_joint', 'torso_constraint_R_joint',
    'left_knee_joint', 'right_knee_joint',
    'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
    'left_ankle_A_joint', 'right_ankle_A_joint',
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
    'left_ankle_B_joint', 'right_ankle_B_joint',
    'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
    'left_elbow_joint', 'right_elbow_joint',
    'left_wrist_roll_joint', 'right_wrist_roll_joint',
    'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
    'left_wrist_yaw_joint', 'right_wrist_yaw_joint',
]


@configclass
class UnitreeG1TrackingPREnvCfg(UnitreeG1TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_G1_PR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis/torso_link"
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/pelvis/.*"
        self.scene.imu.prim_path = "{ENV_REGEX_NS}/Robot/pelvis/pelvis"

        self.actions.joint_pos.joint_names = PR_JOINT_NAMES
        self.actions.joint_pos.scale = G1_PR_ACTION_SCALE

        self.observations.policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=PR_JOINT_NAMES)}
        self.observations.policy.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=PR_JOINT_NAMES)}
        self.observations.critic.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=PR_JOINT_NAMES)}
        self.observations.critic.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=PR_JOINT_NAMES)}
        self.observations.critic.joint_torques.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=PR_JOINT_NAMES)}
        self.observations.critic.joint_accs.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=PR_JOINT_NAMES)}

        self.rewards.tracking_joint_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=PR_JOINT_NAMES)
        self.rewards.tracking_joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=PR_JOINT_NAMES)
        self.rewards.energy.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=PR_JOINT_NAMES)}
        self.rewards.dof_acc_l2.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=PR_JOINT_NAMES)}
        self.rewards.dof_pos_limits.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=PR_JOINT_NAMES)}

        self.events.add_joint_default_pos = None


@configclass
class UnitreeG1TrackingPREnvCfg_PLAY(UnitreeG1TrackingPREnvCfg):
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


@configclass
class UnitreeG1TrackingABEnvCfg(UnitreeG1TrackingPREnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.joint_names = AB_JOINT_NAMES
        self.actions.joint_pos.scale = G1_AB_ACTION_SCALE
        self.observations.policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=AB_JOINT_NAMES)}
        self.observations.policy.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=AB_JOINT_NAMES)}


@configclass
class UnitreeG1TrackingABEnvCfg_PLAY(UnitreeG1TrackingABEnvCfg):
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
