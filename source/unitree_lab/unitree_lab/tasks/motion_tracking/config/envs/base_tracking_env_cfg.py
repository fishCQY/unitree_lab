"""Base environment configuration for motion tracking tasks.

This module defines the abstract base configuration for motion-tracking
locomotion environments. Robot-specific tracking configs in robots/
inherit from TrackingEnvCfg and specialize scene, observations, rewards, etc.
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from unitree_lab.tasks.motion_tracking import mdp
from unitree_lab.terrain import ROUGH_TERRAINS_CFG
from unitree_lab.sensors.imu import DelayedImuCfg


# =============================================================================
# Base Scene
# =============================================================================


@configclass
class BaseTrackingSceneCfg(InteractiveSceneCfg):
    """Default scene for motion tracking tasks."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    imu: DelayedImuCfg = DelayedImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis", delay_range=(0.0, 0.0),
    )


# =============================================================================
# Base MDP settings
# =============================================================================


@configclass
class BaseTrackingCommandsCfg:
    """Motion tracking commands — must be overridden with robot-specific dataset_path / body_names."""

    motion_tracking = mdp.MotionTrackingCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.0, 0.0),
        dataset_path=MISSING,
        root_link_name=MISSING,
        tracking_body_names=MISSING,
        debug_vis=True,
        replay_dataset=False,
        use_world_frame=False,
    )


@configclass
class BaseTrackingActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True,
    )


@configclass
class BaseTrackingObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        motion_tracking_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion_tracking"})
        base_ang_vel = ObsTerm(func=mdp.imu_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.imu_projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        root_link_repr_6d = ObsTerm(func=mdp.imu_repr_6d, params={"add_noise": False})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class BaseTrackingEventCfg:
    physics_material = EventTerm(func=mdp.randomize_rigid_body_material, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": (0.6, 1.0), "dynamic_friction_range": (0.4, 0.8), "restitution_range": (0.0, 1.0), "num_buckets": 64})
    add_base_mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "mass_distribution_params": (-5.0, 5.0), "operation": "add"})
    randomize_rigid_body_com = EventTerm(func=mdp.randomize_rigid_body_com, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}})
    scale_joint_armature = EventTerm(func=mdp.randomize_joint_parameters, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"), "armature_distribution_params": (0.8, 1.2), "operation": "scale"})
    base_external_force_torque = EventTerm(func=mdp.apply_external_force_torque_stochastic, mode="interval", interval_range_s=(0.0, 0.0), params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "force_range": {"x": (-1000.0, 1000.0), "y": (-1000.0, 1000.0), "z": (-500.0, 500.0)}, "torque_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0)}, "probability": 0.002})
    reset_base = EventTerm(func=mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {"x": (-0.7, 0.7), "y": (-0.7, 0.7)}, "velocity_range": {}})


@configclass
class BaseTrackingRewardsCfg:
    """Base reward terms for motion tracking. Body names must be overridden per robot."""

    tracking_joint_pos = RewTerm(func=mdp.tracking_joint_pos_exp, weight=1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "command_name": "motion_tracking", "std": math.sqrt(0.1)})
    tracking_joint_vel = RewTerm(func=mdp.tracking_joint_vel_exp, weight=0.5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "command_name": "motion_tracking", "std": math.sqrt(5.0)})
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-0.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    feet_stumble = RewTerm(func=mdp.feet_stumble, weight=-2.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*ankle_roll.*"])})
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.25, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*")})


@configclass
class BaseTrackingTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_pos_err_termination = DoneTerm(func=mdp.root_pos_err_termination, params={"command_name": "motion_tracking", "threshold": 0.25, "probability": 0.005})
    root_quat_error_magnitude_termination = DoneTerm(func=mdp.root_quat_error_magnitude_termination, params={"command_name": "motion_tracking", "threshold": math.pi / 2, "probability": 0.005})


@configclass
class BaseTrackingCurriculumCfg:
    command_sampling_weights = CurrTerm(func=mdp.command_sampling_weights)


# =============================================================================
# Environment configuration
# =============================================================================


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for motion-tracking environments.

    Robot-specific tracking configs in robots/ inherit from this.
    """

    scene: BaseTrackingSceneCfg = BaseTrackingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: BaseTrackingObservationsCfg = BaseTrackingObservationsCfg()
    actions: BaseTrackingActionsCfg = BaseTrackingActionsCfg()
    commands: BaseTrackingCommandsCfg = BaseTrackingCommandsCfg()
    rewards: BaseTrackingRewardsCfg = BaseTrackingRewardsCfg()
    terminations: BaseTrackingTerminationsCfg = BaseTrackingTerminationsCfg()
    events: BaseTrackingEventCfg = BaseTrackingEventCfg()
    curriculum: BaseTrackingCurriculumCfg = BaseTrackingCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
