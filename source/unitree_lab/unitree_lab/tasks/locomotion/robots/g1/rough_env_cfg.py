"""G1 Rough Terrain Locomotion Environment Configuration.

Hierarchy:
    LocomotionEnvCfg (config/envs/base_env_cfg.py)
    └── UnitreeG1RoughEnvCfg (this file)
        └── UnitreeG1RoughEnvCfg_PLAY
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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

from unitree_lab.tasks.locomotion.config.envs.base_env_cfg import LocomotionEnvCfg
from unitree_lab.tasks.locomotion import mdp
from unitree_lab.assets.robots.unitree import UNITREE_G1_CFG
from unitree_lab.terrain import ROUGH_TERRAINS_CFG
from unitree_lab.sensors.ray_caster import NoiseRayCasterCameraCfg
from unitree_lab.sensors.imu import DelayedImuCfg

# AMP demonstration data directory (LAFAN locomotion subset retargeted to G1 29dof).
_AMP_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "AMP"

# G1 29dof left-right symmetric joint pairs for mirror augmentation.
_G1_LEFT_JOINTS = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
_G1_RIGHT_JOINTS = [
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


# =============================================================================
# G1-specific Scene
# =============================================================================


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """Scene configuration for G1 on rough terrain."""

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
    depth_camera = NoiseRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=NoiseRayCasterCameraCfg.OffsetCfg(
            pos=(0.0576235, 0.01753, 0.42987),
            rot=(0.253, -0.660, 0.660, -0.253),
            convention="ros",
        ),
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=1.898,
            width=32,
            height=24,
        ),
        noise_cfg=NoiseRayCasterCameraCfg.DepthSensorNoiseCfg(
            enable=True,
            gaussian_mean=0.0,
            gaussian_std=0.03,
            dropout_prob=0.1,
            range_min=0.2,
            range_max=3.0,
            invalid_value='max',
            latency_steps=2,
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        data_types=["distance_to_image_plane"],
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
# G1-specific MDP configs
# =============================================================================


@configclass
class G1CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=2.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class G1ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.imu_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.imu_projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100, 100))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-100, 100))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-100, 100))
        actions = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.imu_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.imu_projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-1.0, 1.0),
        )
        joint_torques = ObsTerm(func=mdp.joint_torques)
        joint_accs = ObsTerm(func=mdp.joint_accs)
        feet_lin_vel = ObsTerm(func=mdp.feet_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")})
        feet_contact_force = ObsTerm(func=mdp.feet_contact_force, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")})
        base_mass_rel = ObsTerm(func=mdp.rigid_body_masses, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})
        rigid_body_material = ObsTerm(func=mdp.rigid_body_material, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")})
        base_com = ObsTerm(func=mdp.base_com, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})
        action_delay_legs = ObsTerm(func=mdp.action_delay, params={"actuators_names": "legs"})
        action_delay_feet = ObsTerm(func=mdp.action_delay, params={"actuators_names": "feet"})
        action_delay_shoulders = ObsTerm(func=mdp.action_delay, params={"actuators_names": "shoulders"})
        action_delay_arms = ObsTerm(func=mdp.action_delay, params={"actuators_names": "arms"})
        action_delay_wrist = ObsTerm(func=mdp.action_delay, params={"actuators_names": "wrist"})
        push_force = ObsTerm(func=mdp.push_force, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})
        push_torque = ObsTerm(func=mdp.push_torque, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})
        contact_information = ObsTerm(
            func=mdp.contact_information,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
                'pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
                'left_knee_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
                'right_knee_link', 'waist_yaw_link', 'waist_roll_link', 'torso_link',
                'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link',
                'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
                'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link',
                'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link',
            ])},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DebugCfg(ObsGroup):
        height_scan = ObsTerm(func=mdp.depth_scan, params={"sensor_cfg": SceneEntityCfg("depth_camera")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ImageCfg(ObsGroup):
        depth_image = ObsTerm(
            func=mdp.depth_scan,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "depth_range": (0.2, 3.0), "normalize_range": (0.0, 1.0)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AmpCfg(ObsGroup):
        """Single-step AMP features for AMPPlugin (2D output).

        Used with the plugin-based AMP runner. Each term outputs a
        standard (num_envs, dim) tensor; the AMPPlugin constructs
        multi-frame sequences internally from the rollout storage.
        """
        joint_pos = ObsTerm(func=mdp.amp_joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.amp_joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp.amp_base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        projected_gravity = ObsTerm(func=mdp.amp_projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        body_pos_b = ObsTerm(
            func=mdp.amp_body_pos_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "left_knee_link",
                        "right_knee_link",
                        "left_shoulder_roll_link",
                        "right_shoulder_roll_link",
                    ],
                ),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AmpConditionCfg(ObsGroup):
        """Condition ID for conditional AMP (integer label)."""
        vel_cmd_condition = ObsTerm(
            func=mdp.vel_cmd_condition_id,
            params={
                "command_name": "base_velocity",
                "vx_index": 0,
                "vx_threshold": 1.1,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    guidance: PolicyCfg = PolicyCfg()
    debug: DebugCfg = DebugCfg()
    image: ImageCfg = ImageCfg()
    amp: AmpCfg = AmpCfg()
    amp_condition: AmpConditionCfg = AmpConditionCfg()


@configclass
class G1EventCfg:
    physics_material = EventTerm(func=mdp.randomize_rigid_body_material, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": (0.2, 1.3), "dynamic_friction_range": (0.2, 1.3), "restitution_range": (0.0, 0.8), "num_buckets": 64})
    add_base_mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "mass_distribution_params": (1.0, 1.2), "operation": "scale"})
    scale_link_mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names=["(?!.*torso.*).*"]), "mass_distribution_params": (0.85, 1.15), "operation": "scale"})
    randomize_rigid_body_com = EventTerm(func=mdp.randomize_rigid_body_com, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "com_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.025, 0.025)}})
    scale_actuator_gains = EventTerm(func=mdp.randomize_actuator_gains, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"), "stiffness_distribution_params": (0.8, 1.2), "damping_distribution_params": (0.8, 1.2), "operation": "scale"})
    scale_joint_armature = EventTerm(func=mdp.randomize_joint_parameters, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"), "armature_distribution_params": (0.75, 1.25), "operation": "scale"})
    reset_base = EventTerm(func=mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}, "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5), "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)}})
    reset_robot_joints = EventTerm(func=mdp.reset_joints_by_scale, mode="reset", params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)})
    base_external_force_torque = EventTerm(func=mdp.apply_external_force_torque_stochastic, mode="interval", interval_range_s=(3.7, 4.2), params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "force_range": {"x": (-200.0, 200.0), "y": (-200.0, 200.0), "z": (-100.0, 100.0)}, "torque_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}, "probability": 1.0})


@configclass
class G1RewardsCfg:
    # --- Core tracking rewards (aligned with bfm_training) ---
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.5, params={"command_name": "base_velocity", "std": math.sqrt(0.15)})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.5)})
    # --- Core penalties (aligned with bfm_training) ---
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-2.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"), "threshold": 1.0})
    # --- Regularization rewards (cooperate with AMP style reward) ---
    feet_air_time = RewTerm(func=mdp.feet_air_time_positive_biped, weight=0.3, params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "threshold": 0.25})
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.1, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*")})
    joint_deviation_hip = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*", ".*_shoulder_pitch.*", ".*_elbow.*"])})
    joint_deviation_arms = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_wrist.*"])})
    joint_deviation_legs = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle.*"])})
    # --- Retained but disabled ---
    lin_vel_z_l2 = RewTerm(func=mdp.body_lin_vel_z_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})
    ang_vel_xy_l2 = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})
    energy = RewTerm(func=mdp.energy, weight=0.0)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0.0)
    fly = RewTerm(func=mdp.fly, weight=0.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "threshold": 1.0})
    body_orientation_l2 = RewTerm(func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, weight=0.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=0.0)
    feet_force = RewTerm(func=mdp.body_force, weight=0.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "threshold": 500, "max_reward": 400})
    feet_too_near = RewTerm(func=mdp.feet_too_near_humanoid, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "threshold": 0.2})
    feet_stumble = RewTerm(func=mdp.feet_stumble, weight=0.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*ankle_roll.*"])})


@configclass
class G1TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0})


@configclass
class G1CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    command_levels = CurrTerm(
        func=mdp.command_levels_vel,
        params={
            "delta": [0.1, 0.05, 0.3],
            "max_curriculum": [(-2.0, 2.0), (-0.5, 0.5), (-2.0, 2.0)],
        },
    )


# =============================================================================
# Environment configuration
# =============================================================================


@configclass
class UnitreeG1RoughEnvCfg(LocomotionEnvCfg):
    """G1 rough terrain locomotion with AMP."""

    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: G1ObservationsCfg = G1ObservationsCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    events: G1EventCfg = G1EventCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.episode_length_s = 10.0
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": [-100.0, 100]}
        self.observations.policy.height_scan = None
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.observations.debug = None
        self.observations.image = None

    def load_amp_data(self):
        """Load conditional LAFAN AMP data for the AMPPlugin discriminator.

        Returns walk/run condition labels matching the AmpConditionCfg
        (vx_threshold=1.1 → condition 0=walk, 1=run).
        """
        from unitree_lab.utils.amp_data_loader import (
            load_conditional_amp_data,
            create_mirror_config,
        )

        all_joint_names = [
            "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
            "left_knee_joint", "right_hip_yaw_joint", "right_hip_roll_joint",
            "right_hip_pitch_joint", "right_knee_joint",
            "waist_yaw_joint", "waist_roll_joint", "torso_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]
        mirror_indices, mirror_signs = create_mirror_config(
            _G1_LEFT_JOINTS, _G1_RIGHT_JOINTS, all_joint_names,
        )

        amp_conditions = {
            "walk": [str(_AMP_DATA_DIR / "lafan_walk_clips.pkl")],
            "run": [str(_AMP_DATA_DIR / "lafan_run_clips.pkl")],
        }
        return load_conditional_amp_data(
            amp_conditions,
            keys=["dof_pos", "dof_vel", "root_angle_vel", "proj_grav"],
            device=self.sim.device,
            mirror=True,
            joint_mirror_indices=mirror_indices,
            joint_mirror_signs=mirror_signs,
        )


@configclass
class UnitreeG1RoughEnvCfg_PLAY(UnitreeG1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        if self.scene.depth_camera is not None:
            self.scene.depth_camera.update_period = self.decimation * self.sim.dt
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
