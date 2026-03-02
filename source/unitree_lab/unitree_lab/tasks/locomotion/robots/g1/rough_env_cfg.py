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

_AMP_MOTION_DIR = str(
    Path(__file__).resolve().parents[4] / "data" / "MotionData" / "g1_29dof" / "amp" / "walk_and_run"
)


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
            lin_vel_x=(-2.0, 2.0),
            lin_vel_y=(-0.0, 0.0),
            ang_vel_z=(-2.0, 2.0),
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
    class DiscriminatorCfg(ObsGroup):
        """Agent proprioceptive features for AMP discriminator (3D output)."""
        amp_agent_obs = ObsTerm(
            func=mdp.AMPAgentObsTerm,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "disc_obs_steps": 2,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DiscriminatorDemoCfg(ObsGroup):
        """Demo motion features for AMP discriminator (3D output)."""
        amp_demo_obs = ObsTerm(
            func=mdp.AMPDemoObsTerm,
            params={
                "disc_obs_steps": 2,
                "motion_files": [
                    _AMP_MOTION_DIR + "/B4_-_Stand_to_Walk_backwards_stageii.pkl",
                    _AMP_MOTION_DIR + "/B9_-__Walk_turn_left_90_stageii.pkl",
                    _AMP_MOTION_DIR + "/B10_-__Walk_turn_left_45_stageii.pkl",
                    _AMP_MOTION_DIR + "/B11_-__Walk_turn_left_135_stageii.pkl",
                    _AMP_MOTION_DIR + "/B13_-__Walk_turn_right_90_stageii.pkl",
                    _AMP_MOTION_DIR + "/B14_-__Walk_turn_right_45_t2_stageii.pkl",
                    _AMP_MOTION_DIR + "/B15_-__Walk_turn_around_stageii.pkl",
                    _AMP_MOTION_DIR + "/B22_-__side_step_left_stageii.pkl",
                    _AMP_MOTION_DIR + "/B23_-__side_step_right_stageii.pkl",
                    _AMP_MOTION_DIR + "/Walk_B4_-_Stand_to_Walk_Back_stageii.pkl",
                    _AMP_MOTION_DIR + "/Walk_B10_-_Walk_turn_left_45_stageii.pkl",
                    _AMP_MOTION_DIR + "/Walk_B13_-_Walk_turn_right_45_stageii.pkl",
                    _AMP_MOTION_DIR + "/Walk_B15_-_Walk_turn_around_stageii.pkl",
                    _AMP_MOTION_DIR + "/Walk_B16_-_Walk_turn_change_stageii.pkl",
                    _AMP_MOTION_DIR + "/Walk_B22_-_Side_step_left_stageii.pkl",
                    _AMP_MOTION_DIR + "/Walk_B23_-_Side_step_right_stageii.pkl",
                    _AMP_MOTION_DIR + "/C1_-_stand_to_run_stageii.pkl",
                    _AMP_MOTION_DIR + "/C3_-_run_stageii.pkl",
                    _AMP_MOTION_DIR + "/C4_-_run_to_walk_a_stageii.pkl",
                    _AMP_MOTION_DIR + "/C5_-_walk_to_run_stageii.pkl",
                    _AMP_MOTION_DIR + "/C6_-_stand_to_run_backwards_stageii.pkl",
                    _AMP_MOTION_DIR + "/C8_-_run_backwards_to_stand_stageii.pkl",
                    _AMP_MOTION_DIR + "/C9_-_run_backwards_turn_run_forward_stageii.pkl",
                    _AMP_MOTION_DIR + "/C11_-_run_turn_left_90_stageii.pkl",
                    _AMP_MOTION_DIR + "/C12_-_run_turn_left_45_stageii.pkl",
                    _AMP_MOTION_DIR + "/C13_-_run_turn_left_135_stageii.pkl",
                    _AMP_MOTION_DIR + "/C14_-_run_turn_right_90_stageii.pkl",
                    _AMP_MOTION_DIR + "/C15_-_run_turn_right_45_stageii.pkl",
                    _AMP_MOTION_DIR + "/C16_-_run_turn_right_135_stageii.pkl",
                    _AMP_MOTION_DIR + "/C17_-_run_change_direction_stageii.pkl",
                ],
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
    disc_agent: DiscriminatorCfg = DiscriminatorCfg()
    disc_demo: DiscriminatorDemoCfg = DiscriminatorDemoCfg()


@configclass
class G1EventCfg:
    physics_material = EventTerm(func=mdp.randomize_rigid_body_material, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": (0.6, 1.0), "dynamic_friction_range": (0.4, 0.8), "restitution_range": (0.0, 1.0), "num_buckets": 64})
    add_base_mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "mass_distribution_params": (-5.0, 5.0), "operation": "add"})
    scale_link_mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names=["(?!.*torso.*).*"]), "mass_distribution_params": (0.8, 1.2), "operation": "scale"})
    randomize_rigid_body_com = EventTerm(func=mdp.randomize_rigid_body_com, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}})
    scale_actuator_gains = EventTerm(func=mdp.randomize_actuator_gains, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"), "stiffness_distribution_params": (0.8, 1.2), "damping_distribution_params": (0.8, 1.2), "operation": "scale"})
    scale_joint_armature = EventTerm(func=mdp.randomize_joint_parameters, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"), "armature_distribution_params": (0.8, 1.2), "operation": "scale"})
    reset_base = EventTerm(func=mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}, "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5), "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)}})
    reset_robot_joints = EventTerm(func=mdp.reset_joints_by_scale, mode="reset", params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)})
    base_external_force_torque = EventTerm(func=mdp.apply_external_force_torque_stochastic, mode="interval", interval_range_s=(0.0, 0.0), params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "force_range": {"x": (-1000.0, 1000.0), "y": (-1000.0, 1000.0), "z": (-500.0, 500.0)}, "torque_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0)}, "probability": 0.002})


@configclass
class G1RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    lin_vel_z_l2 = RewTerm(func=mdp.body_lin_vel_z_l2, weight=-0.25, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})
    ang_vel_xy_l2 = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-0.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"), "threshold": 1.0})
    fly = RewTerm(func=mdp.fly, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "threshold": 1.0})
    body_orientation_l2 = RewTerm(func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, weight=-2.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(func=mdp.feet_air_time_positive_biped, weight=0.15, params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "threshold": 0.25})
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.25, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*")})
    feet_force = RewTerm(func=mdp.body_force, weight=-3e-3, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "threshold": 500, "max_reward": 400})
    feet_too_near = RewTerm(func=mdp.feet_too_near_humanoid, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "threshold": 0.2})
    feet_stumble = RewTerm(func=mdp.feet_stumble, weight=-2.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*ankle_roll.*"])})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(func=mdp.joint_deviation_l1, weight=-0.15, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*", ".*_shoulder_pitch.*", ".*_elbow.*"])})
    joint_deviation_arms = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_wrist.*"])})
    joint_deviation_legs = RewTerm(func=mdp.joint_deviation_l1, weight=-0.02, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle.*"])})


@configclass
class G1TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0})


@configclass
class G1CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


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
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": [-100.0, 100]}
        self.observations.policy.height_scan = None
        self.rewards.feet_air_time.weight = 0.4
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -1.0
        # AMP-specific: let discriminator handle style
        self.rewards.joint_deviation_arms.weight = 0.0
        self.rewards.joint_deviation_hip.weight = 0.0
        self.rewards.joint_deviation_legs.weight = 0.0
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.body_orientation_l2.weight = -1.0
        self.rewards.fly.weight = 0.0
        self.rewards.feet_force.weight = 0.0


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
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
