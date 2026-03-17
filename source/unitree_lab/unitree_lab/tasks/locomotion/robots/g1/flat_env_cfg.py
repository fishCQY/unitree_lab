"""G1 Flat Terrain Locomotion Environment Configuration.

Hierarchy:
    UnitreeG1RoughEnvCfg (rough_env_cfg.py)
    └── UnitreeG1FlatEnvCfg (this file)
        └── UnitreeG1FlatEnvCfg_PLAY
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_lab.tasks.locomotion import mdp
from unitree_lab.terrain import RANDOM_TERRAINS_CFG
from .rough_env_cfg import UnitreeG1RoughEnvCfg


@configclass
class UnitreeG1FlatEnvCfg(UnitreeG1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time.weight = 0.4
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -1.0
        # AMP-specific
        self.rewards.joint_deviation_arms.weight = 0.0
        self.rewards.joint_deviation_hip.weight = 0.0
        self.rewards.joint_deviation_legs.weight = 0.0
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.body_orientation_l2.weight = -1.0
        self.rewards.fly.weight = 0.0
        self.rewards.feet_force.weight = 0.0
        # Flat terrain
        self.scene.terrain.terrain_generator = RANDOM_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 10
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # No terrain curriculum; use velocity command curriculum instead
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = CurrTerm(
            func=mdp.command_levels_vel,
            params={
                "delta": [0.1, 0.1, 0.1],
                "max_curriculum": [(-1, 2.0), (-0.5, 0.5), (-math.pi, math.pi)],
            },
        )


class UnitreeG1FlatEnvCfg_PLAY(UnitreeG1FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
