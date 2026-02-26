"""Pre-configured terrain combinations for different robot types.

This module provides ready-to-use terrain configurations combining various
custom terrains for curriculum learning.

Usage:
    ```python
    from unitree_rl_lab.terrain.rough import LEGGED_ROUGH_TERRAINS_CFG
    env_cfg.scene.terrain = LEGGED_ROUGH_TERRAINS_CFG
    ```
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from .custom_mesh_terrains_cfg import (
    MeshBoulderFieldTerrainCfg,
    MeshCamberedRoadTerrainCfg,
    MeshMultipleRailsTerrainCfg,
    MeshPillarFieldTerrainCfg,
    MeshPyramidSlopedTerrainCfg,
    MeshRuttedTerrainCfg,
    MeshStairsTerrainCfg,
    MeshSteppingStonesTerrainCfg,
    MeshWashboardTerrainCfg,
    MeshWaveTerrainCfg,
)

# ==============================================================================
# Legged Robot Terrain Configurations (Human and Quadruped)
# ==============================================================================


@configclass
class LeggedRoughTerrainsCfg(TerrainGeneratorCfg):
    """Rough terrain configuration for legged robots.

    Includes a variety of challenging terrains suitable for both humanoid
    and quadruped robots. Features curriculum learning with increasing difficulty.
    """

    size = (8.0, 8.0)
    border_width = 20.0
    num_rows = 10
    num_cols = 20
    horizontal_scale = 0.1
    vertical_scale = 0.005
    slope_threshold = 0.75

    use_cache = False
    cache_dir = "./terrain_cache/legged_rough"

    curriculum = True
    difficulty_range = (0.0, 1.0)

    sub_terrains = {
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.1,
        ),
        "stepping_stones": MeshSteppingStonesTerrainCfg(
            proportion=0.15,
            stone_width_range=(0.35, 0.15),
            stone_height_range=(0.05, 0.15),
            gap_width_range=(0.1, 0.25),
            max_tilt_angle=10.0,
        ),
        "wave": MeshWaveTerrainCfg(
            proportion=0.15,
            amplitude_range=(0.02, 0.08),
            wavelength_x=1.2,
            wavelength_y=1.2,
        ),
        "rails": MeshMultipleRailsTerrainCfg(
            proportion=0.1,
            rail_height_range=(0.02, 0.08),
            rail_spacing_range=(0.35, 0.15),
            direction="transverse",
        ),
        "pyramid": MeshPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_angle_range=(8.0, 25.0),
            platform_size_ratio=0.25,
        ),
        "rutted": MeshRuttedTerrainCfg(
            proportion=0.1,
            groove_depth_range=(0.03, 0.12),
            groove_width_range=(0.1, 0.2),
            platform_width_range=(0.4, 0.2),
        ),
        "boulder": MeshBoulderFieldTerrainCfg(
            proportion=0.1,
            boulder_radius_range=(0.08, 0.2),
            boulder_height_range=(0.05, 0.15),
            density_range=(0.15, 0.35),
        ),
        "pillars": MeshPillarFieldTerrainCfg(
            proportion=0.15,
            pillar_radius_range=(0.12, 0.06),
            pillar_height_range=(0.25, 0.8),
            spacing_range=(0.8, 0.35),
            safe_zone_ratio=0.15,
        ),
    }


# ==============================================================================
# Humanoid-Specific Terrain Configurations
# ==============================================================================


@configclass
class HumanoidRoughTerrainsCfg(TerrainGeneratorCfg):
    """Rough terrain configuration optimized for humanoid robots.

    Features terrains that challenge bipedal balance and stepping.
    """

    size = (8.0, 8.0)
    border_width = 20.0
    num_rows = 10
    num_cols = 20
    horizontal_scale = 0.1
    vertical_scale = 0.005
    slope_threshold = 0.75

    use_cache = False
    cache_dir = "./terrain_cache/humanoid_rough"

    curriculum = True
    difficulty_range = (0.0, 1.0)

    sub_terrains = {
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.15,
        ),
        "stepping_stones": MeshSteppingStonesTerrainCfg(
            proportion=0.2,
            stone_width_range=(0.4, 0.2),
            stone_height_range=(0.05, 0.12),
            gap_width_range=(0.1, 0.2),
            max_tilt_angle=8.0,
        ),
        "wave": MeshWaveTerrainCfg(
            proportion=0.15,
            amplitude_range=(0.02, 0.06),
            wavelength_x=1.5,
            wavelength_y=1.5,
        ),
        "cambered": MeshCamberedRoadTerrainCfg(
            proportion=0.15,
            cross_slope_range=(5.0, 15.0),
            num_zones=3,
        ),
        "pyramid": MeshPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_angle_range=(8.0, 20.0),
            platform_size_ratio=0.3,
        ),
        "rails": MeshMultipleRailsTerrainCfg(
            proportion=0.1,
            rail_height_range=(0.02, 0.06),
            rail_spacing_range=(0.4, 0.2),
            direction="transverse",
        ),
        "washboard": MeshWashboardTerrainCfg(
            proportion=0.1,
            ridge_height_range=(0.01, 0.03),
            ridge_spacing_range=(0.15, 0.05),
            direction="diagonal",
        ),
    }


@configclass
class HumanoidRoughTerrainsWithStairsCfg(HumanoidRoughTerrainsCfg):
    """Humanoid rough terrains + stairs (楼梯).

    Keeps the original difficulty curriculum but mixes in stairs-up/down.
    Proportions sum to 1.0.
    """

    sub_terrains = {
        # Reduce flat + stepping_stones to make room for stairs (total -0.10)
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        "stepping_stones": MeshSteppingStonesTerrainCfg(
            proportion=0.15,
            stone_width_range=(0.4, 0.2),
            stone_height_range=(0.05, 0.12),
            gap_width_range=(0.1, 0.2),
            max_tilt_angle=8.0,
        ),
        "wave": MeshWaveTerrainCfg(
            proportion=0.15,
            amplitude_range=(0.02, 0.06),
            wavelength_x=1.5,
            wavelength_y=1.5,
        ),
        "cambered": MeshCamberedRoadTerrainCfg(
            proportion=0.15,
            cross_slope_range=(5.0, 15.0),
            num_zones=3,
        ),
        "pyramid": MeshPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_angle_range=(8.0, 20.0),
            platform_size_ratio=0.3,
        ),
        "rails": MeshMultipleRailsTerrainCfg(
            proportion=0.10,
            rail_height_range=(0.02, 0.06),
            rail_spacing_range=(0.4, 0.2),
            direction="transverse",
        ),
        "washboard": MeshWashboardTerrainCfg(
            proportion=0.10,
            ridge_height_range=(0.01, 0.03),
            ridge_spacing_range=(0.15, 0.05),
            direction="diagonal",
        ),
        # New: stairs up/down (total 0.10)
        "stairs_up": MeshStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.03, 0.12),
            step_depth_range=(0.4, 0.15),
            inverted=False,
        ),
        "stairs_down": MeshStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.03, 0.12),
            step_depth_range=(0.4, 0.15),
            inverted=True,
        ),
    }


# ==============================================================================
# Quadruped-Specific Terrain Configurations
# ==============================================================================


@configclass
class QuadrupedRoughTerrainsCfg(TerrainGeneratorCfg):
    """Rough terrain configuration optimized for quadruped robots.

    Features terrains that challenge four-legged locomotion patterns.
    """

    size = (8.0, 8.0)
    border_width = 20.0
    num_rows = 10
    num_cols = 20
    horizontal_scale = 0.1
    vertical_scale = 0.005
    slope_threshold = 0.75

    use_cache = False
    cache_dir = "./terrain_cache/quadruped_rough"

    curriculum = True
    difficulty_range = (0.0, 1.0)

    sub_terrains = {
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.1,
        ),
        "stepping_stones": MeshSteppingStonesTerrainCfg(
            proportion=0.15,
            stone_width_range=(0.3, 0.12),
            stone_height_range=(0.05, 0.18),
            gap_width_range=(0.15, 0.3),
            max_tilt_angle=12.0,
        ),
        "wave": MeshWaveTerrainCfg(
            proportion=0.15,
            amplitude_range=(0.03, 0.1),
            wavelength_x=1.0,
            wavelength_y=1.0,
        ),
        "boulder": MeshBoulderFieldTerrainCfg(
            proportion=0.15,
            boulder_radius_range=(0.1, 0.25),
            boulder_height_range=(0.08, 0.2),
            density_range=(0.2, 0.4),
        ),
        "pyramid": MeshPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_angle_range=(10.0, 30.0),
            platform_size_ratio=0.25,
        ),
        "rutted": MeshRuttedTerrainCfg(
            proportion=0.1,
            groove_depth_range=(0.05, 0.15),
            groove_width_range=(0.1, 0.25),
            platform_width_range=(0.35, 0.15),
        ),
        "rails": MeshMultipleRailsTerrainCfg(
            proportion=0.1,
            rail_height_range=(0.03, 0.1),
            rail_spacing_range=(0.3, 0.12),
            direction="grid",
        ),
        "pillars": MeshPillarFieldTerrainCfg(
            proportion=0.1,
            pillar_radius_range=(0.1, 0.05),
            pillar_height_range=(0.3, 1.0),
            spacing_range=(0.7, 0.3),
            safe_zone_ratio=0.2,
        ),
    }


# ==============================================================================
# Flat Terrain Configuration (for testing/debugging)
# ==============================================================================


@configclass
class FlatTerrainsCfg(TerrainGeneratorCfg):
    """Flat terrain configuration for testing and debugging."""

    size = (8.0, 8.0)
    border_width = 20.0
    num_rows = 1
    num_cols = 1
    horizontal_scale = 0.1
    vertical_scale = 0.005
    slope_threshold = 0.75

    curriculum = False

    sub_terrains = {
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0,
        ),
    }


# ==============================================================================
# Easy Terrain Configurations (for initial training)
# ==============================================================================


@configclass
class EasyTerrainsCfg(TerrainGeneratorCfg):
    """Easy terrain configuration for initial training stages."""

    size = (8.0, 8.0)
    border_width = 20.0
    num_rows = 5
    num_cols = 10
    horizontal_scale = 0.1
    vertical_scale = 0.005
    slope_threshold = 0.75

    use_cache = False
    cache_dir = "./terrain_cache/easy"

    curriculum = True
    difficulty_range = (0.0, 0.5)  # Limited difficulty

    sub_terrains = {
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.4,
        ),
        "wave": MeshWaveTerrainCfg(
            proportion=0.3,
            amplitude_range=(0.01, 0.04),
            wavelength_x=2.0,
            wavelength_y=2.0,
        ),
        "pyramid": MeshPyramidSlopedTerrainCfg(
            proportion=0.3,
            slope_angle_range=(5.0, 12.0),
            platform_size_ratio=0.4,
        ),
    }


# ==============================================================================
# IsaacLab Standard Terrain Configurations (Berkeley Humanoid Style)
# ==============================================================================


stair_width = [0.26, 0.28, 0.30, 0.32, 0.34, 0.36]

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs_1": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.2),
            step_width=stair_width[0],
            platform_width=3.0,
            border_width=0.25,
        ),
        "pyramid_stairs_2": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.2),
            step_width=stair_width[2],
            platform_width=3.0,
            border_width=0.25,
        ),
        "pyramid_stairs_3": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.2),
            step_width=stair_width[4],
            platform_width=3.0,
            border_width=0.25,
        ),
        "pyramid_stairs_inv_1": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.2),
            step_width=stair_width[1],
            platform_width=3.0,
            border_width=0.25,
        ),
        "pyramid_stairs_inv_2": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.2),
            step_width=stair_width[3],
            platform_width=3.0,
            border_width=0.25,
        ),
        "pyramid_stairs_inv_3": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.2),
            step_width=stair_width[5],
            platform_width=3.0,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.75, grid_height_range=(0.05, 0.15), platform_width=3.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.02), noise_step=0.02, border_width=0.25,
            slope_threshold=100.0
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.5), platform_width=3.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.5), platform_width=3.0, border_width=0.25
        ),
    },
)

RANDOM_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=50.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough_1": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.005, 0.005), noise_step=0.02, border_width=0.25,
        ),
        "random_rough_2": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.0075, 0.0075), noise_step=0.02, border_width=0.25,
        ),
        "random_rough_3": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.01, 0.01), noise_step=0.02, border_width=0.25,
        ),
        "random_rough_4": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.0125, 0.0125), noise_step=0.02, border_width=0.25,
        ),
        "random_rough_5": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.015, 0.015), noise_step=0.02, border_width=0.25,
        ),
    },
)


# ==============================================================================
# Terrain Importer Configurations
# ==============================================================================


@configclass
class LeggedTerrainImporterCfg(TerrainImporterCfg):
    """Terrain importer configuration for legged robots."""

    prim_path = "/World/ground"
    terrain_type = "generator"
    terrain_generator = LeggedRoughTerrainsCfg()
    max_init_terrain_level = 5
    collision_group = -1
    visual_material = None
    physics_material = None
    debug_vis = False


@configclass
class HumanoidTerrainImporterCfg(TerrainImporterCfg):
    """Terrain importer configuration for humanoid robots."""

    prim_path = "/World/ground"
    terrain_type = "generator"
    terrain_generator = HumanoidRoughTerrainsCfg()
    max_init_terrain_level = 5
    collision_group = -1
    visual_material = None
    physics_material = None
    debug_vis = False


@configclass
class QuadrupedTerrainImporterCfg(TerrainImporterCfg):
    """Terrain importer configuration for quadruped robots."""

    prim_path = "/World/ground"
    terrain_type = "generator"
    terrain_generator = QuadrupedRoughTerrainsCfg()
    max_init_terrain_level = 5
    collision_group = -1
    visual_material = None
    physics_material = None
    debug_vis = False


@configclass
class FlatTerrainImporterCfg(TerrainImporterCfg):
    """Terrain importer configuration for flat terrain."""

    prim_path = "/World/ground"
    terrain_type = "plane"
    collision_group = -1
    debug_vis = False


# ==============================================================================
# Convenience Aliases
# ==============================================================================

# Main terrain configurations
LEGGED_ROUGH_TERRAINS_CFG = LeggedRoughTerrainsCfg()
HUMANOID_ROUGH_TERRAINS_CFG = HumanoidRoughTerrainsCfg()
HUMANOID_ROUGH_TERRAINS_WITH_STAIRS_CFG = HumanoidRoughTerrainsWithStairsCfg()
QUADRUPED_ROUGH_TERRAINS_CFG = QuadrupedRoughTerrainsCfg()
FLAT_TERRAINS_CFG = FlatTerrainsCfg()
EASY_TERRAINS_CFG = EasyTerrainsCfg()

# Terrain importer configurations
LEGGED_TERRAIN_IMPORTER_CFG = LeggedTerrainImporterCfg()
HUMANOID_TERRAIN_IMPORTER_CFG = HumanoidTerrainImporterCfg()
QUADRUPED_TERRAIN_IMPORTER_CFG = QuadrupedTerrainImporterCfg()
FLAT_TERRAIN_IMPORTER_CFG = FlatTerrainImporterCfg()
