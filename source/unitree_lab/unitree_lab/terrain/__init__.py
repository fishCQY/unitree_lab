"""Custom terrain module for unitree_lab.

This module provides custom terrain generation functions and configurations
for training locomotion policies on challenging terrains.

Terrain Types:
- Stepping Stones (梅花桩): Balance and precise stepping
- Cambered Road (横坡路面): Lateral stability
- Rutted Terrain (回字形沟槽): Obstacle crossing
- Multiple Rails (多横杆): Leg lifting training
- Boulder Field (乱石滩): Obstacle avoidance
- Wave Terrain (波浪地形): Continuous undulation
- Washboard (搓板路): High-frequency vibration
- Pyramid Sloped (金字塔坡): Slope climbing
- Pillar Field (柱阵): Navigation around obstacles

Usage:
    ```python
    from unitree_lab.terrain import LEGGED_ROUGH_TERRAINS_CFG
    env_cfg.scene.terrain = LEGGED_ROUGH_TERRAINS_CFG
    ```
"""

from .custom_mesh_terrains import (
    boulder_field_terrain,
    cambered_road_terrain,
    multiple_rails_terrain,
    pillar_field_terrain,
    pyramid_sloped_terrain,
    rutted_terrain,
    stairs_terrain,
    stepping_stones_terrain,
    washboard_terrain,
    wave_terrain,
)
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
from .rough import (
    # Generator configurations
    EasyTerrainsCfg,
    FlatTerrainsCfg,
    HumanoidRoughTerrainsCfg,
    LeggedRoughTerrainsCfg,
    QuadrupedRoughTerrainsCfg,
    # Importer configurations
    FlatTerrainImporterCfg,
    HumanoidTerrainImporterCfg,
    LeggedTerrainImporterCfg,
    QuadrupedTerrainImporterCfg,
    # Pre-built instances
    EASY_TERRAINS_CFG,
    FLAT_TERRAIN_IMPORTER_CFG,
    FLAT_TERRAINS_CFG,
    HUMANOID_ROUGH_TERRAINS_CFG,
    HUMANOID_ROUGH_TERRAINS_WITH_STAIRS_CFG,
    HUMANOID_TERRAIN_IMPORTER_CFG,
    LEGGED_ROUGH_TERRAINS_CFG,
    LEGGED_TERRAIN_IMPORTER_CFG,
    QUADRUPED_ROUGH_TERRAINS_CFG,
    QUADRUPED_TERRAIN_IMPORTER_CFG,
    # IsaacLab standard terrain configurations
    ROUGH_TERRAINS_CFG,
    RANDOM_TERRAINS_CFG,
)

__all__ = [
    # Terrain generation functions
    "stepping_stones_terrain",
    "cambered_road_terrain",
    "rutted_terrain",
    "multiple_rails_terrain",
    "boulder_field_terrain",
    "wave_terrain",
    "washboard_terrain",
    "pyramid_sloped_terrain",
    "stairs_terrain",
    "pillar_field_terrain",
    # Terrain configuration classes
    "MeshSteppingStonesTerrainCfg",
    "MeshCamberedRoadTerrainCfg",
    "MeshRuttedTerrainCfg",
    "MeshMultipleRailsTerrainCfg",
    "MeshBoulderFieldTerrainCfg",
    "MeshWaveTerrainCfg",
    "MeshWashboardTerrainCfg",
    "MeshPyramidSlopedTerrainCfg",
    "MeshStairsTerrainCfg",
    "MeshPillarFieldTerrainCfg",
    # Generator configurations
    "LeggedRoughTerrainsCfg",
    "HumanoidRoughTerrainsCfg",
    "QuadrupedRoughTerrainsCfg",
    "FlatTerrainsCfg",
    "EasyTerrainsCfg",
    # Importer configurations
    "LeggedTerrainImporterCfg",
    "HumanoidTerrainImporterCfg",
    "QuadrupedTerrainImporterCfg",
    "FlatTerrainImporterCfg",
    # Pre-built instances
    "LEGGED_ROUGH_TERRAINS_CFG",
    "HUMANOID_ROUGH_TERRAINS_CFG",
    "HUMANOID_ROUGH_TERRAINS_WITH_STAIRS_CFG",
    "QUADRUPED_ROUGH_TERRAINS_CFG",
    "FLAT_TERRAINS_CFG",
    "EASY_TERRAINS_CFG",
    "LEGGED_TERRAIN_IMPORTER_CFG",
    "HUMANOID_TERRAIN_IMPORTER_CFG",
    "QUADRUPED_TERRAIN_IMPORTER_CFG",
    "FLAT_TERRAIN_IMPORTER_CFG",
    # IsaacLab standard terrain configurations
    "ROUGH_TERRAINS_CFG",
    "RANDOM_TERRAINS_CFG",
]
