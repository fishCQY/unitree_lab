"""Custom terrain configuration classes.

This module provides configuration classes for various custom terrains:
- Stepping Stones (梅花桩)
- Cambered Road (横坡路面)
- Rutted Terrain (回字形沟槽)
- Multiple Rails (多横杆)
- Boulder Field (乱石滩)
- Wave Terrain (波浪地形)
- Washboard (搓板路)
- Pyramid Sloped (金字塔坡)
- Stairs (楼梯)
- Pillar Field (柱阵)
"""

from dataclasses import MISSING
from typing import Literal

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from . import custom_mesh_terrains


@configclass
class MeshSteppingStonesTerrainCfg(SubTerrainBaseCfg):
    """Configuration for stepping stones terrain (梅花桩).

    Top View:                    Side View:
    ┌───────────────────┐        ●     ●       ●
    │  ●   ●     ●      │      ┌─┴─┐ ┌─┴─┐   ┌─┴─┐
    │    ●   ●     ●    │      │   │ │   │   │   │
    │  ●     ●   ●      │ ─────┴───┴─┴───┴───┴───┴── holes_depth
    └───────────────────┘

    d=0: 大石头小间隙 → d=1: 小石头大间隙+倾斜
    """

    function = custom_mesh_terrains.stepping_stones_terrain

    # Stone parameters
    stone_width_range: tuple[float, float] = (0.3, 0.1)  # (easy, hard) - smaller is harder
    stone_height_range: tuple[float, float] = (0.05, 0.2)
    holes_depth: float = -0.5

    # Gap parameters
    gap_width_range: tuple[float, float] = (0.1, 0.3)  # (easy, hard) - larger is harder

    # Tilt parameters (for harder difficulty)
    max_tilt_angle: float = 15.0  # degrees

    # Edge flattening
    edge_flatten_width: float = 0.5


@configclass
class MeshCamberedRoadTerrainCfg(SubTerrainBaseCfg):
    """Configuration for cambered road terrain (横坡路面).

    Cross-Section:
              ╱
            ╱  cross_slope
          ╱
    ════════════
    edges flattened

    d=0: 缓坡 → d=1: 陡坡
    """

    function = custom_mesh_terrains.cambered_road_terrain

    # Slope parameters
    cross_slope_range: tuple[float, float] = (5.0, 20.0)  # degrees (easy, hard)

    # Zone configuration
    num_zones: int = 3  # Number of alternating slope zones

    # Edge flattening
    edge_flatten_width: float = 0.5


@configclass
class MeshRuttedTerrainCfg(SubTerrainBaseCfg):
    """Configuration for rutted terrain (回字形沟槽).

    Top View:                    Cross-Section:
    ┌────┬─┬────┬─┬────┐        platform  platform
    │████│ │████│ │████│       ┌──────┐  ┌──────┐
    ├────┼─┼────┼─┼────┤       │      ├──┤      │
    │████│ │████│ │████│       └──────┘  └──────┘
    └────┴─┴────┴─┴────┘           groove

    d=0: 浅窄沟大平台 → d=1: 深宽沟小平台
    """

    function = custom_mesh_terrains.rutted_terrain

    # Groove parameters
    groove_depth_range: tuple[float, float] = (0.05, 0.2)  # (easy, hard)
    groove_width_range: tuple[float, float] = (0.1, 0.3)  # (easy, hard)

    # Platform parameters
    platform_width_range: tuple[float, float] = (0.5, 0.2)  # (easy, hard) - smaller is harder

    # Edge flattening
    edge_flatten_width: float = 0.5


@configclass
class MeshMultipleRailsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for multiple rails terrain (多横杆).

    Side View:
          ▄▄▄  ▄▄▄  ▄▄▄
    ─────────────────────
           ↑     ↑
        height  spacing

    d=0: 低杆宽间距 → d=1: 高杆窄间距
    """

    function = custom_mesh_terrains.multiple_rails_terrain

    # Rail parameters
    rail_height_range: tuple[float, float] = (0.02, 0.1)  # (easy, hard)
    rail_width: float = 0.05
    rail_spacing_range: tuple[float, float] = (0.4, 0.15)  # (easy, hard) - smaller is harder

    # Direction
    direction: Literal["transverse", "longitudinal", "grid"] = "transverse"

    # Edge flattening
    edge_flatten_width: float = 0.5


@configclass
class MeshBoulderFieldTerrainCfg(SubTerrainBaseCfg):
    """Configuration for boulder field terrain (乱石滩).

    Top View:                    Side View:
    ┌───────────────────┐         ◎      ○    ◎
    │  ◎    ○     ◎     │       ┌─┴─┐  ┌┴┐ ┌──┴──┐
    │    ○      ◎    ○  │      ─┴───┴──┴─┴─┴─────┴─
    │  ◎    ○     ◎     │       ground (flat)
    └───────────────────┘

    d=0: 稀疏小石 → d=1: 密集大石+倾斜
    """

    function = custom_mesh_terrains.boulder_field_terrain

    # Boulder parameters
    boulder_radius_range: tuple[float, float] = (0.1, 0.3)  # (easy, hard)
    boulder_height_range: tuple[float, float] = (0.05, 0.2)

    # Density
    density_range: tuple[float, float] = (0.1, 0.4)  # boulders per m² (easy, hard)

    # Random seed for reproducibility
    seed: int | None = None

    # Edge flattening
    edge_flatten_width: float = 0.5


@configclass
class MeshWaveTerrainCfg(SubTerrainBaseCfg):
    """Configuration for wave terrain (波浪地形).

    Side View:
         ∿∿∿∿∿∿∿∿∿∿
    ────────────────────
         z=0 at edges

    h(x,y) = A*(sin(kx) + cos(ky))
    d=0: 低振幅 → d=1: 高振幅
    """

    function = custom_mesh_terrains.wave_terrain

    # Wave parameters
    amplitude_range: tuple[float, float] = (0.02, 0.1)  # (easy, hard)
    wavelength_x: float = 1.0
    wavelength_y: float = 1.0

    # Phase offset
    phase_offset: float = 0.0

    # Edge flattening
    edge_flatten_width: float = 0.5


@configclass
class MeshWashboardTerrainCfg(SubTerrainBaseCfg):
    """Configuration for washboard terrain (搓板路).

    Transverse:    Side (sinusoidal):
    ═══════════       ∿∿∿∿∿∿∿∿∿
    ═══════════    ───────────
    ═══════════

    d=0: 宽间距低高度 → d=1: 窄间距高高度
    """

    function = custom_mesh_terrains.washboard_terrain

    # Washboard parameters
    ridge_height_range: tuple[float, float] = (0.01, 0.05)  # (easy, hard)
    ridge_spacing_range: tuple[float, float] = (0.2, 0.05)  # (easy, hard) - smaller is harder

    # Direction
    direction: Literal["transverse", "longitudinal", "diagonal"] = "transverse"

    # Edge flattening
    edge_flatten_width: float = 0.5


@configclass
class MeshPyramidSlopedTerrainCfg(SubTerrainBaseCfg):
    """Configuration for pyramid sloped terrain (金字塔坡).

    Cross-Section (normal):
            ╱▔▔▔▔╲ platform
          ╱      ╲
        ╱          ╲
    ════════════════

    Cross-Section (inverted):
        ╲          ╱
          ╲      ╱
            ╲__╱ (凹槽)

    d=0: 缓坡 → d=1: 陡坡
    """

    function = custom_mesh_terrains.pyramid_sloped_terrain

    # Slope parameters
    slope_angle_range: tuple[float, float] = (10.0, 30.0)  # degrees (easy, hard)

    # Platform size (top flat area)
    platform_size_ratio: float = 0.3  # Ratio of terrain size

    # Inverted (valley instead of hill)
    inverted: bool = False

    # Edge flattening
    edge_flatten_width: float = 0.5


@configclass
class MeshStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for stairs terrain (楼梯).

    d=0: 低台阶/宽台阶（更容易）
    d=1: 高台阶/窄台阶（更困难）
    """

    function = custom_mesh_terrains.stairs_terrain

    # Step geometry
    step_height_range: tuple[float, float] = (0.03, 0.12)  # (easy, hard)
    step_depth_range: tuple[float, float] = (0.4, 0.15)  # (easy, hard)

    # Stairs direction
    inverted: bool = False  # False: up, True: down

    # Edge flattening
    edge_flatten_width: float = 0.5


@configclass
class MeshPillarFieldTerrainCfg(SubTerrainBaseCfg):
    """Configuration for pillar field terrain (柱阵).

    Top View:                    Side View:
    ┌───────────────────┐         ▓     ▓      ▓
    │ ▓    ▓   ▓    ▓   │        ███  ███    ███
    │   ▓  [safe]  ▓    │        ███  ███    ███
    │ ▓    zone    ▓    │       ─────────────────
    └───────────────────┘        ground (flat)

    d=0: 大柱稀疏 → d=1: 小柱密集+更高
    """

    function = custom_mesh_terrains.pillar_field_terrain

    # Pillar parameters
    pillar_radius_range: tuple[float, float] = (0.15, 0.05)  # (easy, hard) - smaller is harder
    pillar_height_range: tuple[float, float] = (0.3, 1.0)

    # Spacing
    spacing_range: tuple[float, float] = (1.0, 0.4)  # (easy, hard) - smaller is harder

    # Safe zone in center (no pillars)
    safe_zone_ratio: float = 0.2

    # Random seed
    seed: int | None = None

    # Edge flattening
    edge_flatten_width: float = 0.5
