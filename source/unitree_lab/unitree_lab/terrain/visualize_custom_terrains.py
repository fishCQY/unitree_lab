"""Standalone terrain visualization script.

This script allows visualization and export of custom terrains without
requiring Isaac Lab to be installed.

Usage:
    # List available terrains
    python visualize_custom_terrains.py --list

    # Visualize a terrain with specific difficulty
    python visualize_custom_terrains.py -t wave -d 0.5 --show

    # Export terrain to STL
    python visualize_custom_terrains.py -t wave --stl output.stl

    # Export all terrains with preview images
    python visualize_custom_terrains.py --all -o previews/

Requirements:
    pip install numpy scipy trimesh
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import trimesh
except ImportError:
    print("Error: trimesh is required. Install with: pip install trimesh")
    exit(1)


# ==============================================================================
# Standalone terrain configuration (without Isaac Lab dependencies)
# ==============================================================================


@dataclass
class TerrainConfig:
    """Base terrain configuration."""

    size: tuple[float, float] = (8.0, 8.0)
    edge_flatten_width: float = 0.5


@dataclass
class SteppingStonesConfig(TerrainConfig):
    """Stepping stones terrain configuration."""

    stone_width_range: tuple[float, float] = (0.3, 0.1)
    stone_height_range: tuple[float, float] = (0.05, 0.2)
    holes_depth: float = -0.5
    gap_width_range: tuple[float, float] = (0.1, 0.3)
    max_tilt_angle: float = 15.0


@dataclass
class CamberedRoadConfig(TerrainConfig):
    """Cambered road terrain configuration."""

    cross_slope_range: tuple[float, float] = (5.0, 20.0)
    num_zones: int = 3


@dataclass
class RuttedConfig(TerrainConfig):
    """Rutted terrain configuration."""

    groove_depth_range: tuple[float, float] = (0.05, 0.2)
    groove_width_range: tuple[float, float] = (0.1, 0.3)
    platform_width_range: tuple[float, float] = (0.5, 0.2)


@dataclass
class RailsConfig(TerrainConfig):
    """Multiple rails terrain configuration."""

    rail_height_range: tuple[float, float] = (0.02, 0.1)
    rail_width: float = 0.05
    rail_spacing_range: tuple[float, float] = (0.4, 0.15)
    direction: str = "transverse"


@dataclass
class BoulderFieldConfig(TerrainConfig):
    """Boulder field terrain configuration."""

    boulder_radius_range: tuple[float, float] = (0.1, 0.3)
    boulder_height_range: tuple[float, float] = (0.05, 0.2)
    density_range: tuple[float, float] = (0.1, 0.4)
    seed: int | None = 42


@dataclass
class WaveConfig(TerrainConfig):
    """Wave terrain configuration."""

    amplitude_range: tuple[float, float] = (0.02, 0.1)
    wavelength_x: float = 1.0
    wavelength_y: float = 1.0
    phase_offset: float = 0.0


@dataclass
class WashboardConfig(TerrainConfig):
    """Washboard terrain configuration."""

    ridge_height_range: tuple[float, float] = (0.01, 0.05)
    ridge_spacing_range: tuple[float, float] = (0.2, 0.05)
    direction: str = "transverse"


@dataclass
class PyramidConfig(TerrainConfig):
    """Pyramid sloped terrain configuration."""

    slope_angle_range: tuple[float, float] = (10.0, 30.0)
    platform_size_ratio: float = 0.3
    inverted: bool = False


@dataclass
class PillarFieldConfig(TerrainConfig):
    """Pillar field terrain configuration."""

    pillar_radius_range: tuple[float, float] = (0.15, 0.05)
    pillar_height_range: tuple[float, float] = (0.3, 1.0)
    spacing_range: tuple[float, float] = (1.0, 0.4)
    safe_zone_ratio: float = 0.2
    seed: int | None = 42


# ==============================================================================
# Terrain generation (copied from custom_mesh_terrains.py for standalone use)
# ==============================================================================


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


def _create_box(center, size, tilt_angle=0.0, tilt_axis="x"):
    """Create a box mesh."""
    box = trimesh.creation.box(extents=size)
    box.apply_translation(center)
    if tilt_angle != 0.0:
        import math

        if tilt_axis == "x":
            rotation = trimesh.transformations.rotation_matrix(
                math.radians(tilt_angle), [1, 0, 0], center
            )
        else:
            rotation = trimesh.transformations.rotation_matrix(
                math.radians(tilt_angle), [0, 1, 0], center
            )
        box.apply_transform(rotation)
    return box


def _create_cylinder(center, radius, height):
    """Create a cylinder mesh."""
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=16)
    cyl.apply_translation((center[0], center[1], center[2] + height / 2))
    return cyl


def _apply_edge_flattening(heights, edge_width, resolution):
    """Apply edge flattening to heightmap."""
    edge_samples = int(edge_width / resolution)
    if edge_samples <= 0:
        return heights

    rows, cols = heights.shape
    weight = np.ones_like(heights)

    for i in range(edge_samples):
        w = i / edge_samples
        weight[:, i] = np.minimum(weight[:, i], w)
        weight[:, cols - 1 - i] = np.minimum(weight[:, cols - 1 - i], w)
        weight[i, :] = np.minimum(weight[i, :], w)
        weight[rows - 1 - i, :] = np.minimum(weight[rows - 1 - i, :], w)

    return heights * weight


def _heightmap_to_mesh(heights, resolution, size_x, size_y):
    """Convert heightmap to mesh."""
    rows, cols = heights.shape
    vertices = []
    for i in range(rows):
        for j in range(cols):
            vertices.append([j * resolution, i * resolution, heights[i, j]])

    vertices = np.array(vertices)
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            faces.append([idx, idx + 1, idx + cols])
            faces.append([idx + 1, idx + cols + 1, idx + cols])

    return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))


# ==============================================================================
# Terrain generation functions
# ==============================================================================


def generate_stepping_stones(difficulty: float, cfg: SteppingStonesConfig) -> trimesh.Trimesh:
    """Generate stepping stones terrain."""
    import math

    size_x, size_y = cfg.size
    meshes = []

    stone_width = _lerp(cfg.stone_width_range[0], cfg.stone_width_range[1], difficulty)
    stone_height = _lerp(cfg.stone_height_range[0], cfg.stone_height_range[1], difficulty)
    gap_width = _lerp(cfg.gap_width_range[0], cfg.gap_width_range[1], difficulty)
    max_tilt = cfg.max_tilt_angle * difficulty

    # Ground
    ground = trimesh.creation.box(extents=(size_x, size_y, 0.1))
    ground.apply_translation((size_x / 2, size_y / 2, cfg.holes_depth - 0.05))
    meshes.append(ground)

    # Stones
    spacing = stone_width + gap_width
    offset = cfg.edge_flatten_width
    x = offset + stone_width / 2
    row = 0
    np.random.seed(42)
    while x < size_x - offset:
        y = offset + stone_width / 2
        if row % 2 == 1:
            y += spacing / 2
        while y < size_y - offset:
            tilt = np.random.uniform(-max_tilt, max_tilt) if max_tilt > 0 else 0
            axis = "x" if np.random.random() > 0.5 else "y"
            stone = _create_box((x, y, stone_height / 2), (stone_width, stone_width, stone_height), tilt, axis)
            meshes.append(stone)
            y += spacing
        x += spacing
        row += 1

    return trimesh.util.concatenate(meshes)


def generate_cambered_road(difficulty: float, cfg: CamberedRoadConfig) -> trimesh.Trimesh:
    """Generate cambered road terrain."""
    import math

    size_x, size_y = cfg.size
    resolution = 0.1

    slope_angle = _lerp(cfg.cross_slope_range[0], cfg.cross_slope_range[1], difficulty)
    slope_rad = math.radians(slope_angle)

    rows = int(size_y / resolution)
    cols = int(size_x / resolution)
    heights = np.zeros((rows, cols))

    zone_width = size_y / cfg.num_zones
    for i in range(rows):
        y = i * resolution
        zone_idx = int(y / zone_width) % cfg.num_zones
        direction = 1 if zone_idx % 2 == 0 else -1
        for j in range(cols):
            y_in_zone = y - zone_idx * zone_width
            heights[i, j] = direction * math.tan(slope_rad) * (y_in_zone - zone_width / 2)

    heights = _apply_edge_flattening(heights, cfg.edge_flatten_width, resolution)
    return _heightmap_to_mesh(heights, resolution, size_x, size_y)


def generate_rutted(difficulty: float, cfg: RuttedConfig) -> trimesh.Trimesh:
    """Generate rutted terrain."""
    size_x, size_y = cfg.size
    resolution = 0.05

    groove_depth = _lerp(cfg.groove_depth_range[0], cfg.groove_depth_range[1], difficulty)
    groove_width = _lerp(cfg.groove_width_range[0], cfg.groove_width_range[1], difficulty)
    platform_width = _lerp(cfg.platform_width_range[0], cfg.platform_width_range[1], difficulty)

    rows = int(size_y / resolution)
    cols = int(size_x / resolution)
    heights = np.zeros((rows, cols))

    pattern_size = platform_width + groove_width
    for i in range(rows):
        for j in range(cols):
            x_pos = (j * resolution) % pattern_size
            y_pos = (i * resolution) % pattern_size
            if x_pos < groove_width or y_pos < groove_width:
                heights[i, j] = -groove_depth

    heights = _apply_edge_flattening(heights, cfg.edge_flatten_width, resolution)
    return _heightmap_to_mesh(heights, resolution, size_x, size_y)


def generate_rails(difficulty: float, cfg: RailsConfig) -> trimesh.Trimesh:
    """Generate multiple rails terrain."""
    size_x, size_y = cfg.size
    meshes = []

    ground = trimesh.creation.box(extents=(size_x, size_y, 0.1))
    ground.apply_translation((size_x / 2, size_y / 2, -0.05))
    meshes.append(ground)

    rail_height = _lerp(cfg.rail_height_range[0], cfg.rail_height_range[1], difficulty)
    rail_spacing = _lerp(cfg.rail_spacing_range[0], cfg.rail_spacing_range[1], difficulty)
    offset = cfg.edge_flatten_width

    if cfg.direction in ("transverse", "grid"):
        x = offset
        while x < size_x - offset:
            rail = trimesh.creation.box(extents=(cfg.rail_width, size_y - 2 * offset, rail_height))
            rail.apply_translation((x, size_y / 2, rail_height / 2))
            meshes.append(rail)
            x += rail_spacing

    if cfg.direction in ("longitudinal", "grid"):
        y = offset
        while y < size_y - offset:
            rail = trimesh.creation.box(extents=(size_x - 2 * offset, cfg.rail_width, rail_height))
            rail.apply_translation((size_x / 2, y, rail_height / 2))
            meshes.append(rail)
            y += rail_spacing

    return trimesh.util.concatenate(meshes)


def generate_boulder_field(difficulty: float, cfg: BoulderFieldConfig) -> trimesh.Trimesh:
    """Generate boulder field terrain."""
    size_x, size_y = cfg.size
    meshes = []

    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    ground = trimesh.creation.box(extents=(size_x, size_y, 0.1))
    ground.apply_translation((size_x / 2, size_y / 2, -0.05))
    meshes.append(ground)

    boulder_radius = _lerp(cfg.boulder_radius_range[0], cfg.boulder_radius_range[1], difficulty)
    boulder_height = _lerp(cfg.boulder_height_range[0], cfg.boulder_height_range[1], difficulty)
    density = _lerp(cfg.density_range[0], cfg.density_range[1], difficulty)

    area = (size_x - 2 * cfg.edge_flatten_width) * (size_y - 2 * cfg.edge_flatten_width)
    num_boulders = int(area * density)
    offset = cfg.edge_flatten_width + boulder_radius

    for _ in range(num_boulders):
        x = np.random.uniform(offset, size_x - offset)
        y = np.random.uniform(offset, size_y - offset)
        r = boulder_radius * np.random.uniform(0.7, 1.3)
        h = boulder_height * np.random.uniform(0.5, 1.5)
        boulder = _create_cylinder((x, y, 0), r, h)
        meshes.append(boulder)

    return trimesh.util.concatenate(meshes)


def generate_wave(difficulty: float, cfg: WaveConfig) -> trimesh.Trimesh:
    """Generate wave terrain."""
    import math

    size_x, size_y = cfg.size
    resolution = 0.1

    amplitude = _lerp(cfg.amplitude_range[0], cfg.amplitude_range[1], difficulty)
    kx = 2 * math.pi / cfg.wavelength_x
    ky = 2 * math.pi / cfg.wavelength_y

    rows = int(size_y / resolution)
    cols = int(size_x / resolution)
    heights = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            x = j * resolution
            y = i * resolution
            heights[i, j] = amplitude * (
                math.sin(kx * x + cfg.phase_offset) + math.cos(ky * y + cfg.phase_offset)
            )

    heights = _apply_edge_flattening(heights, cfg.edge_flatten_width, resolution)
    return _heightmap_to_mesh(heights, resolution, size_x, size_y)


def generate_washboard(difficulty: float, cfg: WashboardConfig) -> trimesh.Trimesh:
    """Generate washboard terrain."""
    import math

    size_x, size_y = cfg.size
    resolution = 0.02

    ridge_height = _lerp(cfg.ridge_height_range[0], cfg.ridge_height_range[1], difficulty)
    ridge_spacing = _lerp(cfg.ridge_spacing_range[0], cfg.ridge_spacing_range[1], difficulty)

    rows = int(size_y / resolution)
    cols = int(size_x / resolution)
    heights = np.zeros((rows, cols))
    k = 2 * math.pi / ridge_spacing

    for i in range(rows):
        for j in range(cols):
            x = j * resolution
            y = i * resolution
            if cfg.direction == "transverse":
                heights[i, j] = ridge_height * 0.5 * (1 + math.sin(k * x))
            elif cfg.direction == "longitudinal":
                heights[i, j] = ridge_height * 0.5 * (1 + math.sin(k * y))
            else:
                heights[i, j] = ridge_height * 0.5 * (1 + math.sin(k * (x + y) / math.sqrt(2)))

    heights = _apply_edge_flattening(heights, cfg.edge_flatten_width, resolution)
    return _heightmap_to_mesh(heights, resolution, size_x, size_y)


def generate_pyramid(difficulty: float, cfg: PyramidConfig) -> trimesh.Trimesh:
    """Generate pyramid sloped terrain."""
    import math

    size_x, size_y = cfg.size
    resolution = 0.1

    slope_angle = _lerp(cfg.slope_angle_range[0], cfg.slope_angle_range[1], difficulty)
    slope = math.tan(math.radians(slope_angle))

    rows = int(size_y / resolution)
    cols = int(size_x / resolution)
    heights = np.zeros((rows, cols))

    center_x = size_x / 2
    center_y = size_y / 2
    platform_radius = min(size_x, size_y) * cfg.platform_size_ratio / 2

    for i in range(rows):
        for j in range(cols):
            x = j * resolution
            y = i * resolution
            dist = max(abs(x - center_x), abs(y - center_y))
            if dist <= platform_radius:
                height = slope * (min(size_x, size_y) / 2 - platform_radius)
            else:
                height = slope * (min(size_x, size_y) / 2 - dist)
            heights[i, j] = -height if cfg.inverted else height

    heights = _apply_edge_flattening(heights, cfg.edge_flatten_width, resolution)
    return _heightmap_to_mesh(heights, resolution, size_x, size_y)


def generate_pillar_field(difficulty: float, cfg: PillarFieldConfig) -> trimesh.Trimesh:
    """Generate pillar field terrain."""
    size_x, size_y = cfg.size
    meshes = []

    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    ground = trimesh.creation.box(extents=(size_x, size_y, 0.1))
    ground.apply_translation((size_x / 2, size_y / 2, -0.05))
    meshes.append(ground)

    pillar_radius = _lerp(cfg.pillar_radius_range[0], cfg.pillar_radius_range[1], difficulty)
    pillar_height = _lerp(cfg.pillar_height_range[0], cfg.pillar_height_range[1], difficulty)
    spacing = _lerp(cfg.spacing_range[0], cfg.spacing_range[1], difficulty)

    safe_x = size_x * cfg.safe_zone_ratio / 2
    safe_y = size_y * cfg.safe_zone_ratio / 2
    center_x = size_x / 2
    center_y = size_y / 2
    offset = cfg.edge_flatten_width + pillar_radius

    x = offset
    while x < size_x - offset:
        y = offset
        while y < size_y - offset:
            if abs(x - center_x) < safe_x and abs(y - center_y) < safe_y:
                y += spacing
                continue
            rx = x + np.random.uniform(-spacing * 0.2, spacing * 0.2)
            ry = y + np.random.uniform(-spacing * 0.2, spacing * 0.2)
            rh = pillar_height * np.random.uniform(0.8, 1.2)
            pillar = _create_cylinder((rx, ry, 0), pillar_radius, rh)
            meshes.append(pillar)
            y += spacing
        x += spacing

    return trimesh.util.concatenate(meshes)


# ==============================================================================
# Terrain registry
# ==============================================================================

TERRAIN_REGISTRY: dict[str, tuple[Callable, type]] = {
    "stepping_stones": (generate_stepping_stones, SteppingStonesConfig),
    "cambered": (generate_cambered_road, CamberedRoadConfig),
    "rutted": (generate_rutted, RuttedConfig),
    "rails": (generate_rails, RailsConfig),
    "boulder": (generate_boulder_field, BoulderFieldConfig),
    "wave": (generate_wave, WaveConfig),
    "washboard": (generate_washboard, WashboardConfig),
    "pyramid": (generate_pyramid, PyramidConfig),
    "pillars": (generate_pillar_field, PillarFieldConfig),
}


# ==============================================================================
# CLI
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="Custom terrain visualization tool")
    parser.add_argument("--list", action="store_true", help="List available terrains")
    parser.add_argument("-t", "--terrain", type=str, help="Terrain type to generate")
    parser.add_argument("-d", "--difficulty", type=float, default=0.5, help="Difficulty level (0.0-1.0)")
    parser.add_argument("--show", action="store_true", help="Show 3D visualization")
    parser.add_argument("--stl", type=str, help="Export to STL file")
    parser.add_argument("--all", action="store_true", help="Generate all terrains")
    parser.add_argument("-o", "--output", type=str, default=".", help="Output directory for --all")

    args = parser.parse_args()

    if args.list:
        print("Available terrains:")
        for name in TERRAIN_REGISTRY:
            print(f"  - {name}")
        return

    if args.all:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, (gen_func, cfg_class) in TERRAIN_REGISTRY.items():
            print(f"Generating {name}...")
            cfg = cfg_class()
            mesh = gen_func(args.difficulty, cfg)
            stl_path = output_dir / f"{name}_d{args.difficulty:.1f}.stl"
            mesh.export(str(stl_path))
            print(f"  Exported to {stl_path}")
        return

    if args.terrain:
        if args.terrain not in TERRAIN_REGISTRY:
            print(f"Error: Unknown terrain '{args.terrain}'")
            print(f"Available: {list(TERRAIN_REGISTRY.keys())}")
            return

        gen_func, cfg_class = TERRAIN_REGISTRY[args.terrain]
        cfg = cfg_class()
        mesh = gen_func(args.difficulty, cfg)

        if args.stl:
            mesh.export(args.stl)
            print(f"Exported to {args.stl}")

        if args.show:
            mesh.show()

        if not args.stl and not args.show:
            print(f"Generated {args.terrain} terrain with difficulty {args.difficulty}")
            print(f"  Vertices: {len(mesh.vertices)}")
            print(f"  Faces: {len(mesh.faces)}")
            print("Use --show to visualize or --stl to export")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
