"""Custom mesh terrain generation functions.

This module provides terrain generation functions for various custom terrains.
Each function takes a difficulty parameter (0.0-1.0) and configuration,
returning a list of trimesh meshes and the origin point.

All terrains support:
- Difficulty-based parameter scaling
- Edge flattening for seamless tiling
- Curriculum learning integration
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

if TYPE_CHECKING:
    from . import custom_mesh_terrains_cfg


def _check_trimesh():
    """Check if trimesh is available."""
    if trimesh is None:
        raise ImportError(
            "trimesh is required for custom terrain generation. "
            "Install with: pip install trimesh"
        )


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t


def _create_box_mesh(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    tilt_angle: float = 0.0,
    tilt_axis: str = "x",
) -> "trimesh.Trimesh":
    """Create a box mesh with optional tilt."""
    _check_trimesh()
    box = trimesh.creation.box(extents=size)
    box.apply_translation(center)

    if tilt_angle != 0.0:
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


def _create_cylinder_mesh(
    center: tuple[float, float, float],
    radius: float,
    height: float,
) -> "trimesh.Trimesh":
    """Create a cylinder mesh."""
    _check_trimesh()
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=16)
    cylinder.apply_translation((center[0], center[1], center[2] + height / 2))
    return cylinder


def _apply_edge_flattening(
    heights: np.ndarray,
    edge_width: float,
    resolution: float,
) -> np.ndarray:
    """Apply edge flattening to height map."""
    edge_samples = int(edge_width / resolution)
    if edge_samples <= 0:
        return heights

    rows, cols = heights.shape

    # Create edge weight mask
    weight = np.ones_like(heights)

    # Left/right edges
    for i in range(edge_samples):
        w = i / edge_samples
        weight[:, i] = np.minimum(weight[:, i], w)
        weight[:, cols - 1 - i] = np.minimum(weight[:, cols - 1 - i], w)

    # Top/bottom edges
    for i in range(edge_samples):
        w = i / edge_samples
        weight[i, :] = np.minimum(weight[i, :], w)
        weight[rows - 1 - i, :] = np.minimum(weight[rows - 1 - i, :], w)

    return heights * weight


def stepping_stones_terrain(
    difficulty: float,
    cfg: "custom_mesh_terrains_cfg.MeshSteppingStonesTerrainCfg",
) -> tuple[list["trimesh.Trimesh"], np.ndarray]:
    """Generate stepping stones terrain (梅花桩).

    Args:
        difficulty: Difficulty level (0.0-1.0).
        cfg: Terrain configuration.

    Returns:
        Tuple of (meshes, origin).
    """
    _check_trimesh()

    size_x, size_y = cfg.size
    meshes = []

    # Calculate parameters based on difficulty
    stone_width = _lerp(cfg.stone_width_range[0], cfg.stone_width_range[1], difficulty)
    stone_height = _lerp(cfg.stone_height_range[0], cfg.stone_height_range[1], difficulty)
    gap_width = _lerp(cfg.gap_width_range[0], cfg.gap_width_range[1], difficulty)
    max_tilt = cfg.max_tilt_angle * difficulty

    # Create ground with holes
    ground = trimesh.creation.box(extents=(size_x, size_y, 0.1))
    ground.apply_translation((size_x / 2, size_y / 2, cfg.holes_depth - 0.05))
    meshes.append(ground)

    # Create stepping stones
    spacing = stone_width + gap_width
    offset_x = cfg.edge_flatten_width
    offset_y = cfg.edge_flatten_width

    x = offset_x + stone_width / 2
    row = 0
    while x < size_x - offset_x:
        y = offset_y + stone_width / 2
        if row % 2 == 1:
            y += spacing / 2  # Stagger rows

        while y < size_y - offset_y:
            # Random tilt for harder difficulties
            tilt = np.random.uniform(-max_tilt, max_tilt) if max_tilt > 0 else 0
            tilt_axis = "x" if np.random.random() > 0.5 else "y"

            stone = _create_box_mesh(
                center=(x, y, stone_height / 2),
                size=(stone_width, stone_width, stone_height),
                tilt_angle=tilt,
                tilt_axis=tilt_axis,
            )
            meshes.append(stone)
            y += spacing

        x += spacing
        row += 1

    origin = np.array([size_x / 2, size_y / 2, stone_height])
    return meshes, origin


def cambered_road_terrain(
    difficulty: float,
    cfg: "custom_mesh_terrains_cfg.MeshCamberedRoadTerrainCfg",
) -> tuple[list["trimesh.Trimesh"], np.ndarray]:
    """Generate cambered road terrain (横坡路面).

    Args:
        difficulty: Difficulty level (0.0-1.0).
        cfg: Terrain configuration.

    Returns:
        Tuple of (meshes, origin).
    """
    _check_trimesh()

    size_x, size_y = cfg.size
    resolution = 0.1

    # Calculate slope based on difficulty
    slope_angle = _lerp(cfg.cross_slope_range[0], cfg.cross_slope_range[1], difficulty)
    slope_rad = math.radians(slope_angle)

    # Create height map
    rows = int(size_y / resolution)
    cols = int(size_x / resolution)
    heights = np.zeros((rows, cols))

    zone_width = size_y / cfg.num_zones
    for i in range(rows):
        y = i * resolution
        zone_idx = int(y / zone_width) % cfg.num_zones
        direction = 1 if zone_idx % 2 == 0 else -1

        for j in range(cols):
            x = j * resolution
            # Cross slope within zone
            y_in_zone = y - zone_idx * zone_width
            heights[i, j] = direction * math.tan(slope_rad) * (y_in_zone - zone_width / 2)

    # Apply edge flattening
    heights = _apply_edge_flattening(heights, cfg.edge_flatten_width, resolution)

    # Convert to mesh
    vertices, faces = _heightmap_to_mesh(heights, resolution, size_x, size_y)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    origin = np.array([size_x / 2, size_y / 2, 0.0])
    return [mesh], origin


def rutted_terrain(
    difficulty: float,
    cfg: "custom_mesh_terrains_cfg.MeshRuttedTerrainCfg",
) -> tuple[list["trimesh.Trimesh"], np.ndarray]:
    """Generate rutted terrain (回字形沟槽).

    Args:
        difficulty: Difficulty level (0.0-1.0).
        cfg: Terrain configuration.

    Returns:
        Tuple of (meshes, origin).
    """
    _check_trimesh()

    size_x, size_y = cfg.size
    resolution = 0.05

    # Calculate parameters based on difficulty
    groove_depth = _lerp(cfg.groove_depth_range[0], cfg.groove_depth_range[1], difficulty)
    groove_width = _lerp(cfg.groove_width_range[0], cfg.groove_width_range[1], difficulty)
    platform_width = _lerp(cfg.platform_width_range[0], cfg.platform_width_range[1], difficulty)

    # Create height map
    rows = int(size_y / resolution)
    cols = int(size_x / resolution)
    heights = np.zeros((rows, cols))

    pattern_size = platform_width + groove_width
    for i in range(rows):
        for j in range(cols):
            x = j * resolution
            y = i * resolution

            # Check if in groove
            x_pos = x % pattern_size
            y_pos = y % pattern_size

            in_groove = (x_pos < groove_width) or (y_pos < groove_width)
            if in_groove:
                heights[i, j] = -groove_depth

    # Apply edge flattening
    heights = _apply_edge_flattening(heights, cfg.edge_flatten_width, resolution)

    # Convert to mesh
    vertices, faces = _heightmap_to_mesh(heights, resolution, size_x, size_y)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    origin = np.array([size_x / 2, size_y / 2, 0.0])
    return [mesh], origin


def multiple_rails_terrain(
    difficulty: float,
    cfg: "custom_mesh_terrains_cfg.MeshMultipleRailsTerrainCfg",
) -> tuple[list["trimesh.Trimesh"], np.ndarray]:
    """Generate multiple rails terrain (多横杆).

    Args:
        difficulty: Difficulty level (0.0-1.0).
        cfg: Terrain configuration.

    Returns:
        Tuple of (meshes, origin).
    """
    _check_trimesh()

    size_x, size_y = cfg.size
    meshes = []

    # Create ground
    ground = trimesh.creation.box(extents=(size_x, size_y, 0.1))
    ground.apply_translation((size_x / 2, size_y / 2, -0.05))
    meshes.append(ground)

    # Calculate parameters
    rail_height = _lerp(cfg.rail_height_range[0], cfg.rail_height_range[1], difficulty)
    rail_spacing = _lerp(cfg.rail_spacing_range[0], cfg.rail_spacing_range[1], difficulty)

    offset = cfg.edge_flatten_width

    if cfg.direction == "transverse":
        # Rails perpendicular to X axis
        x = offset
        while x < size_x - offset:
            rail = trimesh.creation.box(
                extents=(cfg.rail_width, size_y - 2 * offset, rail_height)
            )
            rail.apply_translation((x, size_y / 2, rail_height / 2))
            meshes.append(rail)
            x += rail_spacing

    elif cfg.direction == "longitudinal":
        # Rails perpendicular to Y axis
        y = offset
        while y < size_y - offset:
            rail = trimesh.creation.box(
                extents=(size_x - 2 * offset, cfg.rail_width, rail_height)
            )
            rail.apply_translation((size_x / 2, y, rail_height / 2))
            meshes.append(rail)
            y += rail_spacing

    else:  # grid
        # Both directions
        x = offset
        while x < size_x - offset:
            rail = trimesh.creation.box(
                extents=(cfg.rail_width, size_y - 2 * offset, rail_height)
            )
            rail.apply_translation((x, size_y / 2, rail_height / 2))
            meshes.append(rail)
            x += rail_spacing

        y = offset
        while y < size_y - offset:
            rail = trimesh.creation.box(
                extents=(size_x - 2 * offset, cfg.rail_width, rail_height)
            )
            rail.apply_translation((size_x / 2, y, rail_height / 2))
            meshes.append(rail)
            y += rail_spacing

    origin = np.array([size_x / 2, size_y / 2, rail_height])
    return meshes, origin


def boulder_field_terrain(
    difficulty: float,
    cfg: "custom_mesh_terrains_cfg.MeshBoulderFieldTerrainCfg",
) -> tuple[list["trimesh.Trimesh"], np.ndarray]:
    """Generate boulder field terrain (乱石滩).

    Args:
        difficulty: Difficulty level (0.0-1.0).
        cfg: Terrain configuration.

    Returns:
        Tuple of (meshes, origin).
    """
    _check_trimesh()

    size_x, size_y = cfg.size
    meshes = []

    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    # Create ground
    ground = trimesh.creation.box(extents=(size_x, size_y, 0.1))
    ground.apply_translation((size_x / 2, size_y / 2, -0.05))
    meshes.append(ground)

    # Calculate parameters
    boulder_radius = _lerp(cfg.boulder_radius_range[0], cfg.boulder_radius_range[1], difficulty)
    boulder_height = _lerp(cfg.boulder_height_range[0], cfg.boulder_height_range[1], difficulty)
    density = _lerp(cfg.density_range[0], cfg.density_range[1], difficulty)

    # Calculate number of boulders
    area = (size_x - 2 * cfg.edge_flatten_width) * (size_y - 2 * cfg.edge_flatten_width)
    num_boulders = int(area * density)

    offset = cfg.edge_flatten_width + boulder_radius
    for _ in range(num_boulders):
        x = np.random.uniform(offset, size_x - offset)
        y = np.random.uniform(offset, size_y - offset)
        r = boulder_radius * np.random.uniform(0.7, 1.3)
        h = boulder_height * np.random.uniform(0.5, 1.5)

        boulder = _create_cylinder_mesh((x, y, 0), r, h)
        meshes.append(boulder)

    origin = np.array([size_x / 2, size_y / 2, boulder_height])
    return meshes, origin


def wave_terrain(
    difficulty: float,
    cfg: "custom_mesh_terrains_cfg.MeshWaveTerrainCfg",
) -> tuple[list["trimesh.Trimesh"], np.ndarray]:
    """Generate wave terrain (波浪地形).

    h(x,y) = A*(sin(kx) + cos(ky))

    Args:
        difficulty: Difficulty level (0.0-1.0).
        cfg: Terrain configuration.

    Returns:
        Tuple of (meshes, origin).
    """
    _check_trimesh()

    size_x, size_y = cfg.size
    resolution = 0.1

    # Calculate amplitude based on difficulty
    amplitude = _lerp(cfg.amplitude_range[0], cfg.amplitude_range[1], difficulty)

    # Wave numbers
    kx = 2 * math.pi / cfg.wavelength_x
    ky = 2 * math.pi / cfg.wavelength_y

    # Create height map
    rows = int(size_y / resolution)
    cols = int(size_x / resolution)
    heights = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            x = j * resolution
            y = i * resolution
            heights[i, j] = amplitude * (
                math.sin(kx * x + cfg.phase_offset) +
                math.cos(ky * y + cfg.phase_offset)
            )

    # Apply edge flattening
    heights = _apply_edge_flattening(heights, cfg.edge_flatten_width, resolution)

    # Convert to mesh
    vertices, faces = _heightmap_to_mesh(heights, resolution, size_x, size_y)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    origin = np.array([size_x / 2, size_y / 2, amplitude])
    return [mesh], origin


def washboard_terrain(
    difficulty: float,
    cfg: "custom_mesh_terrains_cfg.MeshWashboardTerrainCfg",
) -> tuple[list["trimesh.Trimesh"], np.ndarray]:
    """Generate washboard terrain (搓板路).

    Args:
        difficulty: Difficulty level (0.0-1.0).
        cfg: Terrain configuration.

    Returns:
        Tuple of (meshes, origin).
    """
    _check_trimesh()

    size_x, size_y = cfg.size
    resolution = 0.02  # Fine resolution for washboard

    # Calculate parameters
    ridge_height = _lerp(cfg.ridge_height_range[0], cfg.ridge_height_range[1], difficulty)
    ridge_spacing = _lerp(cfg.ridge_spacing_range[0], cfg.ridge_spacing_range[1], difficulty)

    # Create height map
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
            else:  # diagonal
                heights[i, j] = ridge_height * 0.5 * (1 + math.sin(k * (x + y) / math.sqrt(2)))

    # Apply edge flattening
    heights = _apply_edge_flattening(heights, cfg.edge_flatten_width, resolution)

    # Convert to mesh
    vertices, faces = _heightmap_to_mesh(heights, resolution, size_x, size_y)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    origin = np.array([size_x / 2, size_y / 2, ridge_height])
    return [mesh], origin


def pyramid_sloped_terrain(
    difficulty: float,
    cfg: "custom_mesh_terrains_cfg.MeshPyramidSlopedTerrainCfg",
) -> tuple[list["trimesh.Trimesh"], np.ndarray]:
    """Generate pyramid sloped terrain (金字塔坡).

    Args:
        difficulty: Difficulty level (0.0-1.0).
        cfg: Terrain configuration.

    Returns:
        Tuple of (meshes, origin).
    """
    _check_trimesh()

    size_x, size_y = cfg.size
    resolution = 0.1

    # Calculate slope based on difficulty
    slope_angle = _lerp(cfg.slope_angle_range[0], cfg.slope_angle_range[1], difficulty)
    slope = math.tan(math.radians(slope_angle))

    # Create height map
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

            # Distance from center (L-infinity norm for pyramid shape)
            dist = max(abs(x - center_x), abs(y - center_y))

            if dist <= platform_radius:
                # On platform
                height = slope * (min(size_x, size_y) / 2 - platform_radius)
            else:
                # On slope
                height = slope * (min(size_x, size_y) / 2 - dist)

            if cfg.inverted:
                heights[i, j] = -height
            else:
                heights[i, j] = height

    # Apply edge flattening
    heights = _apply_edge_flattening(heights, cfg.edge_flatten_width, resolution)

    # Convert to mesh
    vertices, faces = _heightmap_to_mesh(heights, resolution, size_x, size_y)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    max_height = slope * (min(size_x, size_y) / 2 - platform_radius)
    origin = np.array([size_x / 2, size_y / 2, max_height if not cfg.inverted else 0.0])
    return [mesh], origin


def stairs_terrain(
    difficulty: float,
    cfg: "custom_mesh_terrains_cfg.MeshStairsTerrainCfg",
) -> tuple[list["trimesh.Trimesh"], np.ndarray]:
    """Generate stairs terrain (楼梯).

    We generate a simple 1D staircase along X:
    - Each step has constant height and depth (both difficulty-scaled).
    - `inverted=False` => stairs up (height increases with x)
    - `inverted=True`  => stairs down (height decreases with x)
    """
    _check_trimesh()

    size_x, size_y = cfg.size
    resolution = 0.05

    step_height = _lerp(cfg.step_height_range[0], cfg.step_height_range[1], difficulty)
    step_depth = _lerp(cfg.step_depth_range[0], cfg.step_depth_range[1], difficulty)

    # Create height map
    rows = int(size_y / resolution)
    cols = int(size_x / resolution)
    heights = np.zeros((rows, cols))

    # Avoid degenerate values
    step_depth = max(step_depth, resolution)

    for j in range(cols):
        x = j * resolution
        step_idx = int(x / step_depth)
        h = step_idx * step_height
        if cfg.inverted:
            h = -h
        heights[:, j] = h

    # Apply edge flattening
    heights = _apply_edge_flattening(heights, cfg.edge_flatten_width, resolution)

    # Convert to mesh
    vertices, faces = _heightmap_to_mesh(heights, resolution, size_x, size_y)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    max_h = float(np.max(heights))
    origin = np.array([size_x / 2, size_y / 2, max_h])
    return [mesh], origin


def pillar_field_terrain(
    difficulty: float,
    cfg: "custom_mesh_terrains_cfg.MeshPillarFieldTerrainCfg",
) -> tuple[list["trimesh.Trimesh"], np.ndarray]:
    """Generate pillar field terrain (柱阵).

    Args:
        difficulty: Difficulty level (0.0-1.0).
        cfg: Terrain configuration.

    Returns:
        Tuple of (meshes, origin).
    """
    _check_trimesh()

    size_x, size_y = cfg.size
    meshes = []

    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    # Create ground
    ground = trimesh.creation.box(extents=(size_x, size_y, 0.1))
    ground.apply_translation((size_x / 2, size_y / 2, -0.05))
    meshes.append(ground)

    # Calculate parameters
    pillar_radius = _lerp(cfg.pillar_radius_range[0], cfg.pillar_radius_range[1], difficulty)
    pillar_height = _lerp(cfg.pillar_height_range[0], cfg.pillar_height_range[1], difficulty)
    spacing = _lerp(cfg.spacing_range[0], cfg.spacing_range[1], difficulty)

    # Safe zone
    safe_x = size_x * cfg.safe_zone_ratio / 2
    safe_y = size_y * cfg.safe_zone_ratio / 2
    center_x = size_x / 2
    center_y = size_y / 2

    offset = cfg.edge_flatten_width + pillar_radius
    x = offset
    while x < size_x - offset:
        y = offset
        while y < size_y - offset:
            # Skip safe zone
            if abs(x - center_x) < safe_x and abs(y - center_y) < safe_y:
                y += spacing
                continue

            # Add random offset
            rx = x + np.random.uniform(-spacing * 0.2, spacing * 0.2)
            ry = y + np.random.uniform(-spacing * 0.2, spacing * 0.2)
            rh = pillar_height * np.random.uniform(0.8, 1.2)

            pillar = _create_cylinder_mesh((rx, ry, 0), pillar_radius, rh)
            meshes.append(pillar)

            y += spacing
        x += spacing

    origin = np.array([size_x / 2, size_y / 2, 0.0])
    return meshes, origin


def _heightmap_to_mesh(
    heights: np.ndarray,
    resolution: float,
    size_x: float,
    size_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert height map to mesh vertices and faces.

    Args:
        heights: 2D array of height values.
        resolution: Grid resolution.
        size_x: Terrain size in X.
        size_y: Terrain size in Y.

    Returns:
        Tuple of (vertices, faces).
    """
    rows, cols = heights.shape

    # Create vertices
    vertices = []
    for i in range(rows):
        for j in range(cols):
            x = j * resolution
            y = i * resolution
            z = heights[i, j]
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    # Create faces (two triangles per grid cell)
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            # First triangle
            faces.append([idx, idx + 1, idx + cols])
            # Second triangle
            faces.append([idx + 1, idx + cols + 1, idx + cols])

    faces = np.array(faces)

    return vertices, faces
