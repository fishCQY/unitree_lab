"""Terrain generator for MuJoCo matching IsaacLab terrains.

This module generates heightfield terrains that match IsaacLab's terrain types:
- Flat
- Random uniform
- Pyramid stairs (up/down)
- Pyramid sloped (up/down)
- Discrete obstacles

Key alignment:
- Horizontal/vertical scale
- Terrain size and border
- Height range and difficulty
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class TerrainType(Enum):
    """Terrain types matching IsaacLab."""
    FLAT = "flat"
    RANDOM_UNIFORM = "random_uniform"
    PYRAMID_STAIRS = "pyramid_stairs"
    PYRAMID_STAIRS_INV = "pyramid_stairs_inv"
    PYRAMID_SLOPED = "pyramid_sloped"
    PYRAMID_SLOPED_INV = "pyramid_sloped_inv"
    DISCRETE_OBSTACLES = "discrete_obstacles"
    WAVE = "wave"

    # IsaacLab humanoid rough sub-terrain keys (unitree_rl_lab/terrain/rough.py)
    STEPPING_STONES = "stepping_stones"
    CAMBERED = "cambered"
    PYRAMID = "pyramid"
    RAILS = "rails"
    WASHBOARD = "washboard"

    # A mixed/palette terrain that stitches multiple IsaacLab-style sub-terrains
    # into one heightfield (useful for sim2sim "rough" tasks).
    MIXED = "mixed"


@dataclass
class TerrainConfig:
    """Configuration for terrain generation."""
    
    terrain_type: str = "flat"
    
    # Size
    size: tuple[float, float] = (8.0, 8.0)  # meters
    border_width: float = 0.5
    
    # Resolution
    horizontal_scale: float = 0.1  # meters per heightfield sample
    vertical_scale: float = 0.005  # meters per height unit
    
    # Difficulty (0-1)
    difficulty: float = 0.5

    # RNG seed (None = nondeterministic)
    seed: int | None = None
    
    # Type-specific params
    # Random uniform
    noise_range: tuple[float, float] = (-0.1, 0.1)  # meters
    noise_step: float = 0.05
    downsampled_scale: float | None = None
    
    # Stairs
    step_height: float = 0.1  # meters
    step_width: float = 0.3  # meters
    
    # Slopes
    slope_angle: float = 0.3  # radians

    # Pyramid (IsaacLab key: "pyramid")
    platform_size_ratio: float = 0.3
    
    # Discrete obstacles
    obstacle_height_range: tuple[float, float] = (0.05, 0.15)
    obstacle_width_range: tuple[float, float] = (0.3, 0.6)
    num_obstacles: int = 20
    
    # Wave
    wave_amplitude: float = 0.1
    wave_frequency: float = 1.0

    # Stepping stones (IsaacLab key: "stepping_stones")
    stone_width_range: tuple[float, float] = (0.4, 0.2)
    stone_height_range: tuple[float, float] = (0.05, 0.12)
    gap_width_range: tuple[float, float] = (0.1, 0.2)

    # Cambered road (IsaacLab key: "cambered")
    cross_slope_range_deg: tuple[float, float] = (5.0, 15.0)
    cambered_num_zones: int = 3

    # Rails (IsaacLab key: "rails")
    rail_height_range: tuple[float, float] = (0.02, 0.06)
    rail_spacing_range: tuple[float, float] = (0.4, 0.2)
    rail_width: float = 0.06

    # Washboard (IsaacLab key: "washboard")
    ridge_height_range: tuple[float, float] = (0.01, 0.03)
    ridge_spacing_range: tuple[float, float] = (0.15, 0.05)
    washboard_direction: str = "diagonal"

    # Mixed terrain (stitched sub-terrains)
    # - Types must be valid TerrainType values (strings).
    # - Proportions must sum to 1.0 (best-effort; we normalize if not).
    # Default palette matches IsaacLab humanoid rough (keys + proportions).
    mixed_types: tuple[str, ...] = (
        "flat",
        "stepping_stones",
        "wave",
        "cambered",
        "pyramid",
        "rails",
        "washboard",
    )
    mixed_proportions: tuple[float, ...] = (0.15, 0.20, 0.15, 0.15, 0.15, 0.10, 0.10)
    # Mixed layout:
    # - "stripes": stitch along +X stripes (fast, but doesn't look like IsaacLab tiles)
    # - "grid": stitch NxM tiles with border (matches IsaacLab rough world structure much better)
    mixed_layout: str = "stripes"
    # Stripe layout: width of each stitched stripe along +X (meters). Smaller => more frequent changes.
    mixed_stripe_width: float = 1.0
    # Grid layout (IsaacLab-like)
    mixed_tile_size: float = 8.0
    mixed_num_rows: int = 10
    mixed_num_cols: int = 20
    mixed_border_width: float = 20.0
    # Guarantee a flat spawn zone around the origin so the robot doesn't start on stairs/bumps.
    mixed_spawn_flat: bool = True
    mixed_spawn_flat_halfwidth: float = 1.2  # meters (square patch centered at origin)


class MujocoTerrainGenerator:
    """Generator for MuJoCo heightfield terrains.
    
    Creates heightfield data matching IsaacLab terrain semantics.
    """
    
    def __init__(self, config: TerrainConfig | dict | None = None):
        """Initialize terrain generator.
        
        Args:
            config: Terrain configuration
        """
        if config is None:
            config = TerrainConfig()
        elif isinstance(config, dict):
            config = self._config_from_dict(config)
        
        self.config = config
        self.rng = np.random.default_rng(self.config.seed)
        
        # Compute heightfield dimensions
        self.nx = int(config.size[0] / config.horizontal_scale) + 1
        self.ny = int(config.size[1] / config.horizontal_scale) + 1
        
        # Height data
        self.heightfield: np.ndarray | None = None
    
    @staticmethod
    def _config_from_dict(d: dict) -> TerrainConfig:
        """Create config from dictionary."""
        config = TerrainConfig()
        for key, value in d.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def generate(self) -> np.ndarray:
        """Generate heightfield terrain.
        
        Returns:
            2D array of heights in meters
        """
        terrain_type = TerrainType(self.config.terrain_type)
        
        heights = self._generate_raw(terrain_type=terrain_type, rng=self.rng)
        
        # Apply difficulty scaling
        if terrain_type != TerrainType.FLAT:
            heights = heights * self.config.difficulty
        
        self.heightfield = heights
        return heights

    def _generate_raw(self, terrain_type: TerrainType, rng: np.random.Generator) -> np.ndarray:
        """Generate terrain heights in meters without the final global difficulty scaling."""
        if terrain_type == TerrainType.FLAT:
            return self._generate_flat()
        if terrain_type == TerrainType.RANDOM_UNIFORM:
            return self._generate_random_uniform(rng=rng)
        if terrain_type == TerrainType.PYRAMID_STAIRS:
            return self._generate_pyramid_stairs(inverted=False)
        if terrain_type == TerrainType.PYRAMID_STAIRS_INV:
            return self._generate_pyramid_stairs(inverted=True)
        if terrain_type == TerrainType.PYRAMID_SLOPED:
            return self._generate_pyramid_sloped(inverted=False)
        if terrain_type == TerrainType.PYRAMID_SLOPED_INV:
            return self._generate_pyramid_sloped(inverted=True)
        if terrain_type == TerrainType.DISCRETE_OBSTACLES:
            return self._generate_discrete_obstacles(rng=rng)
        if terrain_type == TerrainType.WAVE:
            return self._generate_wave()
        if terrain_type == TerrainType.STEPPING_STONES:
            return self._generate_stepping_stones(rng=rng)
        if terrain_type == TerrainType.CAMBERED:
            return self._generate_cambered()
        if terrain_type == TerrainType.PYRAMID:
            return self._generate_pyramid()
        if terrain_type == TerrainType.RAILS:
            return self._generate_rails(rng=rng)
        if terrain_type == TerrainType.WASHBOARD:
            return self._generate_washboard()
        if terrain_type == TerrainType.MIXED:
            return self._generate_mixed(rng=rng)
        return self._generate_flat()
    
    def _generate_flat(self) -> np.ndarray:
        """Generate flat terrain."""
        return np.zeros((self.ny, self.nx))
    
    def _generate_random_uniform(self, rng: np.random.Generator) -> np.ndarray:
        """Generate random uniform terrain."""
        cfg = self.config
        
        # Random noise
        noise_min, noise_max = cfg.noise_range
        heights = rng.uniform(noise_min, noise_max, (self.ny, self.nx))
        
        # Discretize to step
        if cfg.noise_step > 0:
            heights = np.round(heights / cfg.noise_step) * cfg.noise_step
        
        # Smooth with downsampling if specified
        if cfg.downsampled_scale and cfg.downsampled_scale > cfg.horizontal_scale:
            from scipy.ndimage import zoom
            factor = cfg.horizontal_scale / cfg.downsampled_scale
            heights_small = zoom(heights, factor, order=1)
            heights = zoom(heights_small, 1/factor, order=1)
            # Ensure correct size
            heights = heights[:self.ny, :self.nx]
        
        return heights
    
    def _generate_pyramid_stairs(self, inverted: bool = False) -> np.ndarray:
        """Generate pyramid stairs terrain."""
        cfg = self.config
        heights = np.zeros((self.ny, self.nx))
        
        # Center of terrain
        cx = self.nx // 2
        cy = self.ny // 2
        
        # Step size in samples
        step_samples = int(cfg.step_width / cfg.horizontal_scale)
        
        # Generate concentric rectangular rings
        max_steps = min(cx, cy) // step_samples
        
        for i in range(max_steps):
            step_height = (i + 1) * cfg.step_height
            
            # Inner boundary
            x_inner = cx - (i + 1) * step_samples
            y_inner = cy - (i + 1) * step_samples
            x_outer = cx + (i + 1) * step_samples
            y_outer = cy + (i + 1) * step_samples
            
            # Clip to bounds
            x_inner = max(0, x_inner)
            y_inner = max(0, y_inner)
            x_outer = min(self.nx, x_outer)
            y_outer = min(self.ny, y_outer)
            
            heights[y_inner:y_outer, x_inner:x_outer] = step_height
        
        if inverted:
            heights = heights.max() - heights

        # IMPORTANT:
        # MuJoCo hfields are later normalized into [base_z, base_z + size_z]. If this terrain has a much larger
        # max height than others in a mixed map, the whole map will be scaled down and will look "flat".
        # Cap the peak height to a reasonable range so mixed terrains remain visually distinct.
        max_abs = 0.30  # meters (matches our default XML hfield size_z)
        cur = float(np.max(np.abs(heights))) if heights.size else 0.0
        if cur > max_abs and cur > 1e-9:
            heights = heights * (max_abs / cur)
        
        return heights
    
    def _generate_pyramid_sloped(self, inverted: bool = False) -> np.ndarray:
        """Generate pyramid sloped terrain."""
        cfg = self.config
        
        # Create distance from center
        x = np.linspace(-cfg.size[0]/2, cfg.size[0]/2, self.nx)
        y = np.linspace(-cfg.size[1]/2, cfg.size[1]/2, self.ny)
        xx, yy = np.meshgrid(x, y)
        
        # Distance from center (L-infinity norm for pyramid shape)
        dist = np.maximum(np.abs(xx), np.abs(yy))
        
        # Convert to height using slope angle
        max_dist = max(cfg.size) / 2
        heights = (max_dist - dist) * np.tan(cfg.slope_angle)
        heights = np.maximum(heights, 0)
        
        if inverted:
            heights = heights.max() - heights
        
        return heights

    @staticmethod
    def _scale_to_max_abs(heights: np.ndarray, max_abs: float) -> np.ndarray:
        """Scale heights so that max(abs(h)) <= max_abs (best-effort)."""
        max_abs = float(max_abs)
        if max_abs <= 0:
            return heights
        cur = float(np.max(np.abs(heights))) if heights.size else 0.0
        if cur <= 1e-9 or cur <= max_abs:
            return heights
        return heights * (max_abs / cur)

    def _generate_pyramid(self) -> np.ndarray:
        """Generate IsaacLab-style pyramid slopes with a top platform (key: 'pyramid')."""
        cfg = self.config

        x = np.linspace(-cfg.size[0] / 2, cfg.size[0] / 2, self.nx)
        y = np.linspace(-cfg.size[1] / 2, cfg.size[1] / 2, self.ny)
        xx, yy = np.meshgrid(x, y)

        dist = np.maximum(np.abs(xx), np.abs(yy))
        max_dist = max(cfg.size) / 2

        platform_half = float(np.clip(cfg.platform_size_ratio, 0.0, 1.0)) * max_dist
        dist_eff = np.maximum(dist - platform_half, 0.0)
        heights = (max_dist - platform_half - dist_eff) * np.tan(cfg.slope_angle)
        heights = np.maximum(heights, 0.0)

        # Keep within a sane range so the whole mixed terrain doesn't get globally scaled down.
        heights = self._scale_to_max_abs(heights, max_abs=0.20)
        return heights
    
    def _generate_discrete_obstacles(self, rng: np.random.Generator) -> np.ndarray:
        """Generate terrain with discrete obstacles."""
        cfg = self.config
        heights = np.zeros((self.ny, self.nx))
        
        for _ in range(cfg.num_obstacles):
            # Random position
            ox = rng.integers(0, self.nx)
            oy = rng.integers(0, self.ny)
            
            # Random size
            width = rng.uniform(*cfg.obstacle_width_range)
            width_samples = int(width / cfg.horizontal_scale)
            
            # Random height
            height = rng.uniform(*cfg.obstacle_height_range)
            
            # Place obstacle
            x1 = max(0, ox - width_samples // 2)
            x2 = min(self.nx, ox + width_samples // 2)
            y1 = max(0, oy - width_samples // 2)
            y2 = min(self.ny, oy + width_samples // 2)
            
            heights[y1:y2, x1:x2] = np.maximum(heights[y1:y2, x1:x2], height)
        
        return heights

    def _generate_stepping_stones(self, rng: np.random.Generator) -> np.ndarray:
        """Generate stepping stones (key: 'stepping_stones') as a grid of elevated tiles."""
        cfg = self.config
        heights = np.zeros((self.ny, self.nx), dtype=np.float32)

        # Difficulty interpolates geometric params; final amplitude still goes through global difficulty scaling.
        d = float(np.clip(cfg.difficulty, 0.0, 1.0))
        stone_w = float(cfg.stone_width_range[0] + (cfg.stone_width_range[1] - cfg.stone_width_range[0]) * d)
        gap_w = float(cfg.gap_width_range[0] + (cfg.gap_width_range[1] - cfg.gap_width_range[0]) * d)
        h_lo = float(cfg.stone_height_range[0])
        h_hi = float(cfg.stone_height_range[1])

        cell = max(cfg.horizontal_scale, stone_w + gap_w)
        stone_samples = max(1, int(round(stone_w / cfg.horizontal_scale)))
        cell_samples = max(stone_samples + 1, int(round(cell / cfg.horizontal_scale)))

        for y0 in range(0, self.ny, cell_samples):
            for x0 in range(0, self.nx, cell_samples):
                y1 = min(self.ny, y0 + stone_samples)
                x1 = min(self.nx, x0 + stone_samples)
                if y1 <= y0 or x1 <= x0:
                    continue
                h = float(rng.uniform(h_lo, h_hi))
                heights[y0:y1, x0:x1] = np.maximum(heights[y0:y1, x0:x1], h)

        return heights

    def _generate_cambered(self) -> np.ndarray:
        """Generate cambered road (key: 'cambered') as piecewise cross-slope zones."""
        cfg = self.config
        d = float(np.clip(cfg.difficulty, 0.0, 1.0))
        deg = float(cfg.cross_slope_range_deg[0] + (cfg.cross_slope_range_deg[1] - cfg.cross_slope_range_deg[0]) * d)
        ang = float(np.deg2rad(deg))

        y = np.linspace(-cfg.size[1] / 2, cfg.size[1] / 2, self.ny)[:, None]
        x = np.linspace(-cfg.size[0] / 2, cfg.size[0] / 2, self.nx)[None, :]

        zones = max(1, int(cfg.cambered_num_zones))
        edges = np.linspace(float(x.min()), float(x.max()), zones + 1)

        heights = np.zeros((self.ny, self.nx), dtype=np.float32)
        for i in range(zones):
            x0, x1 = float(edges[i]), float(edges[i + 1])
            mask = (x >= x0) & (x < x1 if i < zones - 1 else x <= x1)
            sign = -1.0 if (i % 2 == 0) else 1.0
            heights = np.where(mask, sign * np.tan(ang) * y, heights)

        heights = self._scale_to_max_abs(heights, max_abs=0.15).astype(np.float32)
        return heights

    def _generate_rails(self, rng: np.random.Generator) -> np.ndarray:
        """Generate transverse rails (key: 'rails') as repeated ridges along +X."""
        cfg = self.config
        d = float(np.clip(cfg.difficulty, 0.0, 1.0))
        rail_h = float(cfg.rail_height_range[0] + (cfg.rail_height_range[1] - cfg.rail_height_range[0]) * d)
        spacing = float(cfg.rail_spacing_range[0] + (cfg.rail_spacing_range[1] - cfg.rail_spacing_range[0]) * d)
        width = float(max(cfg.rail_width, cfg.horizontal_scale))
        spacing = max(spacing, width * 1.5)

        x = np.linspace(-cfg.size[0] / 2, cfg.size[0] / 2, self.nx)[None, :]
        phase = float(rng.uniform(0.0, spacing))
        dist = np.mod(x - phase + spacing / 2.0, spacing) - spacing / 2.0

        sigma = width / 2.0
        ridge = np.exp(-0.5 * (dist / max(1e-6, sigma)) ** 2)
        row = (rail_h * ridge).astype(np.float32)
        heights = np.repeat(row, self.ny, axis=0)
        return heights

    def _generate_washboard(self) -> np.ndarray:
        """Generate washboard terrain (key: 'washboard') as high-frequency ridges."""
        cfg = self.config
        d = float(np.clip(cfg.difficulty, 0.0, 1.0))
        amp = float(cfg.ridge_height_range[0] + (cfg.ridge_height_range[1] - cfg.ridge_height_range[0]) * d)
        spacing = float(cfg.ridge_spacing_range[0] + (cfg.ridge_spacing_range[1] - cfg.ridge_spacing_range[0]) * d)
        spacing = max(spacing, cfg.horizontal_scale * 2.0)

        x = np.linspace(0, cfg.size[0], self.nx)
        y = np.linspace(0, cfg.size[1], self.ny)
        xx, yy = np.meshgrid(x, y)

        direction = str(cfg.washboard_direction).lower()
        if direction == "x":
            u = xx
        elif direction == "y":
            u = yy
        else:
            u = (xx + yy) * 0.70710678

        heights = amp * np.sin(2 * np.pi * u / spacing)
        return heights.astype(np.float32)

    def _generate_mixed(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a stitched 'mixed' terrain.

        Layout options:
        - stripes: stitch along +X stripes (legacy)
        - grid: IsaacLab-like tile grid + border (recommended for "looks like IsaacLab rough")
        """
        cfg = self.config
        layout = str(getattr(cfg, "mixed_layout", "stripes") or "stripes").lower()
        if layout == "grid":
            return self._generate_mixed_grid(rng=rng)

        types = tuple(cfg.mixed_types)
        probs = np.asarray(cfg.mixed_proportions, dtype=np.float64)
        if len(types) == 0:
            return self._generate_flat()

        # Fix/normalize probabilities (best-effort)
        if probs.shape[0] != len(types) or np.any(~np.isfinite(probs)) or float(np.sum(probs)) <= 0.0:
            probs = np.ones((len(types),), dtype=np.float64)
        probs = probs / float(np.sum(probs))

        stripe_w = float(cfg.mixed_stripe_width)
        if stripe_w <= 1e-6:
            stripe_w = float(cfg.size[0])  # single stripe

        stripe_samples = max(1, int(round(stripe_w / float(cfg.horizontal_scale))))
        num_stripes = int(np.ceil(self.nx / float(stripe_samples)))

        # Pre-generate full maps for each terrain type (then slice stripes).
        # Use child RNGs for per-type randomness but keep overall determinism.
        maps: dict[str, np.ndarray] = {}

        def _get_map(tt: str) -> np.ndarray:
            if tt in maps:
                return maps[tt]
            # Make per-type RNG stable relative to base RNG stream.
            child = np.random.default_rng(rng.integers(0, 2**32 - 1))
            try:
                t = TerrainType(tt)
            except Exception:
                t = TerrainType.FLAT
            m = self._generate_raw(terrain_type=t, rng=child)
            maps[tt] = m
            return m

        out = np.zeros((self.ny, self.nx), dtype=np.float32)
        choices = rng.choice(np.asarray(types, dtype=object), size=(num_stripes,), p=probs, replace=True)

        # Force the stripe covering the origin to be flat (if available), so the robot spawns on flat.
        if bool(getattr(cfg, "mixed_spawn_flat", True)) and ("flat" in types):
            try:
                center_x = int(self.nx // 2)
                center_stripe = int(center_x // stripe_samples)
                if 0 <= center_stripe < int(num_stripes):
                    choices[center_stripe] = "flat"
            except Exception:
                pass

        for i in range(num_stripes):
            x0 = i * stripe_samples
            x1 = min(self.nx, (i + 1) * stripe_samples)
            tt = str(choices[i])
            m = _get_map(tt)
            out[:, x0:x1] = m[:, x0:x1]

        # Additionally flatten a local patch around the origin to be robust to footprint size.
        if bool(getattr(cfg, "mixed_spawn_flat", True)):
            try:
                hw_m = float(getattr(cfg, "mixed_spawn_flat_halfwidth", 0.0) or 0.0)
                if hw_m > 1e-6:
                    r = int(max(1, round(hw_m / float(cfg.horizontal_scale))))
                    cx = int(self.nx // 2)
                    cy = int(self.ny // 2)
                    y0 = int(np.clip(cy - r, 0, self.ny - 1))
                    y1 = int(np.clip(cy + r + 1, 0, self.ny))
                    x0 = int(np.clip(cx - r, 0, self.nx - 1))
                    x1 = int(np.clip(cx + r + 1, 0, self.nx))
                    if y1 > y0 and x1 > x0:
                        out[y0:y1, x0:x1] = 0.0
            except Exception:
                pass

        return out.astype(np.float32)

    def _copy_config_for_tile(self, terrain_type: str, tile_size: float) -> TerrainConfig:
        """Create a TerrainConfig for a single tile, copying over known fields."""
        base = self.config
        cfg = TerrainConfig()
        for k, v in vars(base).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.terrain_type = str(terrain_type)
        cfg.size = (float(tile_size), float(tile_size))
        # Tile generation should NOT use mixed stitching inside tiles.
        cfg.mixed_layout = "stripes"
        return cfg

    def _generate_mixed_grid(self, rng: np.random.Generator) -> np.ndarray:
        """IsaacLab-like mixed terrain: border + (num_rows x num_cols) tiles."""
        cfg = self.config

        types = tuple(cfg.mixed_types)
        probs = np.asarray(cfg.mixed_proportions, dtype=np.float64)
        if len(types) == 0:
            return self._generate_flat()
        if probs.shape[0] != len(types) or np.any(~np.isfinite(probs)) or float(np.sum(probs)) <= 0.0:
            probs = np.ones((len(types),), dtype=np.float64)
        probs = probs / float(np.sum(probs))

        tile_size = float(getattr(cfg, "mixed_tile_size", 8.0) or 8.0)
        tile_step = int(max(1, round(tile_size / float(cfg.horizontal_scale))))  # samples per tile edge (without +1)

        border_w = float(getattr(cfg, "mixed_border_width", 0.0) or 0.0)
        border = int(max(0, round(border_w / float(cfg.horizontal_scale))))

        num_rows = int(max(1, getattr(cfg, "mixed_num_rows", 10)))
        num_cols = int(max(1, getattr(cfg, "mixed_num_cols", 20)))

        out = np.zeros((self.ny, self.nx), dtype=np.float32)

        # Choose tile types
        tile_choices = rng.choice(np.asarray(types, dtype=object), size=(num_rows, num_cols), p=probs, replace=True)

        # Force the tile covering the origin (center) to be flat if requested.
        if bool(getattr(cfg, "mixed_spawn_flat", True)) and ("flat" in types):
            try:
                center_y = self.ny // 2
                center_x = self.nx // 2
                # Compute which tile contains the center, under the assumed border+grid layout.
                j = int((center_x - border) // tile_step)
                i = int((center_y - border) // tile_step)
                if 0 <= i < num_rows and 0 <= j < num_cols:
                    tile_choices[i, j] = "flat"
            except Exception:
                pass

        # Paste tiles
        for i in range(num_rows):
            for j in range(num_cols):
                x0 = border + j * tile_step
                y0 = border + i * tile_step
                x1 = x0 + tile_step + 1
                y1 = y0 + tile_step + 1

                # If the requested grid doesn't fit the current heightfield, just stop pasting out-of-bounds tiles.
                if x0 >= self.nx or y0 >= self.ny:
                    continue
                x1c = int(min(self.nx, x1))
                y1c = int(min(self.ny, y1))
                if x1c <= x0 or y1c <= y0:
                    continue

                tt = str(tile_choices[i, j])
                child = np.random.default_rng(rng.integers(0, 2**32 - 1))
                tile_cfg = self._copy_config_for_tile(terrain_type=tt, tile_size=tile_size)
                tile_gen = MujocoTerrainGenerator(tile_cfg)
                tile_raw = tile_gen._generate_raw(terrain_type=TerrainType(tt), rng=child).astype(np.float32)

                # Crop tile to the region size (handles edge truncation).
                out[y0:y1c, x0:x1c] = tile_raw[: (y1c - y0), : (x1c - x0)]

        # Flatten spawn patch around the origin (footprint safety).
        if bool(getattr(cfg, "mixed_spawn_flat", True)):
            try:
                hw_m = float(getattr(cfg, "mixed_spawn_flat_halfwidth", 0.0) or 0.0)
                if hw_m > 1e-6:
                    r = int(max(1, round(hw_m / float(cfg.horizontal_scale))))
                    cx = int(self.nx // 2)
                    cy = int(self.ny // 2)
                    y0 = int(np.clip(cy - r, 0, self.ny - 1))
                    y1 = int(np.clip(cy + r + 1, 0, self.ny))
                    x0 = int(np.clip(cx - r, 0, self.nx - 1))
                    x1 = int(np.clip(cx + r + 1, 0, self.nx))
                    if y1 > y0 and x1 > x0:
                        out[y0:y1, x0:x1] = 0.0
            except Exception:
                pass

        return out.astype(np.float32)
    
    def _generate_wave(self) -> np.ndarray:
        """Generate wave terrain."""
        cfg = self.config
        
        x = np.linspace(0, cfg.size[0], self.nx)
        y = np.linspace(0, cfg.size[1], self.ny)
        xx, yy = np.meshgrid(x, y)
        
        # Sinusoidal waves
        heights = cfg.wave_amplitude * np.sin(2 * np.pi * cfg.wave_frequency * xx)
        heights += cfg.wave_amplitude * 0.5 * np.sin(2 * np.pi * cfg.wave_frequency * yy)
        
        return heights
    
    def get_mujoco_heightfield_data(self) -> np.ndarray:
        """Get heightfield data formatted for MuJoCo.
        
        MuJoCo expects heights as integers scaled by vertical_scale.
        
        Returns:
            Heightfield data for MuJoCo (float array)
        """
        if self.heightfield is None:
            self.generate()
        
        # MuJoCo uses heights directly in meters
        return self.heightfield.astype(np.float32).flatten()
    
    def get_spawn_height(self, x: float = 0, y: float = 0) -> float:
        """Get safe spawn height at position.
        
        Args:
            x: X position in meters
            y: Y position in meters
            
        Returns:
            Safe spawn height (terrain height + margin)
        """
        if self.heightfield is None:
            self.generate()
        
        # Convert to grid indices
        ix = int((x + self.config.size[0] / 2) / self.config.horizontal_scale)
        iy = int((y + self.config.size[1] / 2) / self.config.horizontal_scale)
        
        # Clamp to bounds
        ix = np.clip(ix, 0, self.nx - 1)
        iy = np.clip(iy, 0, self.ny - 1)
        
        # Get height with margin
        terrain_height = self.heightfield[iy, ix]
        return terrain_height + 0.5  # 50cm margin
