"""Evaluation task definitions for sim2sim.

This module defines evaluation tasks with:
- Terrain configurations
- Velocity command profiles
- Success criteria
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EvalTask:
    """Base evaluation task definition."""
    
    name: str
    description: str = ""
    
    # Task parameters
    max_episode_steps: int = 1000
    num_episodes: int = 10
    
    # Terrain
    terrain_type: str = "flat"
    terrain_config: dict = field(default_factory=dict)
    
    # Initial state
    initial_position: tuple[float, float, float] | None = None
    initial_orientation: tuple[float, float, float, float] | None = None
    
    # Success criteria
    survival_time_threshold: float = 10.0  # seconds
    
    def get_terrain_config(self) -> dict:
        """Get complete terrain configuration."""
        return {
            "terrain_type": self.terrain_type,
            **self.terrain_config,
        }


@dataclass
class LocomotionEvalTask(EvalTask):
    """Locomotion-specific evaluation task."""
    
    # Velocity command
    velocity_command: tuple[float, float, float] = (0.5, 0.0, 0.0)  # vx, vy, wz
    velocity_command_range: tuple[tuple[float, float], ...] | None = None
    resample_command_interval: float | None = None  # None = fixed command
    
    # Success criteria
    velocity_tracking_threshold: float = 0.3  # m/s error
    base_height_threshold: float = 0.1  # minimum height
    orientation_threshold: float = 0.8  # cos(angle) from vertical
    
    # Metrics to collect
    track_metrics: list[str] = field(default_factory=lambda: [
        "survival_rate",
        "velocity_error",
        "distance_traveled",
        "energy",
    ])


# Predefined evaluation tasks
LOCOMOTION_EVAL_TASKS = {
    "flat_forward": LocomotionEvalTask(
        name="flat_forward",
        description="Walk forward on flat ground",
        terrain_type="flat",
        velocity_command=(0.5, 0.0, 0.0),
        max_episode_steps=500,
    ),
    "flat_backward": LocomotionEvalTask(
        name="flat_backward",
        description="Walk backward on flat ground",
        terrain_type="flat",
        velocity_command=(-0.3, 0.0, 0.0),
        max_episode_steps=500,
    ),
    "flat_lateral": LocomotionEvalTask(
        name="flat_lateral",
        description="Walk sideways on flat ground",
        terrain_type="flat",
        velocity_command=(0.0, 0.3, 0.0),
        max_episode_steps=500,
    ),
    "flat_turn": LocomotionEvalTask(
        name="flat_turn",
        description="Turn in place on flat ground",
        terrain_type="flat",
        velocity_command=(0.0, 0.0, 0.5),
        max_episode_steps=500,
    ),
    "flat_fast": LocomotionEvalTask(
        name="flat_fast",
        description="Fast walking on flat ground",
        terrain_type="flat",
        velocity_command=(1.0, 0.0, 0.0),
        max_episode_steps=500,
    ),
    "rough_forward": LocomotionEvalTask(
        name="rough_forward",
        description="Walk forward on rough terrain",
        # Align with IsaacLab "rough" semantics: use the SAME sub-terrain palette + proportions
        # as unitree_lab/terrain/rough.py -> HumanoidRoughTerrainsCfg (grid layout).
        terrain_type="mixed",
        terrain_config={
            "difficulty": 0.6,
            # IsaacLab-like world layout (tiles + border)
            "mixed_layout": "grid",
            "mixed_tile_size": 8.0,
            "mixed_num_rows": 10,
            "mixed_num_cols": 20,
            "mixed_border_width": 20.0,
            "mixed_types": (
                "flat",
                "stepping_stones",
                "wave",
                "cambered",
                "pyramid",
                "rails",
                "washboard",
            ),
            # EXACT match to IsaacLab HumanoidRoughTerrainsCfg (no stairs)
            "mixed_proportions": (0.15, 0.20, 0.15, 0.15, 0.15, 0.10, 0.10),
            # Guarantee the robot starts on a flat patch at the origin.
            "mixed_spawn_flat": True,
            "mixed_spawn_flat_halfwidth": 1.2,
            # Keep deterministic-by-default so "rough_forward" is comparable run-to-run.
            "seed": 0,
            # Best-effort param alignment with IsaacLab ranges (used by heightfield approximations)
            "stone_width_range": (0.4, 0.2),
            "stone_height_range": (0.05, 0.12),
            "gap_width_range": (0.1, 0.2),
            "wave_amplitude": 0.06,
            "wave_frequency": 1.0 / 1.5,
            "cross_slope_range_deg": (5.0, 15.0),
            "cambered_num_zones": 3,
            "platform_size_ratio": 0.3,
            "rail_height_range": (0.02, 0.06),
            "rail_spacing_range": (0.4, 0.2),
            "washboard_direction": "diagonal",
            "ridge_height_range": (0.01, 0.03),
            "ridge_spacing_range": (0.15, 0.05),
        },
        velocity_command=(0.4, 0.0, 0.0),
        max_episode_steps=500,
    ),
    "stairs_up": LocomotionEvalTask(
        name="stairs_up",
        description="Climb stairs",
        terrain_type="pyramid_stairs",
        terrain_config={
            "step_height": 0.1,
            "step_width": 0.3,
            "difficulty": 0.7,
        },
        velocity_command=(0.3, 0.0, 0.0),
        max_episode_steps=600,
    ),
    "stairs_down": LocomotionEvalTask(
        name="stairs_down",
        description="Descend stairs",
        terrain_type="pyramid_stairs_inv",
        terrain_config={
            "step_height": 0.1,
            "step_width": 0.3,
            "difficulty": 0.7,
        },
        velocity_command=(0.3, 0.0, 0.0),
        initial_position=(0.0, 0.0, 1.5),  # Start higher
        max_episode_steps=600,
    ),
    "slope_up": LocomotionEvalTask(
        name="slope_up",
        description="Walk up slope",
        terrain_type="pyramid_sloped",
        terrain_config={
            "slope_angle": 0.2,
            "difficulty": 0.6,
        },
        velocity_command=(0.3, 0.0, 0.0),
        max_episode_steps=600,
    ),
    "mixed_terrain": LocomotionEvalTask(
        name="mixed_terrain",
        description="Mixed terrain with varying commands",
        terrain_type="mixed",
        terrain_config={
            "difficulty": 0.6,
            # IMPORTANT: Use IsaacLab-like grid layout (tiles + border) so it looks like IsaacLab rough.
            "mixed_layout": "grid",
            "mixed_tile_size": 8.0,
            "mixed_num_rows": 10,
            "mixed_num_cols": 20,
            "mixed_border_width": 20.0,
            "mixed_types": (
                "flat",
                "stepping_stones",
                "wave",
                "cambered",
                "pyramid",
                "rails",
                "washboard",
                # Stairs (IsaacLab HumanoidRoughTerrainsWithStairsCfg)
                "pyramid_stairs",
                "pyramid_stairs_inv",
            ),
            # EXACT match to IsaacLab HumanoidRoughTerrainsCfg (keys + proportions):
            # unitree_lab/terrain/rough.py -> HumanoidRoughTerrainsCfg.sub_terrains
            # If you want stairs included, match HumanoidRoughTerrainsWithStairsCfg instead:
            # flat=0.10, stepping_stones=0.15, wave=0.15, cambered=0.15, pyramid=0.15,
            # rails=0.10, washboard=0.10, stairs_up=0.05, stairs_down=0.05
            "mixed_proportions": (0.10, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05),
            "mixed_spawn_flat": True,
            "mixed_spawn_flat_halfwidth": 1.2,
            "seed": 1,
            # Match IsaacLab parameter ranges as closely as the heightfield implementation allows.
            "stone_width_range": (0.4, 0.2),
            "stone_height_range": (0.05, 0.12),
            "gap_width_range": (0.1, 0.2),
            # IsaacLab wave uses wavelength ~1.5m (=> frequency ~0.6667 cycles/m)
            "wave_amplitude": 0.06,
            "wave_frequency": 1.0 / 1.5,
            "cross_slope_range_deg": (5.0, 15.0),
            "cambered_num_zones": 3,
            "platform_size_ratio": 0.3,
            "rail_height_range": (0.02, 0.06),
            "rail_spacing_range": (0.4, 0.2),
            "washboard_direction": "diagonal",
            "ridge_height_range": (0.01, 0.03),
            "ridge_spacing_range": (0.15, 0.05),
            # Stairs parameters (approximate IsaacLab ranges)
            "step_height": 0.12,
            "step_width": 0.25,
        },
        velocity_command_range=(
            (-0.5, 1.0),  # vx range
            (-0.3, 0.3),  # vy range
            (-0.5, 0.5),  # wz range
        ),
        resample_command_interval=5.0,
        max_episode_steps=1000,
    ),
}


def get_eval_task(name: str) -> LocomotionEvalTask:
    """Get predefined evaluation task by name.
    
    Args:
        name: Task name
        
    Returns:
        Evaluation task configuration
    """
    if name not in LOCOMOTION_EVAL_TASKS:
        available = list(LOCOMOTION_EVAL_TASKS.keys())
        raise ValueError(f"Unknown task '{name}'. Available: {available}")
    
    return LOCOMOTION_EVAL_TASKS[name]


def list_eval_tasks() -> list[str]:
    """List available evaluation tasks."""
    return list(LOCOMOTION_EVAL_TASKS.keys())
