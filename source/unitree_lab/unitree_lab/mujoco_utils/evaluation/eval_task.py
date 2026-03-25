# Copyright (c) 2024-2026, unitree_lab contributors.

"""Evaluation task definitions: terrain presets, velocity command callables, and task sets.

This module provides:
atomic and composite velocity commands, terrain dictionaries for
:class:`~unitree_lab.mujoco_utils.terrain.generator.MujocoTerrainGenerator`, and
predefined :class:`EvalTask` instances for batch sim2sim.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# =============================================================================
# Velocity Command Utilities
# =============================================================================


def _v(vx: float, vy: float = 0, yaw: float = 0) -> np.ndarray:
    """Create velocity array."""
    return np.array([vx, vy, yaw], dtype=np.float32)


def _const(vx: float, vy: float = 0, yaw: float = 0) -> Callable[[float], np.ndarray]:
    """Create constant velocity command."""
    vel = _v(vx, vy, yaw)
    return lambda t: vel


def _cyclic(period: float, *segments: tuple[float, list]) -> Callable[[float], np.ndarray]:
    """Create cyclic velocity command.

    Args:
        period: Total cycle period in seconds.
        segments: (threshold, [vx, vy, yaw]) tuples. Returns velocity if (t % period) < threshold.
    """

    def cmd(t: float) -> np.ndarray:
        c = t % period
        for thresh, vel in segments:
            if c < thresh:
                return _v(*vel)
        return _v(*segments[-1][1])

    return cmd


def _sequence(profile: list[tuple]) -> Callable[[float], np.ndarray]:
    """Create velocity command from time-velocity profile.

    Args:
        profile: List of (end_time, velocity_or_func) tuples.
                 velocity can be [vx, vy, yaw] list or a callable(t) -> ndarray
    """

    def cmd(t: float) -> np.ndarray:
        for end_time, vel in profile:
            if t < end_time:
                return _v(*vel(t)) if callable(vel) else _v(*vel)
        return _v(0, 0, 0)

    return cmd


def with_warmup(vel_fn: Callable, warmup: float = 2.0) -> Callable[[float], np.ndarray]:
    """Wrap velocity function with warmup period (stand still first)."""
    zero = _v(0, 0, 0)

    def wrapped(t: float) -> np.ndarray:
        return zero if t < warmup else vel_fn(t - warmup)

    return wrapped


# =============================================================================
# Atomic Velocity Commands
# =============================================================================

# Constant velocities
vel_cmd_zero = _const(0, 0, 0)
vel_cmd_forward_slow = _const(0.5)
vel_cmd_forward_medium = _const(1.0)
vel_cmd_forward_fast = _const(1.5)
vel_cmd_backward = _const(-0.5)
vel_cmd_strafe_left = _const(0, 0.5)
vel_cmd_strafe_right = _const(0, -0.5)
vel_cmd_turn_left = _const(0, 0, 0.6)
vel_cmd_turn_right = _const(0, 0, -0.6)

# Circular motion: r = v / ω
vel_cmd_circle_fast = _const(1.0, 0, 1.0)  # r = 1m
vel_cmd_circle_slow = _const(0.8, 0, 0.4)  # r = 2m


def vel_cmd_figure_eight(t: float) -> np.ndarray:
    """Figure-8: alternating circles."""
    yaw = 0.6 if int(t / 10.0) % 2 == 0 else -0.6
    return _v(1.0, 0, yaw)


# Cyclic patterns
vel_cmd_rapid_start_stop = _cyclic(4.0, (1.5, [1.2, 0, 0]), (4.0, [0, 0, 0]))
vel_cmd_zigzag = _cyclic(2.0, (1.0, [1.0, 0.4, 0.5]), (2.0, [1.0, -0.4, -0.5]))
vel_cmd_sprint_brake = _cyclic(5.0, (2.0, [1.5, 0, 0]), (2.5, [0, 0, 0]), (3.5, [-0.5, 0, 0]), (5.0, [0, 0, 0]))
vel_cmd_spin_walk = _cyclic(6.0, (3.0, [0.6, 0, 1.2]), (6.0, [0.6, 0, -1.2]))

# Chaotic: sequence of different patterns
_CHAOS_PATTERNS = [
    [1.0, 0, 0],
    [0, 0, 1.0],
    [0.8, 0.4, 0],
    [-0.5, 0, 0],
    [0, 0, -1.0],
    [1.0, -0.4, 0.4],
    [0, 0, 0],
    [1.2, 0, 0.6],
    [0, 0.5, 0],
    [-0.3, 0, 0.6],
]


def vel_cmd_chaos(t: float) -> np.ndarray:
    """Chaotic velocity pattern."""
    return _v(*_CHAOS_PATTERNS[int(t / 2.0) % len(_CHAOS_PATTERNS)])


# =============================================================================
# Comprehensive Combined Commands
# =============================================================================


def vel_cmd_omnidirectional(t: float) -> np.ndarray:
    """Comprehensive flat terrain test: all directions. Duration: 50s, ~20m max displacement."""
    V, VY, YAW = 1.0, 0.5, 0.6
    profile = [
        (8, [V, 0, 0]),  # Forward
        (10, [0, 0, 0]),  # Stop
        (18, [V, 0, YAW / 2]),  # Forward + curve left
        (20, [0, 0, 0]),  # Stop
        (25, [-V / 3, 0, 0]),  # Backward
        (27, [0, 0, 0]),  # Stop
        (32, [0, VY, 0]),  # Strafe left
        (34, [0, 0, 0]),  # Stop
        (39, [0, -VY, 0]),  # Strafe right
        (41, [0, 0, 0]),  # Stop
        (46, [V, 0, -YAW / 2]),  # Forward + curve right
        (50, [0, 0, 0]),  # Final stop
    ]
    for end_time, vel in profile:
        if t < end_time:
            return _v(*vel)
    return _v(0, 0, 0)


# Terrain-specific comprehensive command: higher speed for testing
vel_cmd_terrain_comprehensive = _sequence(
    [
        # Phase 1: Steady forward (8s, ~8m)
        (8, [1.0, 0, 0]),
        (10, [0, 0, 0]),
        # Phase 2: Circular motion (12s)
        (16, [0.8, 0, 0.5]),  # Circle left
        (18, [0, 0, 0]),
        (24, [0.8, 0, -0.5]),  # Circle right
        (26, [0, 0, 0]),
        # Phase 3: Start-stop cycles (12s, ~6m)
        (38, lambda t: [1.0, 0, 0] if ((t - 26) % 4.0) < 2.0 else [0, 0, 0]),
        (40, [0, 0, 0]),
        # Phase 4: Zigzag (8s, ~8m)
        (48, lambda t: [1.0, 0.3, 0.4] if ((t - 40) % 2.0) < 1.0 else [1.0, -0.3, -0.4]),
        (50, [0, 0, 0]),
    ]
)


# =============================================================================
# Terrain Presets
# =============================================================================


def _terrain_single(type_name: str, **params) -> dict:
    """Create single-type terrain config."""
    return {type_name: {"proportion": 1.0, **params}}


# Flat
TERRAIN_FLAT = {"plane": {"proportion": 1.0}}

# Random rough
TERRAIN_ROUGH = _terrain_single("RandomUniform", noise_range=(-0.05, 0.05), noise_step=0.01)
TERRAIN_ROUGH_EASY = _terrain_single("RandomUniform", noise_range=(-0.03, 0.03), noise_step=0.01)
TERRAIN_ROUGH_HARD = _terrain_single("RandomUniform", noise_range=(-0.1, 0.1), noise_step=0.01)

# Stairs
TERRAIN_STAIRS_UP = _terrain_single(
    "InvertedPyramidStairs", step_height_range=(0.05, 0.15), step_width=0.30, platform_width=2.0
)
TERRAIN_STAIRS_UP_EASY = _terrain_single(
    "InvertedPyramidStairs", step_height_range=(0.01, 0.08), step_width=0.36, platform_width=3.0
)
TERRAIN_STAIRS_UP_HARD = _terrain_single(
    "InvertedPyramidStairs", step_height_range=(0.10, 0.19), step_width=0.36, platform_width=3.0
)
TERRAIN_STAIRS_DOWN = _terrain_single(
    "PyramidStairs", step_height_range=(0.05, 0.15), step_width=0.30, platform_width=2.0
)
TERRAIN_STAIRS_DOWN_EASY = _terrain_single(
    "PyramidStairs", step_height_range=(0.01, 0.08), step_width=0.30, platform_width=2.0
)
TERRAIN_STAIRS_DOWN_HARD = _terrain_single(
    "PyramidStairs", step_height_range=(0.10, 0.19), step_width=0.36, platform_width=3.0
)

# Slopes
TERRAIN_SLOPE_UP = _terrain_single("InvertedPyramidSloped", slope_range=(0.0, 0.5), platform_width=2.0)
TERRAIN_SLOPE_UP_EASY = _terrain_single("InvertedPyramidSloped", slope_range=(0.0, 0.2), platform_width=2.0)
TERRAIN_SLOPE_UP_HARD = _terrain_single("InvertedPyramidSloped", slope_range=(0.0, 0.8), platform_width=2.0)
TERRAIN_SLOPE_DOWN = _terrain_single("PyramidSloped", slope_range=(0.0, 0.5), platform_width=2.0)
TERRAIN_SLOPE_DOWN_EASY = _terrain_single("PyramidSloped", slope_range=(0.0, 0.2), platform_width=2.0)
TERRAIN_SLOPE_DOWN_HARD = _terrain_single("PyramidSloped", slope_range=(0.0, 0.8), platform_width=2.0)

# Grid/obstacles
TERRAIN_GRID = _terrain_single("RandomGrid", grid_width=0.45, grid_height_range=(0.01, 0.10), platform_width=3.0)
TERRAIN_GRID_EASY = _terrain_single("RandomGrid", grid_width=0.45, grid_height_range=(0.01, 0.03), platform_width=3.0)
TERRAIN_GRID_HARD = _terrain_single("RandomGrid", grid_width=0.45, grid_height_range=(0.01, 0.15), platform_width=3.0)

# Rails
TERRAIN_RAILS = _terrain_single(
    "Rails", rail_thickness_range=(0.05, 0.1), rail_height_range=(0.01, 0.15), platform_width=2.0
)
TERRAIN_RAILS_EASY = _terrain_single(
    "Rails", rail_thickness_range=(0.05, 0.1), rail_height_range=(0.01, 0.05), platform_width=2.0
)
TERRAIN_RAILS_HARD = _terrain_single(
    "Rails", rail_thickness_range=(0.05, 0.1), rail_height_range=(0.01, 0.2), platform_width=2.0
)

# Mixed
TERRAIN_MIXED = {
    "plane": {"proportion": 0.1},
    "RandomUniform": {"proportion": 0.1, "noise_range": (-0.1, 0.1)},
    "PyramidSloped": {"proportion": 0.1, "slope_range": (0.0, 0.8), "platform_width": 2.0},
    "InvertedPyramidSloped": {"proportion": 0.1, "slope_range": (0.0, 0.8), "platform_width": 2.0},
    "PyramidStairs": {"proportion": 0.1, "step_height_range": (0.01, 0.19), "step_width": 0.36, "platform_width": 3.0},
    "InvertedPyramidStairs": {
        "proportion": 0.1,
        "step_height_range": (0.01, 0.19),
        "step_width": 0.36,
        "platform_width": 3.0,
    },
    "RandomGrid": {"proportion": 0.2, "grid_width": 0.45, "grid_height_range": (0.01, 0.15), "platform_width": 3.0},
    "Rails": {
        "proportion": 0.1,
        "rail_thickness_range": (0.05, 0.1),
        "rail_height_range": (0.01, 0.2),
        "platform_width": 2.0,
    },
}


# =============================================================================
# EvalTask
# =============================================================================


@dataclass
class EvalTask:
    """Evaluation task: terrain + velocity command + duration.

    Example:
        >>> task = EvalTask("flat_forward", TERRAIN_FLAT, vel_cmd_forward_medium, duration=30.0)
        >>> cmd = task.get_velocity_command(t=5.0)
    """

    name: str
    terrain: dict
    vel_cmd_fn: Callable[[float], np.ndarray]
    duration: float = 30.0
    description: str = ""
    warmup: float = 2.0  # Warmup duration (0 = no warmup)

    def __post_init__(self):
        """Apply warmup wrapper if needed."""
        if self.warmup > 0:
            original_fn = self.vel_cmd_fn
            self.vel_cmd_fn = with_warmup(original_fn, self.warmup)
            self.duration += self.warmup

    def get_velocity_command(self, t: float) -> np.ndarray:
        """Get velocity command at time t."""
        return self.vel_cmd_fn(t)


# =============================================================================
# Predefined Tasks
# =============================================================================

# Default task set (slim): flat + slopes + hard terrains + mixed
EVAL_TASKS_DEFAULT: dict[str, EvalTask] = {
    "flat_stand": EvalTask("flat_stand", TERRAIN_FLAT, vel_cmd_zero, 20, "Standing still on flat", warmup=0),
    "slope_comprehensive": EvalTask(
        "slope_comprehensive", TERRAIN_SLOPE_UP, vel_cmd_terrain_comprehensive, 50, "Comprehensive on slope", warmup=0
    ),
    "rough_hard_comprehensive": EvalTask(
        "rough_hard_comprehensive",
        TERRAIN_ROUGH_HARD,
        vel_cmd_terrain_comprehensive,
        50,
        "Hard rough terrain",
        warmup=0,
    ),
    "grid_hard_comprehensive": EvalTask(
        "grid_hard_comprehensive", TERRAIN_GRID_HARD, vel_cmd_terrain_comprehensive, 50, "Hard grid terrain", warmup=0
    ),
    "rails_hard_comprehensive": EvalTask(
        "rails_hard_comprehensive", TERRAIN_RAILS_HARD, vel_cmd_terrain_comprehensive, 50, "Hard rails", warmup=0
    ),
    "stairs_up_hard": EvalTask("stairs_up_hard", TERRAIN_STAIRS_UP_HARD, vel_cmd_forward_slow, 25, "Hard stairs up"),
    "stairs_down_hard": EvalTask(
        "stairs_down_hard", TERRAIN_STAIRS_DOWN_HARD, vel_cmd_forward_slow, 25, "Hard stairs down"
    ),
    "mixed_terrain": EvalTask("mixed_terrain", TERRAIN_MIXED, vel_cmd_omnidirectional, 60, "Mixed terrain", warmup=0),
}

# Very easy task set (slim)
EVAL_TASKS_BABY: dict[str, EvalTask] = {
    "flat_stand": EvalTask("flat_stand", TERRAIN_FLAT, vel_cmd_zero, 20, "Standing still on flat", warmup=0),
    "slope_comprehensive": EvalTask(
        "slope_comprehensive",
        TERRAIN_SLOPE_UP_EASY,
        vel_cmd_terrain_comprehensive,
        50,
        "Comprehensive on slope",
        warmup=0,
    ),
    "rough_easy_comprehensive": EvalTask(
        "rough_easy_comprehensive",
        TERRAIN_ROUGH_EASY,
        vel_cmd_terrain_comprehensive,
        50,
        "Easy rough terrain",
        warmup=0,
    ),
    "grid_easy_comprehensive": EvalTask(
        "grid_easy_comprehensive", TERRAIN_GRID_EASY, vel_cmd_terrain_comprehensive, 50, "Easy grid terrain", warmup=0
    ),
    "rails_easy_comprehensive": EvalTask(
        "rails_easy_comprehensive", TERRAIN_RAILS_EASY, vel_cmd_terrain_comprehensive, 50, "Easy rails", warmup=0
    ),
    "stairs_up_easy": EvalTask("stairs_up_easy", TERRAIN_STAIRS_UP_EASY, vel_cmd_forward_slow, 25, "Easy stairs up"),
    "stairs_down_easy": EvalTask(
        "stairs_down_easy", TERRAIN_STAIRS_DOWN_EASY, vel_cmd_forward_slow, 25, "Easy stairs down"
    ),
}

# Full task set (for local testing)
EVAL_TASKS_FULL: dict[str, EvalTask] = {
    "flat_stand": EvalTask("flat_stand", TERRAIN_FLAT, vel_cmd_zero, 5, "Standing still on flat", warmup=0),
    "flat_comprehensive": EvalTask(
        "flat_comprehensive", TERRAIN_FLAT, vel_cmd_omnidirectional, 50, "Comprehensive movement on flat", warmup=0
    ),
    "slope_up": EvalTask("slope_up", TERRAIN_SLOPE_UP, vel_cmd_forward_medium, 15, "Walking up slope"),
    "slope_down": EvalTask("slope_down", TERRAIN_SLOPE_DOWN, vel_cmd_forward_medium, 15, "Walking down slope"),
    "slope_comprehensive": EvalTask(
        "slope_comprehensive", TERRAIN_SLOPE_UP, vel_cmd_terrain_comprehensive, 50, "Comprehensive on slope", warmup=0
    ),
    "rough_comprehensive": EvalTask(
        "rough_comprehensive", TERRAIN_ROUGH, vel_cmd_terrain_comprehensive, 50, "Rough terrain", warmup=0
    ),
    "rough_hard_comprehensive": EvalTask(
        "rough_hard_comprehensive",
        TERRAIN_ROUGH_HARD,
        vel_cmd_terrain_comprehensive,
        50,
        "Hard rough terrain",
        warmup=0,
    ),
    "grid_comprehensive": EvalTask(
        "grid_comprehensive", TERRAIN_GRID, vel_cmd_terrain_comprehensive, 50, "Grid terrain", warmup=0
    ),
    "grid_hard_comprehensive": EvalTask(
        "grid_hard_comprehensive", TERRAIN_GRID_HARD, vel_cmd_terrain_comprehensive, 50, "Hard grid terrain", warmup=0
    ),
    "rails_comprehensive": EvalTask(
        "rails_comprehensive", TERRAIN_RAILS, vel_cmd_terrain_comprehensive, 50, "Rails", warmup=0
    ),
    "rails_hard_comprehensive": EvalTask(
        "rails_hard_comprehensive", TERRAIN_RAILS_HARD, vel_cmd_terrain_comprehensive, 50, "Hard rails", warmup=0
    ),
    "stairs_up": EvalTask("stairs_up", TERRAIN_STAIRS_UP, vel_cmd_forward_slow, 20, "Simple stairs up"),
    "stairs_down": EvalTask("stairs_down", TERRAIN_STAIRS_DOWN, vel_cmd_forward_slow, 20, "Simple stairs down"),
    "stairs_up_hard": EvalTask("stairs_up_hard", TERRAIN_STAIRS_UP_HARD, vel_cmd_forward_slow, 25, "Hard stairs up"),
    "stairs_down_hard": EvalTask(
        "stairs_down_hard", TERRAIN_STAIRS_DOWN_HARD, vel_cmd_forward_slow, 25, "Hard stairs down"
    ),
    "mixed_terrain": EvalTask("mixed_terrain", TERRAIN_MIXED, vel_cmd_omnidirectional, 60, "Mixed terrain", warmup=0),
}

# Default: use slim task set for training evaluation
EVAL_TASKS = EVAL_TASKS_DEFAULT


def get_eval_task(name: str) -> EvalTask:
    """Get predefined eval task by name (searches all sets).

    Args:
        name: Task name.

    Returns:
        EvalTask instance.

    Raises:
        ValueError: If task name is not found.
    """
    if name in EVAL_TASKS_DEFAULT:
        return EVAL_TASKS_DEFAULT[name]
    if name in EVAL_TASKS_FULL:
        return EVAL_TASKS_FULL[name]
    if name in EVAL_TASKS_BABY:
        return EVAL_TASKS_BABY[name]
    all_tasks = (
        set(EVAL_TASKS_DEFAULT.keys()) | set(EVAL_TASKS_FULL.keys()) | set(EVAL_TASKS_BABY.keys())
    )
    raise ValueError(f"Unknown task: '{name}'. Available: {', '.join(sorted(all_tasks))}")


def list_eval_tasks(full: bool = False) -> list[str]:
    """List available task names.

    Args:
        full: If True, return all tasks. If False, return default (slim) set.

    Returns:
        List of task names.
    """
    return list(EVAL_TASKS_FULL.keys() if full else EVAL_TASKS_DEFAULT.keys())
