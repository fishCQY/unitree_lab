# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass
from isaaclab.envs.mdp import UniformVelocityCommandCfg as VelocityCommandCfg

from .velocity_command import UniformVelocityCommand


@configclass
class UniformVelocityCommandCfg(VelocityCommandCfg):
    """Velocity command generator with optional terrain-aware ranges.

    If ``terrain_velocity_ranges`` is set, environments on different terrain
    types will sample velocity commands from terrain-specific ranges.

    Format::

        terrain_velocity_ranges = {
            "plane": ((-1.5, 2.0), (-0.6, 0.6), (-2.0, 2.0)),
            "stairs": ((-0.5, 0.8), (0.0, 0.0), (-1.0, 1.0)),
        }

    Keywords are matched as case-insensitive substrings of terrain names.
    First match wins. Unmatched terrains use default ``ranges``.
    """

    class_type: type = UniformVelocityCommand

    terrain_velocity_ranges: dict | None = None
    """Optional terrain-keyword → (lin_vel_x, lin_vel_y, ang_vel_z) range mapping."""
