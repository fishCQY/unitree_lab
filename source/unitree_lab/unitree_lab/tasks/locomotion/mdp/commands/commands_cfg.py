# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass
from isaaclab.envs.mdp import UniformVelocityCommandCfg as VelocityCommandCfg

from .velocity_command import UniformVelocityCommand


@configclass
class UniformVelocityCommandCfg(VelocityCommandCfg):
    """Configuration for the normal velocity command generator."""

    class_type: type = UniformVelocityCommand
