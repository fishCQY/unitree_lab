# Copyright (c) 2024-2025, Light Robotics.
# All rights reserved.

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.utils import configclass

from .actuator_pd import DelayedDCMotor, DelayedImplicitActuator


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = DelayedImplicitActuator

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    base_velocity: dict[str, float] | float | None = None
    """Velocity at which the constant-power region starts."""

    no_load_peak_velocity: dict[str, float] | float | None = None
    """no-load peak velocity."""


@configclass
class DelayedDCMotorCfg(DCMotorCfg):
    """Configuration for a delayed DC motor actuator."""

    class_type: type = DelayedDCMotor

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""
