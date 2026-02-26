# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast camera sensor with noise (ported from unitree_lab)."""

from __future__ import annotations

from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.utils import configclass

from .ray_caster_camera import NoiseRayCasterCamera


@configclass
class NoiseRayCasterCameraCfg(RayCasterCameraCfg):
    """Ray-caster camera configuration with noise/latency modeling."""

    class_type: type = NoiseRayCasterCamera

    @configclass
    class DepthSensorNoiseCfg:
        # Enable/disable
        enable: bool = False

        # Gaussian noise parameters
        gaussian_mean: float = 0.0
        gaussian_std: float = 0.01

        # Random dropout parameters
        dropout_prob: float = 0.0

        # Depth range filtering
        range_min: float = 0.1
        range_max: float = 10.0

        # Invalid value for dropout/out-of-range/inf/nan
        invalid_value: float | str = "max"  # "max" | "min" | "inf" | float

        # Inf/NaN handling
        handle_invalid_as_max: bool = True

        # Latency simulation
        latency_steps: int = 0

    noise_cfg: DepthSensorNoiseCfg = DepthSensorNoiseCfg()

