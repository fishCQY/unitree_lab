# Copyright (c) 2024-2025, unitree_lab contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for delayed IMU sensor (ported from unitree_lab)."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.sensors.imu import ImuCfg
from isaaclab.utils import configclass

from .delayed_imu import DelayedImu


@configclass
class DelayedImuCfg(ImuCfg):
    """IMU config with communication delay simulation."""

    class_type: type = DelayedImu

    delay_range: tuple[float, float] = MISSING
    """Delay range in seconds, sampled per env at reset."""

