# Copyright (c) 2024-2025, Light Robotics.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Delayed IMU sensor (ported from unitree_lab)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.sensors.imu import Imu, ImuData

if TYPE_CHECKING:
    from .delayed_imu_cfg import DelayedImuCfg


class DelayedImu(Imu):
    """IMU with per-env communication delay (with interpolation)."""

    cfg: "DelayedImuCfg"

    def __init__(self, cfg: "DelayedImuCfg"):
        super().__init__(cfg)
        self.delay_range = cfg.delay_range
        self._max_history = 10

        self._delays: torch.Tensor | None = None
        self._delay_steps_floor: torch.Tensor | None = None
        self._delay_frac: torch.Tensor | None = None

        self._pos_w_hist: torch.Tensor | None = None
        self._quat_w_hist: torch.Tensor | None = None
        self._grav_hist: torch.Tensor | None = None
        self._lin_vel_hist: torch.Tensor | None = None
        self._ang_vel_hist: torch.Tensor | None = None
        self._lin_acc_hist: torch.Tensor | None = None
        self._ang_acc_hist: torch.Tensor | None = None

        self._delayed_data = ImuData()

    def _initialize_impl(self):
        super()._initialize_impl()
        n, d = self._num_envs, self._device

        self._delays = torch.rand(n, device=d) * (self.delay_range[1] - self.delay_range[0]) + self.delay_range[0]
        self._compute_delay_params()

        h = self._max_history
        self._pos_w_hist = torch.zeros(h, n, 3, device=d)
        self._quat_w_hist = torch.zeros(h, n, 4, device=d)
        self._quat_w_hist[:, :, 0] = 1.0
        self._grav_hist = torch.zeros(h, n, 3, device=d)
        self._grav_hist[:, :, 2] = -1.0
        self._lin_vel_hist = torch.zeros(h, n, 3, device=d)
        self._ang_vel_hist = torch.zeros(h, n, 3, device=d)
        self._lin_acc_hist = torch.zeros(h, n, 3, device=d)
        self._ang_acc_hist = torch.zeros(h, n, 3, device=d)

        self._delayed_data.pos_w = torch.zeros(n, 3, device=d)
        self._delayed_data.quat_w = torch.zeros(n, 4, device=d)
        self._delayed_data.quat_w[:, 0] = 1.0
        self._delayed_data.projected_gravity_b = torch.zeros(n, 3, device=d)
        self._delayed_data.projected_gravity_b[:, 2] = -1.0
        self._delayed_data.lin_vel_b = torch.zeros(n, 3, device=d)
        self._delayed_data.ang_vel_b = torch.zeros(n, 3, device=d)
        self._delayed_data.lin_acc_b = torch.zeros(n, 3, device=d)
        self._delayed_data.ang_acc_b = torch.zeros(n, 3, device=d)

    def _compute_delay_params(self):
        delay_steps = self._delays / self._sim_physics_dt
        self._delay_steps_floor = torch.clamp(delay_steps.long(), 0, self._max_history - 2)
        self._delay_frac = (delay_steps - self._delay_steps_floor.float()).unsqueeze(-1)

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if self._delays is None:
            return

        if env_ids is None:
            env_ids = slice(None)
            num_reset = self._num_envs
        else:
            num_reset = len(env_ids)

        self._delays[env_ids] = (
            torch.rand(num_reset, device=self._device) * (self.delay_range[1] - self.delay_range[0]) + self.delay_range[0]
        )
        self._compute_delay_params()

        self._pos_w_hist[:, env_ids] = 0.0
        self._quat_w_hist[:, env_ids] = 0.0
        self._quat_w_hist[:, env_ids, 0] = 1.0
        self._grav_hist[:, env_ids] = 0.0
        self._grav_hist[:, env_ids, 2] = -1.0
        self._lin_vel_hist[:, env_ids] = 0.0
        self._ang_vel_hist[:, env_ids] = 0.0
        self._lin_acc_hist[:, env_ids] = 0.0
        self._ang_acc_hist[:, env_ids] = 0.0

    def update(self, dt: float, force_recompute: bool = False):
        super().update(dt, force_recompute=True)
        if self._pos_w_hist is None:
            return

        # Roll histories
        self._pos_w_hist = self._pos_w_hist.roll(1, dims=0)
        self._quat_w_hist = self._quat_w_hist.roll(1, dims=0)
        self._grav_hist = self._grav_hist.roll(1, dims=0)
        self._lin_vel_hist = self._lin_vel_hist.roll(1, dims=0)
        self._ang_vel_hist = self._ang_vel_hist.roll(1, dims=0)
        self._lin_acc_hist = self._lin_acc_hist.roll(1, dims=0)
        self._ang_acc_hist = self._ang_acc_hist.roll(1, dims=0)

        # Insert newest data at index 0
        self._pos_w_hist[0] = self._data.pos_w
        self._quat_w_hist[0] = self._data.quat_w
        self._grav_hist[0] = self._data.projected_gravity_b
        self._lin_vel_hist[0] = self._data.lin_vel_b
        self._ang_vel_hist[0] = self._data.ang_vel_b
        self._lin_acc_hist[0] = self._data.lin_acc_b
        self._ang_acc_hist[0] = self._data.ang_acc_b

    @property
    def data(self) -> ImuData:
        self._update_outdated_buffers()
        if self._pos_w_hist is None or self._delay_steps_floor is None or self._delay_frac is None:
            return self._data

        idx0 = self._delay_steps_floor
        idx1 = idx0 + 1
        env_idx = torch.arange(self._num_envs, device=self._device)
        frac = self._delay_frac

        self._delayed_data.pos_w = (1 - frac) * self._pos_w_hist[idx0, env_idx] + frac * self._pos_w_hist[idx1, env_idx]
        self._delayed_data.projected_gravity_b = (1 - frac) * self._grav_hist[idx0, env_idx] + frac * self._grav_hist[idx1, env_idx]
        self._delayed_data.lin_vel_b = (1 - frac) * self._lin_vel_hist[idx0, env_idx] + frac * self._lin_vel_hist[idx1, env_idx]
        self._delayed_data.ang_vel_b = (1 - frac) * self._ang_vel_hist[idx0, env_idx] + frac * self._ang_vel_hist[idx1, env_idx]
        self._delayed_data.lin_acc_b = (1 - frac) * self._lin_acc_hist[idx0, env_idx] + frac * self._lin_acc_hist[idx1, env_idx]
        self._delayed_data.ang_acc_b = (1 - frac) * self._ang_acc_hist[idx0, env_idx] + frac * self._ang_acc_hist[idx1, env_idx]

        q0 = self._quat_w_hist[idx0, env_idx]
        q1 = self._quat_w_hist[idx1, env_idx]
        self._delayed_data.quat_w = self._quat_slerp(q0, q1, frac)
        return self._delayed_data

    def _quat_slerp(self, q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0, -q1, q1)
        dot = dot.abs()

        linear_threshold = 0.9995
        linear_mask = dot > linear_threshold

        dot_clamped = torch.clamp(dot, -1.0, 1.0)
        theta = torch.acos(dot_clamped)
        sin_theta = torch.sin(theta)
        sin_theta = torch.where(sin_theta.abs() < 1e-6, torch.ones_like(sin_theta), sin_theta)

        s0 = torch.sin((1 - t) * theta) / sin_theta
        s1 = torch.sin(t * theta) / sin_theta
        result = s0 * q0 + s1 * q1

        linear_result = (1 - t) * q0 + t * q1
        linear_result = linear_result / linear_result.norm(dim=-1, keepdim=True)

        result = torch.where(linear_mask, linear_result, result)
        return result / result.norm(dim=-1, keepdim=True)

    @property
    def delays(self) -> torch.Tensor | None:
        return self._delays

