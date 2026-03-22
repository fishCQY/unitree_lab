# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING


import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import UniformVelocityCommand as VelocityCommand


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import UniformVelocityCommandCfg


class UniformVelocityCommand(VelocityCommand):
    """Velocity command generator with per-env ranges and terrain-aware overrides."""

    cfg: UniformVelocityCommandCfg

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.lin_vel_x_ranges = torch.zeros(self.num_envs, 2, device=self.device)
        self.lin_vel_y_ranges = torch.zeros(self.num_envs, 2, device=self.device)
        self.ang_vel_z_ranges = torch.zeros(self.num_envs, 2, device=self.device)

        self.lin_vel_x_ranges[:, 0] = self.cfg.ranges.lin_vel_x[0]
        self.lin_vel_x_ranges[:, 1] = self.cfg.ranges.lin_vel_x[1]
        self.lin_vel_y_ranges[:, 0] = self.cfg.ranges.lin_vel_y[0]
        self.lin_vel_y_ranges[:, 1] = self.cfg.ranges.lin_vel_y[1]
        self.ang_vel_z_ranges[:, 0] = self.cfg.ranges.ang_vel_z[0]
        self.ang_vel_z_ranges[:, 1] = self.cfg.ranges.ang_vel_z[1]

        self._col_vel_x_range: torch.Tensor | None = None
        self._col_vel_y_range: torch.Tensor | None = None
        self._col_vel_z_range: torch.Tensor | None = None

        if getattr(self.cfg, "terrain_velocity_ranges", None):
            self._setup_terrain_aware_velocity()

    def _setup_terrain_aware_velocity(self):
        """Precompute per-column velocity range tensors from terrain_velocity_ranges."""
        terrain = self._env.scene.terrain
        if terrain.cfg.terrain_generator is None:
            print("[WARNING] terrain_velocity_ranges set but no terrain generator. Disabling.")
            return

        gen_cfg = terrain.cfg.terrain_generator
        sub_terrains = gen_cfg.sub_terrains
        terrain_names = list(sub_terrains.keys())
        num_cols = gen_cfg.num_cols

        proportions = [sub_terrains[n].proportion for n in terrain_names]
        total = sum(proportions)
        proportions = [p / total for p in proportions]

        col_to_name: dict[int, str] = {}
        cur = 0
        for i, name in enumerate(terrain_names):
            w = (num_cols - cur) if i == len(terrain_names) - 1 else max(1 if proportions[i] > 0 else 0, int(proportions[i] * num_cols + 1e-4))
            for c in range(cur, cur + w):
                col_to_name[c] = name
            cur += w

        dx = (self.cfg.ranges.lin_vel_x[0], self.cfg.ranges.lin_vel_x[1])
        dy = (self.cfg.ranges.lin_vel_y[0], self.cfg.ranges.lin_vel_y[1])
        dz = (self.cfg.ranges.ang_vel_z[0], self.cfg.ranges.ang_vel_z[1])

        self._col_vel_x_range = torch.tensor([[dx[0], dx[1]]] * num_cols, device=self.device)
        self._col_vel_y_range = torch.tensor([[dy[0], dy[1]]] * num_cols, device=self.device)
        self._col_vel_z_range = torch.tensor([[dz[0], dz[1]]] * num_cols, device=self.device)

        for col_idx, tname in col_to_name.items():
            tname_lower = tname.lower()
            for keyword, vel_ranges in self.cfg.terrain_velocity_ranges.items():
                if keyword.lower() in tname_lower:
                    vx, vy, vz = vel_ranges
                    self._col_vel_x_range[col_idx] = torch.tensor([vx[0], vx[1]], device=self.device)
                    self._col_vel_y_range[col_idx] = torch.tensor([vy[0], vy[1]], device=self.device)
                    self._col_vel_z_range[col_idx] = torch.tensor([vz[0], vz[1]], device=self.device)
                    break

    def _apply_terrain_aware_velocity(self, env_ids):
        """Override velocity with terrain-specific ranges."""
        terrain = self._env.scene.terrain
        cols = terrain.terrain_types[env_ids]
        n = len(env_ids)
        rand = torch.rand(n, 3, device=self.device)
        self.vel_command_b[env_ids, 0] = self._col_vel_x_range[cols, 0] + rand[:, 0] * (self._col_vel_x_range[cols, 1] - self._col_vel_x_range[cols, 0])
        self.vel_command_b[env_ids, 1] = self._col_vel_y_range[cols, 0] + rand[:, 1] * (self._col_vel_y_range[cols, 1] - self._col_vel_y_range[cols, 0])
        self.vel_command_b[env_ids, 2] = self._col_vel_z_range[cols, 0] + rand[:, 2] * (self._col_vel_z_range[cols, 1] - self._col_vel_z_range[cols, 0])

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)

        self.vel_command_b[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * \
            (self.lin_vel_x_ranges[env_ids, 1] - self.lin_vel_x_ranges[env_ids, 0]) + \
            self.lin_vel_x_ranges[env_ids, 0]
        self.vel_command_b[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * \
            (self.lin_vel_y_ranges[env_ids, 1] - self.lin_vel_y_ranges[env_ids, 0]) + \
            self.lin_vel_y_ranges[env_ids, 0]
        self.vel_command_b[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * \
            (self.ang_vel_z_ranges[env_ids, 1] - self.ang_vel_z_ranges[env_ids, 0]) + \
            self.ang_vel_z_ranges[env_ids, 0]

        # Override with terrain-specific ranges if configured
        if self._col_vel_x_range is not None:
            self._apply_terrain_aware_velocity(env_ids)

        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.ang_vel_z_ranges[env_ids, 0],
                max=self.ang_vel_z_ranges[env_ids, 1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0
