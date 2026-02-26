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
        # sample velocity commands
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

        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
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
