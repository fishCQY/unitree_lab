# Copyright (c) 2024-2025, Light Robotics.
# All rights reserved.

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaacsim.core.utils.types import ArticulationActions

from isaaclab.actuators import DCMotor, ImplicitActuator
from isaaclab.utils import DelayBuffer

if TYPE_CHECKING:
    from .actuator_cfg import DelayedDCMotorCfg, DelayedImplicitActuatorCfg


class DelayedImplicitActuator(ImplicitActuator):
    """Implicit PD actuator with delayed command application.

    This class extends the :class:`ImplicitActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: DelayedImplicitActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # all of the envs
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)
        # check if torque-speed curve is used
        if self.cfg.base_velocity is None and self.cfg.no_load_peak_velocity is None:
            # No torque-speed curve, use simple clipping
            self.use_torque_speed_curve = False
        else:
            if self.cfg.base_velocity is None or self.cfg.no_load_peak_velocity is None:
                raise ValueError(
                    "base_velocity and no_load_peak_velocity must be set together to enable torque-speed curve"
                )
            self.base_velocity = self._parse_joint_parameter(self.cfg.base_velocity, None)
            self.no_load_peak_velocity = self._parse_joint_parameter(self.cfg.no_load_peak_velocity, None)
            # Use custom torque-speed curve
            self.use_torque_speed_curve = True
            self._saturation_effort = (
                self.effort_limit_sim * self.no_load_peak_velocity / (self.no_load_peak_velocity - self.base_velocity + 1e-6)
            )
            self._vel_at_effort_lim = self.no_load_peak_velocity * (1 + self.effort_limit_sim / self._saturation_effort)

        # prepare joint vel buffer for max effort computation
        self._joint_vel = torch.zeros_like(self.computed_effort)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # save current joint vel
        self._joint_vel = joint_vel
        # compte actuator model
        control_action = super().compute(control_action, joint_pos, joint_vel)
        if self.use_torque_speed_curve:
            # use control_action.joint_efforts to lower the applied effort to the approximate torque clip
            control_action.joint_efforts = self.applied_effort - self.computed_effort
        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        if not self.use_torque_speed_curve:
            # No torque-speed curve, use parent's simple clipping
            return super()._clip_effort(effort)

        # Use custom torque-speed curve
        # Clip joint velocity to valid range
        self._joint_vel[:] = torch.clip(self._joint_vel, min=-self._vel_at_effort_lim, max=self._vel_at_effort_lim)

        # Compute torque limits based on torque-speed curve
        torque_speed_top = self._saturation_effort * (1.0 - self._joint_vel.abs() / self.no_load_peak_velocity)
        torque_speed_bottom = self._saturation_effort * (-1.0 + self._joint_vel.abs() / self.no_load_peak_velocity)

        # Apply effort limits
        max_effort = torch.clip(torque_speed_top, max=self.effort_limit_sim)
        min_effort = torch.clip(torque_speed_bottom, min=-self.effort_limit_sim)

        # Clip the torques based on the motor limits
        clamped = torch.clip(effort, min=min_effort, max=max_effort)
        return clamped


class DelayedDCMotor(DCMotor):
    """DC motor actuator with delayed command application."""

    cfg: DelayedDCMotorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # all of the envs
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # compte actuator model
        return super().compute(control_action, joint_pos, joint_vel)
