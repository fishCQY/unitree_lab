# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def apply_external_force_torque_stochastic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # clear the existing forces and torques
    asset._external_force_b *= 0
    asset._external_torque_b *= 0

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    random_values = torch.rand(env_ids.shape, device=env_ids.device)
    mask = random_values < probability
    masked_env_ids = env_ids[mask]

    if len(masked_env_ids) == 0:
        return

    # resolve number of bodies
    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
    )

    # sample random forces and torques
    size = (len(masked_env_ids), num_bodies, 3)
    force_range_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    force_range = torch.tensor(force_range_list, device=asset.device)
    forces = math_utils.sample_uniform(
        force_range[:, 0], force_range[:, 1], size, asset.device
    )
    torque_range_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    torque_range = torch.tensor(torque_range_list, device=asset.device)
    torques = math_utils.sample_uniform(
        torque_range[:, 0], torque_range[:, 1], size, asset.device
    )
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(
        forces, torques, env_ids=masked_env_ids, body_ids=asset_cfg.body_ids
    )

from typing import TYPE_CHECKING, Literal
from isaaclab.envs.mdp.events import _randomize_prop_by_op


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


# =============================================================================
# Domain randomization: joint friction and motor Kt (aligned with bfm_training)
# =============================================================================

from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import EventTermCfg
from isaaclab.actuators import ImplicitActuator


class randomize_actuator_gains_coupled(ManagerTermBase):
    """Randomize stiffness and damping with a shared factor per joint (motor Kt model).

    Unlike independent randomization, this applies the same random scale to
    both Kp and Kd, modeling motor torque constant variations.
    """

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

    def __call__(
        self, env, env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        distribution_params: tuple[float, float] = (0.875, 1.075),
        operation: str = "scale",
        distribution: str = "uniform",
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        for actuator in self.asset.actuators.values():
            if isinstance(self.asset_cfg.joint_ids, slice):
                act_indices = slice(None)
                global_indices = actuator.joint_indices if not isinstance(actuator.joint_indices, slice) else slice(None)
            else:
                asset_jids = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
                act_jids = actuator.joint_indices
                if isinstance(act_jids, slice):
                    act_indices = asset_jids
                    global_indices = asset_jids
                else:
                    act_indices = torch.nonzero(torch.isin(act_jids, asset_jids)).view(-1)
                    if len(act_indices) == 0:
                        continue
                    global_indices = act_jids[act_indices]

            factor = math_utils.sample_uniform(
                distribution_params[0], distribution_params[1],
                actuator.stiffness[env_ids].shape, self.asset.device,
            )

            stiffness = actuator.stiffness[env_ids].clone()
            stiffness[:, act_indices] = self.asset.data.default_joint_stiffness[env_ids][:, global_indices] * factor[:, act_indices]
            actuator.stiffness[env_ids] = stiffness
            if isinstance(actuator, ImplicitActuator):
                self.asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)

            damping = actuator.damping[env_ids].clone()
            damping[:, act_indices] = self.asset.data.default_joint_damping[env_ids][:, global_indices] * factor[:, act_indices]
            actuator.damping[env_ids] = damping
            if isinstance(actuator, ImplicitActuator):
                self.asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)


class randomize_joint_coulomb_friction(ManagerTermBase):
    """Randomize static and dynamic Coulomb friction coefficients of joints.

    Uses IsaacLab's write_joint_friction_coefficient_to_sim API.
    Ensures dynamic friction does not exceed static friction.
    """

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

    def __call__(
        self, env, env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        static_friction_range: tuple[float, float] = (0.7, 1.3),
        dynamic_friction_range: tuple[float, float] = (0.7, 1.3),
        operation: str = "scale",
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        joint_ids = self.asset_cfg.joint_ids if self.asset_cfg.joint_ids != slice(None) else slice(None)

        static_coeff = self.asset.data.default_joint_friction_coeff.clone()
        static_coeff = _randomize_prop_by_op(
            static_coeff, static_friction_range, env_ids, joint_ids, operation=operation,
        )
        static_coeff = torch.clamp(static_coeff, min=0.0)

        dynamic_coeff = self.asset.data.default_joint_friction_coeff.clone() * 0.5
        dynamic_coeff = _randomize_prop_by_op(
            dynamic_coeff, dynamic_friction_range, env_ids, joint_ids, operation=operation,
        )
        dynamic_coeff = torch.clamp(dynamic_coeff, min=0.0)
        dynamic_coeff = torch.minimum(dynamic_coeff, static_coeff)

        env_slice = env_ids[:, None] if isinstance(joint_ids, torch.Tensor) else env_ids
        self.asset.write_joint_friction_coefficient_to_sim(
            joint_friction_coeff=static_coeff[env_slice, joint_ids],
            joint_dynamic_friction_coeff=dynamic_coeff[env_slice, joint_ids],
            joint_ids=joint_ids,
            env_ids=env_ids,
        )


class randomize_joint_viscous_friction(ManagerTermBase):
    """Randomize viscous friction coefficient of joints (velocity-proportional resistance)."""

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

    def __call__(
        self, env, env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        viscous_friction_range: tuple[float, float] = (0.0, 0.05),
        operation: str = "add",
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        joint_ids = self.asset_cfg.joint_ids if self.asset_cfg.joint_ids != slice(None) else slice(None)

        viscous = torch.zeros_like(self.asset.data.default_joint_friction_coeff)
        viscous = _randomize_prop_by_op(
            viscous, viscous_friction_range, env_ids, joint_ids, operation=operation,
        )
        viscous = torch.clamp(viscous, min=0.0)

        env_slice = env_ids[:, None] if isinstance(joint_ids, torch.Tensor) else env_ids
        self.asset.write_joint_friction_coefficient_to_sim(
            joint_friction_coeff=self.asset.data.joint_friction_coeff[env_slice, joint_ids],
            joint_viscous_friction_coeff=viscous[env_slice, joint_ids],
            joint_ids=joint_ids,
            env_ids=env_ids,
        )