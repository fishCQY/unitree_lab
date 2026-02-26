# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
import unitree_lab.tasks.locomotion.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_buffer: float = 3.0,
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = (
            torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        )
        y_out_of_bounds = (
            torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        )
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError(
            "Received unsupported terrain type, must be either 'plane' or 'generator'."
        )


def falling(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0,
    probability: float = 0.02,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    random_values = torch.rand(env.num_envs, device=env.device)
    return (asset.data.projected_gravity_b[:, 2] > threshold) & (random_values < probability)


def end_effector_tracking_error_termination(
    env: ManagerBasedRLEnv,
    command_name: str,
    probability: float = 0.005,
    threshold: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the command is not met."""
    asset: Articulation = env.scene[asset_cfg.name]
    # Convert the target body positions to the asset's root link frame
    body_pos_xyz = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_link_quat_w.unsqueeze(1)),
        asset.data.body_link_pos_w[:, asset_cfg.body_ids] - asset.data.root_link_pos_w.unsqueeze(1)
    )
    body_names = [asset.body_names[i] for i in asset_cfg.body_ids]
    body_pos_xyz_target = env.command_manager.get_term(command_name).tracking_body_pos(body_names)
    distance = torch.norm(body_pos_xyz_target - body_pos_xyz, dim=-1).max(dim=-1).values

    random_values = torch.rand(env.num_envs, device=env.device)
    return (distance > threshold) & (random_values < probability)


def root_pos_err_termination(
    env: ManagerBasedRLEnv,
    command_name: str,
    probability: float = 0.005,
    threshold: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    random_values = torch.rand(env.num_envs, device=env.device)
    command: mdp.MotionTrackingCommand = env.command_manager.get_term(command_name)
    pos_err = command.current_body_positions_w[:, command.root_link_ids].squeeze(1) - asset.data.root_link_pos_w
    return (torch.abs(pos_err[:, 2]) > threshold) & (random_values < probability)


def root_quat_error_magnitude_termination(
    env: ManagerBasedRLEnv,
    command_name: str,
    probability: float = 0.005,
    threshold: float = 1.57,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    random_values = torch.rand(env.num_envs, device=env.device)
    command: mdp.MotionTrackingCommand = env.command_manager.get_term(command_name)
    quat_error_magnitude = math_utils.quat_error_magnitude(
        command.current_body_orientations[:, command.root_link_ids].squeeze(1),
        asset.data.root_link_quat_w
    )
    return (quat_error_magnitude > threshold) & (random_values < probability)
