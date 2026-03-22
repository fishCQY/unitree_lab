# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils
from isaaclab.sensors import RayCasterCamera, Imu

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class rigid_body_masses(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(
                self.asset.num_bodies, dtype=torch.int, device="cpu"
            )
        else:
            self.body_ids = torch.tensor(
                self.asset_cfg.body_ids, dtype=torch.int, device="cpu"
            )

        self.sum_mass = torch.sum(
            self.asset.root_physx_view.get_masses()[:, self.body_ids].to(env.device),
            dim=-1,
        ).unsqueeze(-1)
        self.count = 0

    def __call__(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        if self.count < 5:
            self.count += 1
            self.sum_mass = torch.sum(
                self.asset.root_physx_view.get_masses()[:, self.body_ids].to(
                    env.device
                ),
                dim=-1,
            ).unsqueeze(-1)
        return self.sum_mass


class rigid_body_material(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(
                self.asset.num_bodies, dtype=torch.int, device="cpu"
            )
        else:
            self.body_ids = torch.tensor(
                self.asset_cfg.body_ids, dtype=torch.int, device="cpu"
            )

        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(
            None
        ):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
        self.idxs = []
        for body_id in self.body_ids:
            idx = sum(self.num_shapes_per_body[:body_id])
            self.idxs.append(idx)

        materials = self.asset.root_physx_view.get_material_properties()
        self.materials = (
            materials[:, self.idxs].reshape(env.num_envs, -1).to(env.device)
        )

        self.count = 0

    def __call__(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        if self.count < 5:
            self.count += 1
            materials = self.asset.root_physx_view.get_material_properties()
            self.materials = (
                materials[:, self.idxs].reshape(env.num_envs, -1).to(env.device)
            )
        return self.materials


class base_com(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(
                self.asset.num_bodies, dtype=torch.int, device="cpu"
            )
        else:
            self.body_ids = torch.tensor(
                self.asset_cfg.body_ids, dtype=torch.int, device="cpu"
            )

        self.coms = (
            self.asset.root_physx_view.get_coms()[:, self.body_ids, :3]
            .to(env.device)
            .squeeze(1)
        )
        self.count = 0

    def __call__(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        if self.count < 5:
            self.count += 1
            self.coms = (
                self.asset.root_physx_view.get_coms()[:, self.body_ids, :3]
                .to(env.device)
                .squeeze(1)
            )
        return self.coms


def contact_information(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    data = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

    contact_information = torch.sum(torch.square(data), dim=-1) > 0

    return contact_information.float()


def action_delay(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actuators_names: str = "base_legs",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return (
        asset.actuators[actuators_names]
        .positions_delay_buffer.time_lags.float()
        .to(env.device)
        .unsqueeze(1)
    )


def joint_torques(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def joint_accs(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    asset: Articulation = env.scene[asset_cfg.name]
    root_link_quat_w = math_utils.yaw_quat(asset.data.root_link_quat_w).unsqueeze(1)
    root_link_quat_w = root_link_quat_w.expand(-1, contact_force_w.shape[1], -1)
    contact_force_b = math_utils.quat_apply_inverse(root_link_quat_w, contact_force_w)
    return contact_force_b.flatten(1, 2)


def feet_lin_vel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    root_link_quat_w = math_utils.yaw_quat(asset.data.root_link_quat_w).unsqueeze(1)
    root_link_quat_w = root_link_quat_w.expand(-1, body_lin_vel_w.shape[1], -1).contiguous()
    body_lin_vel_b = math_utils.quat_apply_yaw(root_link_quat_w, body_lin_vel_w)
    return body_lin_vel_b.flatten(1)


def push_force(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    external_force_b = asset._external_force_b[:, asset_cfg.body_ids, :]
    return external_force_b.flatten(1)


def push_torque(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    external_torque_b = asset._external_torque_b[:, asset_cfg.body_ids, :]
    return external_torque_b.flatten(1)


def feet_air_time_obs(
        env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    return air_time


def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b


def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_b


def root_link_quat_w(
    env: ManagerBasedEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_link_quat_w
    # make the quaternion real-part positive if configured
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def body_link_pos_b(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    body_pos_xyz_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :3] - asset.data.root_link_pos_w[:, :3].unsqueeze(1)
    body_pos_xyz = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_link_quat_w.unsqueeze(1)), body_pos_xyz_w
    )
    return body_pos_xyz.flatten(1)


def body_link_lin_vel_b(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    body_lin_vel_b = math_utils.quat_apply_inverse(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids],
        asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    )
    return body_lin_vel_b.flatten(1)


def body_link_ang_vel_b(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    body_ang_vel_b = math_utils.quat_apply_inverse(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids],
        asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]
    )
    return body_ang_vel_b.flatten(1)


def key_points_pos_b(
    env: ManagerBasedEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    keypoints_b = env.command_manager.get_term(command_name).cfg.side_length * \
        torch.eye(3, device=asset.device, dtype=torch.float)
    keypoints_b = keypoints_b.unsqueeze(0).expand(body_link_pos_w.shape[0], -1, -1, -1)
    keypoints_w = math_utils.quat_apply(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids].unsqueeze(2).expand(-1, -1, keypoints_b.shape[2], -1),
        keypoints_b.expand(-1, body_link_pos_w.shape[1], -1, -1)
    ) + body_link_pos_w.unsqueeze(2)
    keypoints = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_link_quat_w.unsqueeze(1).unsqueeze(1)).expand(-1, keypoints_w.shape[1], keypoints_w.shape[2], -1),
        keypoints_w - asset.data.root_link_pos_w.unsqueeze(1).unsqueeze(1)
    )
    return keypoints.flatten(1)


def root_link_repr_6d(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    add_noise: bool = False, noise_range: tuple[float, float] = (-0.1, 0.1)
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    quat_w_xyz = asset.data.root_link_quat_w.clone()
    if add_noise:
        euler_angle = math_utils.euler_xyz_from_quat(quat_w_xyz)
        roll = euler_angle[0] + torch.rand_like(euler_angle[0]) * (noise_range[1] - noise_range[0]) + noise_range[0]
        pitch = euler_angle[1] + torch.rand_like(euler_angle[1]) * (noise_range[1] - noise_range[0]) + noise_range[0]
        yaw = euler_angle[2] + torch.rand_like(euler_angle[2]) * (noise_range[1] - noise_range[0]) + noise_range[0]
        quat_w_xyz = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    rot_matrix = math_utils.matrix_from_quat(quat_w_xyz)
    repr_6d: torch.Tensor = rot_matrix[..., :, :2]
    return repr_6d.reshape(*quat_w_xyz.shape[:-1], 6)

def imu_repr_6d(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
    add_noise: bool = False, noise_range: tuple[float, float] = (-0.1, 0.1)
) -> torch.Tensor:
    asset: Imu = env.scene[asset_cfg.name]
    quat_w_xyz = asset.data.quat_w.clone()
    if add_noise:
        euler_angle = math_utils.euler_xyz_from_quat(quat_w_xyz)
        roll = euler_angle[0] + torch.rand_like(euler_angle[0]) * (noise_range[1] - noise_range[0]) + noise_range[0]
        pitch = euler_angle[1] + torch.rand_like(euler_angle[1]) * (noise_range[1] - noise_range[0]) + noise_range[0]
        yaw = euler_angle[2] + torch.rand_like(euler_angle[2]) * (noise_range[1] - noise_range[0]) + noise_range[0]
        quat_w_xyz = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    rot_matrix = math_utils.matrix_from_quat(quat_w_xyz)
    repr_6d: torch.Tensor = rot_matrix[..., :, :2]
    return repr_6d.reshape(*quat_w_xyz.shape[:-1], 6)


def depth_scan(
    env: ManagerBasedEnv, 
    sensor_cfg: SceneEntityCfg = None,
    depth_range: tuple[float, float] | None = None,
    normalize_range: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Get depth image from ray caster camera with optional normalization.
    
    Args:
        env: The environment instance
        sensor_cfg: Scene entity configuration for the depth camera
        depth_range: (min_depth, max_depth) for normalization. If None, no normalization is applied.
        normalize_range: (min, max) range for normalized values. Default is (0.0, 1.0).
        
    Returns:
        Depth image tensor of shape (env_nums, height, width, 1)
    """
    sensor: RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    depth_distances = sensor.data.output["distance_to_image_plane"]  # env_nums * height * width * 1
 
    # Apply normalization if depth_range is provided
    if depth_range is not None:
        min_depth, max_depth = depth_range
        range_min, range_max = normalize_range
        
        # Clamp depth values to valid range
        depth_distances = torch.clamp(depth_distances, min_depth, max_depth)
        
        # Linear normalization: (depth - min_depth) / (max_depth - min_depth) * (range_max - range_min) + range_min
        depth_range_size = max_depth - min_depth
        if depth_range_size > 0:
            normalized = (depth_distances - min_depth) / depth_range_size
            normalized = normalized * (range_max - range_min) + range_min
            depth_distances = normalized
    
    return depth_distances


# =============================================================================
# Critic temporal observations (privileged information)
# =============================================================================


def episode_time_remaining(env: ManagerBasedEnv) -> torch.Tensor:
    """Fraction of episode time remaining. Shape: (num_envs, 1)."""
    current_time = env.episode_length_buf * env.step_dt
    max_time = env.max_episode_length_s
    return ((max_time - current_time) / max_time).unsqueeze(1)


def time_until_next_resample(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity",
) -> torch.Tensor:
    """Normalized time until next command resample. Shape: (num_envs, 1)."""
    vel_cmd = env.command_manager._terms.get(command_name)
    if vel_cmd is None:
        return torch.zeros((env.num_envs, 1), device=env.device)
    time_left = vel_cmd.time_left
    max_resample = vel_cmd.cfg.resampling_time_range[1]
    return (time_left / max(max_resample, 1e-6)).unsqueeze(1)


def time_until_next_push(
    env: ManagerBasedEnv, time_scale: float = 0.2,
) -> torch.Tensor:
    """Scaled time until next push event. Shape: (num_envs, 1)."""
    event_mgr = env.event_manager
    try:
        mode_names = event_mgr._mode_term_names.get("interval", [])
        push_idx = mode_names.index("push_robot") if "push_robot" in mode_names else None
    except (AttributeError, ValueError):
        push_idx = None

    if push_idx is None:
        for attr_name in ["base_external_force_torque"]:
            try:
                idx = mode_names.index(attr_name)
                push_idx = idx
                break
            except (ValueError, UnboundLocalError):
                pass

    if push_idx is None:
        return torch.zeros(env.num_envs, 1, device=env.device)

    time_left = event_mgr._interval_term_time_left[push_idx]
    return (time_left * time_scale).unsqueeze(1)


def robot_mass_obs(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Total robot mass as observation. Shape: (num_envs, 1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses()
    return masses.sum(dim=1, keepdim=True).to(env.device)


# =============================================================================
# AMP Plugin observation terms (single-step, for use with AMPPlugin)
# =============================================================================


def amp_joint_pos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint positions for AMP observation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] if asset_cfg.joint_ids is not None else asset.data.joint_pos


def amp_joint_vel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint velocities for AMP observation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] if asset_cfg.joint_ids is not None else asset.data.joint_vel


def amp_base_ang_vel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base angular velocity in body frame for AMP observation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_b


def amp_projected_gravity(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Projected gravity vector in body frame for AMP observation."""
    asset: Articulation = env.scene[asset_cfg.name]
    grav_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
    return math_utils.quat_apply_inverse(asset.data.root_link_quat_w, grav_w)


def amp_body_pos_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Key body positions in body frame for AMP observation.

    Computes positions of selected bodies relative to root, expressed in
    the body (root) frame. This provides task-space features that complement
    the joint-space features (dof_pos, dof_vel).

    Configure ``asset_cfg.body_ids`` to select which bodies to track
    (e.g., knees and elbows).

    Returns:
        Tensor of shape (num_envs, num_bodies * 3).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    relative_pos = asset.data.body_pos_w[:, body_ids] - asset.data.root_pos_w[:, None, :]
    num_bodies = relative_pos.shape[1]
    root_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, num_bodies, -1)
    pos_b = math_utils.quat_apply_inverse(root_quat, relative_pos)
    return pos_b.reshape(env.num_envs, -1)


def vel_cmd_condition_id(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    vx_index: int = 0,
    vx_threshold: float = 1.1,
) -> torch.Tensor:
    """Velocity command condition ID for conditional AMP.

    Maps the current velocity command to a discrete condition ID:
      - 0 (walk) if |vx| <= vx_threshold
      - 1 (run)  if |vx| > vx_threshold

    Returns:
        Tensor of shape (num_envs, 1) with integer condition IDs.
    """
    vx = env.command_manager.get_command(command_name)[:, vx_index]
    cond_id = (vx.abs() > vx_threshold).long()
    return cond_id.unsqueeze(-1)


