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
    body_ang_vel_b = math_utils.quat_apply_inverse(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids],
        asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]
    )
    return body_ang_vel_b.flatten(1)


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
# AMP discriminator observation terms
# =============================================================================


class AMPAgentObsTerm(ManagerTermBase):
    """Agent proprioceptive observation for AMP discriminator.

    Returns a 3D tensor (num_envs, disc_obs_steps, obs_dim) containing the
    last N steps of [ang_vel, proj_grav, joint_pos_rel, joint_vel].
    On environment reset, the history buffer is filled with the current frame.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.disc_obs_steps: int = cfg.params.get("disc_obs_steps", 2)
        num_joints = self.asset.num_joints
        self.obs_dim = 3 + 3 + num_joints + num_joints
        self._history = torch.zeros(
            env.num_envs, self.disc_obs_steps, self.obs_dim,
            device=env.device, dtype=torch.float32,
        )
        self._initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def _get_features(self, env: ManagerBasedEnv) -> torch.Tensor:
        """Compute [ang_vel(3), proj_grav(3), dof_pos_rel(N), dof_vel(N)]."""
        ang_vel = self.asset.data.root_link_ang_vel_b
        grav_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
        proj_grav = math_utils.quat_apply_inverse(self.asset.data.root_link_quat_w, grav_w)
        dof_pos_rel = self.asset.data.joint_pos - self.asset.data.default_joint_pos
        dof_vel = self.asset.data.joint_vel
        return torch.cat([ang_vel, proj_grav, dof_pos_rel, dof_vel], dim=-1)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        disc_obs_steps: int = 2,
    ) -> torch.Tensor:
        features = self._get_features(env)
        reset_mask = env.episode_length_buf == 0
        needs_init = reset_mask | ~self._initialized
        if needs_init.any():
            self._history[needs_init] = features[needs_init].unsqueeze(1).expand(-1, self.disc_obs_steps, -1)
            self._initialized[needs_init] = True
        non_reset = ~needs_init
        if non_reset.any():
            self._history[non_reset] = torch.roll(self._history[non_reset], -1, dims=1)
            self._history[non_reset, -1] = features[non_reset]
        return self._history.clone()


class AMPDemoObsTerm(ManagerTermBase):
    """Demonstration motion observation for AMP discriminator.

    Loads motion pkl files at initialization and samples random consecutive
    frames each step. Returns (num_envs, disc_obs_steps, obs_dim).

    The pkl files must contain: 'dof_pos', 'root_rot', 'fps', and optionally
    precomputed 'dof_pos_rel', 'dof_vel', 'root_angle_vel', 'proj_grav'.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.disc_obs_steps: int = cfg.params.get("disc_obs_steps", 2)
        motion_files: list[str] = cfg.params["motion_files"]
        default_dof_pos = cfg.params.get("default_dof_pos", None)

        all_features, lengths = self._load_all(motion_files, default_dof_pos, env.device)
        self._demo_data = all_features
        self._demo_lengths = lengths

        self._demo_pairs = self._build_consecutive_pairs()
        self._num_envs = env.num_envs

    @staticmethod
    def _resolve_path(p: str | Path) -> Path:
        """Resolve motion file path — try absolute, then relative to workspace root."""
        p = Path(p)
        if p.is_absolute() and p.exists():
            return p
        # Workspace root: up from .../tasks/locomotion/mdp/observations.py
        ws_root = Path(__file__).resolve().parents[6]
        candidate = ws_root / p
        if candidate.exists():
            return candidate
        return p

    def _load_all(self, paths: list[str], default_dof_pos, device) -> tuple[torch.Tensor, list[int]]:
        all_feats = []
        lengths = []
        for raw_p in paths:
            p = self._resolve_path(raw_p)
            if not p.exists():
                print(f"[AMPDemoObs] Warning: motion file not found, skipping: {raw_p}")
                continue
            with open(p, "rb") as f:
                motion = pickle.load(f)
            feats = self._extract_features(motion, default_dof_pos)
            all_feats.append(feats)
            lengths.append(feats.shape[0])
        if not all_feats:
            raise FileNotFoundError("No valid motion data files found for AMP demo observations")
        return torch.cat(all_feats, dim=0).to(device), lengths

    @staticmethod
    def _extract_features(motion: dict, default_dof_pos) -> torch.Tensor:
        """Extract [ang_vel, proj_grav, dof_pos_rel, dof_vel] from motion dict.

        Expects preprocessed pkl files (from preprocess_amp_motion.py).
        """
        required = ("root_angle_vel", "proj_grav", "dof_pos_rel", "dof_vel")
        missing = [k for k in required if k not in motion]
        if missing:
            raise KeyError(
                f"Motion pkl is missing precomputed fields {missing}. "
                "Run scripts/preprocess_amp_motion.py on the data first."
            )
        ang_vel = np.asarray(motion["root_angle_vel"], dtype=np.float32)
        proj_grav = np.asarray(motion["proj_grav"], dtype=np.float32)
        dof_pos_rel = np.asarray(motion["dof_pos_rel"], dtype=np.float32)
        dof_vel = np.asarray(motion["dof_vel"], dtype=np.float32)
        return torch.from_numpy(np.concatenate([ang_vel, proj_grav, dof_pos_rel, dof_vel], axis=-1))

    def _build_consecutive_pairs(self) -> torch.Tensor:
        pairs = []
        offset = 0
        for length in self._demo_lengths:
            if length < self.disc_obs_steps:
                offset += length
                continue
            clip = self._demo_data[offset:offset + length]
            for t in range(length - self.disc_obs_steps + 1):
                pairs.append(clip[t:t + self.disc_obs_steps])
            offset += length
        if not pairs:
            raise ValueError("No valid demo pairs could be built")
        return torch.stack(pairs)

    def __call__(
        self,
        env: ManagerBasedEnv,
        disc_obs_steps: int = 2,
        motion_files: list[str] | None = None,
    ) -> torch.Tensor:
        idx = torch.randint(0, self._demo_pairs.shape[0], (self._num_envs,), device=self._demo_pairs.device)
        return self._demo_pairs[idx]
