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
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
import numpy as np
import re
import os
import glob

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import MotionTrackingCommandCfg


class MotionTrackingCommand(CommandTerm):

    cfg: MotionTrackingCommandCfg

    def __init__(self, cfg: MotionTrackingCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # load the dataset
        self._load_dataset()

        # setup sampling weights
        self._setup_sampling_weights()

        # initialize the current state tensors
        self.index_offset = self._sample_indices(env.num_envs)
        self.current_body_positions_w = torch.zeros((env.num_envs, len(self.body_names), 3), device=env.device, dtype=torch.float32)
        self.current_body_positions = torch.zeros((env.num_envs, len(self.body_names), 3), device=env.device, dtype=torch.float32)
        self.current_body_orientations = torch.zeros((env.num_envs, len(self.body_names), 4), device=env.device, dtype=torch.float32)
        self.current_body_velocities = torch.zeros((env.num_envs, len(self.body_names), 3), device=env.device, dtype=torch.float32)
        self.current_body_angle_velocities = torch.zeros((env.num_envs, len(self.body_names), 3), device=env.device, dtype=torch.float32)
        self.current_joint_pos = torch.zeros((env.num_envs, len(self.joint_names)), device=env.device, dtype=torch.float32)
        self.current_joint_vel = torch.zeros((env.num_envs, len(self.joint_names)), device=env.device, dtype=torch.float32)
        self.current_key_points_w = torch.zeros((env.num_envs, len(self.body_names), 3, 3), device=env.device, dtype=torch.float32)
        self.current_key_points = torch.zeros((env.num_envs, len(self.body_names), 3, 3), device=env.device, dtype=torch.float32)

    """
    Properties
    """

    def _get_command_indices(self) -> torch.Tensor:
        start_indices = self.command_counter + self.index_offset - 1
        offsets = torch.arange(self.cfg.sequence_len, device=start_indices.device)
        indices = start_indices.unsqueeze(1) + offsets
        return indices % self.dataset_length

    @property
    def command(self) -> torch.Tensor:
        indices = self._get_command_indices()
        return torch.cat([
            # self.keypoints[:, self.tracking_body_ids][indices].flatten(start_dim=1),
            self.repr_6d[:, self.root_link_ids][indices].flatten(start_dim=1),
            # self.body_pos_xyz[:,self.tracking_body_ids][indices].flatten(start_dim=1),
            # self.body_quat_wxyz[:,self.tracking_body_ids][indices].flatten(start_dim=1),
            self.body_linear_velocities[:, self.root_link_ids][indices].flatten(start_dim=1),
            self.body_angular_velocities[:, self.root_link_ids][indices].flatten(start_dim=1),
            self.joint_pos[:, self.joint_indices][indices].flatten(start_dim=1),
            self.joint_vel[:, self.joint_indices][indices].flatten(start_dim=1),
        ], dim=-1)

    def command_index(self) -> torch.Tensor:
        return self._get_command_indices()

    def _get_body_indices(self, body_name) -> torch.Tensor:
        _, body_indices = find_pattern_matches(self.body_names, body_name)
        return body_indices

    def _get_joint_indices(self, joint_name) -> torch.Tensor:
        _, joint_indices = find_pattern_matches(self.joint_names, joint_name)
        return joint_indices

    def tracking_body_pos_w(self, body_name) -> torch.Tensor:
        body_indices = self._get_body_indices(body_name)
        return self.current_body_positions_w[:, body_indices, :]

    def tracking_body_pos(self, body_name) -> torch.Tensor:
        body_indices = self._get_body_indices(body_name)
        return self.current_body_positions[:, body_indices, :]

    def tracking_body_quat(self, body_name) -> torch.Tensor:
        body_indices = self._get_body_indices(body_name)
        return self.current_body_orientations[:, body_indices, :]

    def tracking_body_vel(self, body_name) -> torch.Tensor:
        body_indices = self._get_body_indices(body_name)
        return self.current_body_velocities[:, body_indices, :]

    def tracking_body_ang_vel(self, body_name) -> torch.Tensor:
        body_indices = self._get_body_indices(body_name)
        return self.current_body_angle_velocities[:, body_indices, :]

    def tracking_joint_pos(self, joint_name) -> torch.Tensor:
        joint_indices = self._get_joint_indices(joint_name)
        return self.current_joint_pos[:, joint_indices]

    def tracking_joint_vel(self, joint_name) -> torch.Tensor:
        joint_indices = self._get_joint_indices(joint_name)
        return self.current_joint_vel[:, joint_indices]

    def tracking_key_points_w(self, body_name) -> torch.Tensor:
        body_indices = self._get_body_indices(body_name)
        return self.current_key_points_w[:, body_indices, :, :]

    def tracking_key_points(self, body_name) -> torch.Tensor:
        body_indices = self._get_body_indices(body_name)
        return self.current_key_points[:, body_indices, :, :]

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        # reset the index offset with weighted sampling
        self.index_offset[env_ids] = self._sample_indices(len(env_ids) if not isinstance(env_ids, (list, tuple)) else self._env.num_envs)
        # self.index_offset[env_ids] = 0
        # resample the command
        extras = super().reset(env_ids=env_ids)
        # set the initial state from the dataset
        self._data_replay(env_ids=env_ids)
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        # get the current index
        index = self.command_counter[env_ids] + self.index_offset[env_ids] - 1
        index = index % self.dataset_length  # wrap around if the index exceeds the dataset length

        if self.cfg.use_world_frame:
            self.current_body_positions_w[env_ids] = self.body_pos_xyz_w[index] + self._env.scene.env_origins[env_ids].unsqueeze(1)
            self.current_key_points_w[env_ids] = self.keypoints_w[index] + self._env.scene.env_origins[env_ids].unsqueeze(1).unsqueeze(1)
        else:
            root_link_pos_w = self.robot.data.root_link_pos_w[env_ids, :].clone()
            root_link_pos_w[:, 2] = self.body_pos_xyz_w[index, self.root_link_ids, 2].clone()  # keep the z position of the root link
            root_yaw_quat = math_utils.yaw_quat(self.body_quat_wxyz[index, self.root_link_ids, :])
            self.current_body_positions_w[env_ids] = math_utils.quat_apply(
                root_yaw_quat.unsqueeze(1).expand(-1, self.body_pos_xyz.shape[1], -1),
                self.body_pos_xyz[index]
            ) + root_link_pos_w.unsqueeze(1)

            self.current_key_points_w[env_ids] = math_utils.quat_apply(
                root_yaw_quat.unsqueeze(1).unsqueeze(1).expand(-1, self.keypoints.shape[1], self.keypoints.shape[2], -1),
                self.keypoints[index]
            ) + root_link_pos_w.unsqueeze(1).unsqueeze(1)

        self.current_body_positions[env_ids] = self.body_pos_xyz[index]
        self.current_body_orientations[env_ids] = self.body_quat_wxyz[index]
        self.current_body_velocities[env_ids] = self.body_linear_velocities[index]
        self.current_body_angle_velocities[env_ids] = self.body_angular_velocities[index]
        self.current_joint_pos[env_ids] = self.joint_pos[index]
        self.current_joint_vel[env_ids] = self.joint_vel[index]
        self.current_key_points[env_ids] = self.keypoints[index]

    def _update_command(self):
        if self.cfg.replay_dataset:
            self._data_replay()

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "tracking_body_frame_visualizer"):
                self.tracking_body_frame_visualizer = VisualizationMarkers(self.cfg.tracking_body_frame_visualizer)
                self.key_points_visualizer = VisualizationMarkers(self.cfg.key_points_visualizer)
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.tracking_body_frame_visualizer.set_visibility(True)
            self.key_points_visualizer.set_visibility(False)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "tracking_body_frame_visualizer"):
                self.tracking_body_frame_visualizer.set_visibility(False)
                self.key_points_visualizer.set_visibility(False)
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return

        self.tracking_body_frame_visualizer.visualize(
            translations=self.current_body_positions_w[:, self.tracking_body_ids, :].reshape(-1, 3),
            orientations=self.current_body_orientations[:, self.tracking_body_ids, :].reshape(-1, 4),
        )
        self.key_points_visualizer.visualize(
            translations=self.current_key_points_w[:, self.tracking_body_ids, :].reshape(-1, 3),
        )

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 1.0
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.current_body_velocities[:, self.root_link_ids, :2].squeeze(1)
        )
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_link_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _load_dataset(self):
        # load the dataset from the path
        if isinstance(self.cfg.dataset_path, str):
            self.cfg.dataset_path = [self.cfg.dataset_path]
        # find all files in the dataset path
        self.cfg.dataset_path = find_all_files(self.cfg.dataset_path, exclude_prefix=self.cfg.exclude_prefix)

        body_pos_xyz_w_list = []
        body_quat_wxyz_list = []
        body_linear_velocities_list = []
        body_angular_velocities_list = []
        joint_pos_list = []
        joint_vel_list = []

        for dataset_path in self.cfg.dataset_path:
            # load the dataset
            dataset = np.load(dataset_path)

            # get the robot data
            if not hasattr(self, "dof_names") and not hasattr(self, "body_names"):
                self.joint_names = dataset["dof_names"].astype('U').tolist()
                self.body_names = dataset["body_names"].astype('U').tolist()
                _, self.body_indices = find_pattern_matches(self.body_names, self.robot.data.body_names)
                _, self.joint_indices = find_pattern_matches(self.joint_names, self.robot.data.joint_names)
                _, self.joint_indices_inv = find_pattern_matches(self.robot.data.joint_names, self.joint_names)
                _, self.root_link_ids = find_pattern_matches(self.body_names, self.cfg.root_link_name)
                # get the tracking body names and ids
                if self.cfg.tracking_body_names is not None:
                    _, self.tracking_body_ids = find_pattern_matches(self.body_names, self.cfg.tracking_body_names)
                else:
                    _, self.tracking_body_ids = find_pattern_matches(self.body_names, ".*")

            body_pos_xyz_w_list.append(torch.from_numpy(dataset["body_positions"]).to(self._env.device).to(torch.float32))
            body_quat_wxyz_list.append(torch.from_numpy(dataset["body_rotations_wxyz"]).to(self._env.device).to(torch.float32))
            body_linear_velocities_list.append(torch.from_numpy(dataset["body_linear_velocities"]).to(self._env.device).to(torch.float32))
            body_angular_velocities_list.append(torch.from_numpy(dataset["body_angular_velocities"]).to(self._env.device).to(torch.float32))
            joint_pos_list.append(torch.from_numpy(dataset["dof_positions"]).to(self._env.device).to(torch.float32))
            joint_vel_list.append(torch.from_numpy(dataset["dof_velocities"]).to(self._env.device).to(torch.float32))

        self.body_pos_xyz_w = torch.cat(body_pos_xyz_w_list, dim=0)
        self.body_quat_wxyz = torch.cat(body_quat_wxyz_list, dim=0)
        self.body_linear_velocities = torch.cat(body_linear_velocities_list, dim=0)
        self.body_angular_velocities = torch.cat(body_angular_velocities_list, dim=0)
        self.joint_pos = torch.cat(joint_pos_list, dim=0)
        self.joint_vel = torch.cat(joint_vel_list, dim=0)

        self.dataset_length = self.body_pos_xyz_w.shape[0]
        self.body_pos_xyz = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.body_quat_wxyz[:, self.root_link_ids, :]).expand(-1, self.body_pos_xyz_w.shape[1], -1),
            self.body_pos_xyz_w - self.body_pos_xyz_w[:, self.root_link_ids, :]
        )
        keypoints_b = self.cfg.side_length * torch.eye(3, device=self.body_pos_xyz_w.device, dtype=self.body_pos_xyz_w.dtype)
        keypoints_b = keypoints_b.unsqueeze(0).unsqueeze(0).expand(self.body_pos_xyz_w.shape[0], self.body_quat_wxyz.shape[1], -1, -1)
        self.keypoints_w = math_utils.quat_apply(self.body_quat_wxyz.unsqueeze(2).expand(-1, -1, keypoints_b.shape[2], -1), keypoints_b) + self.body_pos_xyz_w.unsqueeze(2)
        self.keypoints = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.body_quat_wxyz[:, self.root_link_ids, :].unsqueeze(2)).expand(-1, self.body_pos_xyz_w.shape[1], self.keypoints_w.shape[2], -1),
            self.keypoints_w - self.body_pos_xyz_w[:, self.root_link_ids, :].unsqueeze(2)
        )
        body_rot_matrix = math_utils.matrix_from_quat(self.body_quat_wxyz)
        repr_6d: torch.Tensor = body_rot_matrix[..., :, :2]
        self.repr_6d = repr_6d.reshape(*self.body_quat_wxyz.shape[:-1], 6)

    def _data_replay(self, env_ids: Sequence[int] | None = None):
        env_idx = slice(None) if env_ids is None else env_ids

        positions = self.current_body_positions_w[env_idx, self.root_link_ids].reshape(-1, 3)
        orientations = self.current_body_orientations[env_idx, self.root_link_ids].reshape(-1, 4)
        velocities_b = self.current_body_velocities[env_idx, self.root_link_ids].reshape(-1, 3)
        angle_velocities_b = self.current_body_angle_velocities[env_idx, self.root_link_ids].reshape(-1, 3)
        # convert the velocities and angle_velocities to the format expected by the robot
        velocities_w = math_utils.quat_apply(orientations, velocities_b)
        angle_velocities_w = math_utils.quat_apply(orientations, angle_velocities_b)
        joint_pos = self.current_joint_pos[env_idx]
        joint_vel = self.current_joint_vel[env_idx]
        # set into the physics simulation
        root_link_state = torch.cat([positions, orientations, velocities_w, angle_velocities_w], dim=-1)
        self.robot.write_root_link_state_to_sim(root_link_state, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self.joint_indices_inv, env_ids=env_ids)

    def _setup_sampling_weights(self):
        """Setup sampling weights for the dataset."""
        weights = torch.ones(self.dataset_length, device=self._env.device, dtype=torch.float32)
        # Use the update method to set the weights
        self.update_sampling_weights(weights)

    def _sample_indices(self, num_samples: int) -> torch.Tensor:
        """Sample indices from the dataset using the configured sampling strategy."""
        # Always use weighted sampling (weights default to uniform if not specified)
        indices = torch.multinomial(
            self.sampling_probabilities,
            num_samples,
            replacement=True
        )
        return indices

    def update_sampling_weights(self, new_weights: list[float] | torch.Tensor):
        """
        Update the sampling weights during runtime.

        Args:
            new_weights: New weights for sampling. Should have the same length as dataset_length.
        """
        if isinstance(new_weights, list):
            if len(new_weights) != self.dataset_length:
                raise ValueError(
                    f"Length of new_weights ({len(new_weights)}) "
                    f"must match dataset length ({self.dataset_length})"
                )
            weights = torch.tensor(new_weights, device=self._env.device, dtype=torch.float32)
        else:
            if new_weights.shape[0] != self.dataset_length:
                raise ValueError(
                    f"Length of new_weights ({new_weights.shape[0]}) "
                    f"must match dataset length ({self.dataset_length})"
                )
            weights = new_weights.to(device=self._env.device, dtype=torch.float32)

        # Ensure weights are positive
        self.weights = torch.clamp(weights, min=1e-8)
        # Normalize to sum to 1
        self.sampling_probabilities = weights / weights.sum()

    def get_current_weights(self) -> torch.Tensor:
        """Get the current sampling weights (probabilities)."""
        return self.weights.clone()


def find_pattern_matches(string_list, patterns):
    matches_names = []
    matches_ids = []
    if isinstance(patterns, str):
        pattern_list = [patterns]
    elif isinstance(patterns, list):
        pattern_list = patterns
    else:
        raise TypeError("patterns must be a string or a list of strings")

    for pattern_str in pattern_list:
        for index, name in enumerate(string_list):
            if re.fullmatch(pattern_str, name):
                matches_ids.append(index)
                matches_names.append(name)
    return matches_names, matches_ids


def find_all_files(path_list: list[str], exclude_prefix: str = None) -> list[str]:
    found_files = set()
    for item in path_list:
        matching_paths = glob.glob(item)
        for path in matching_paths:
            if os.path.isfile(path):
                if exclude_prefix is None or not os.path.basename(path).startswith(exclude_prefix):
                    found_files.add(os.path.abspath(path))
    return list(found_files)
