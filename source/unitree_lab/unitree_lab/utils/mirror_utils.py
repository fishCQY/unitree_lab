"""Observation mirroring utilities for left-right symmetry enforcement.

Used by SymmetryClassifier to create mirrored versions of observation sequences.
"""

from __future__ import annotations

import torch


def mirror_obs_by_keys(
    obs: torch.Tensor,
    keys: list[str],
    joint_mirror_indices: list[int],
    joint_mirror_signs: list[float],
) -> torch.Tensor:
    """Mirror observation tensor based on keys for left-right symmetry.

    Args:
        obs: Observation tensor of shape (batch, obs_dim).
        keys: List of observation keys defining the layout.
        joint_mirror_indices: Indices for joint mirroring (left-right swap).
        joint_mirror_signs: Signs for joint mirroring.

    Returns:
        Mirrored observation tensor.

    Supported keys and their mirror rules:
        - dof_pos, dof_vel: Use joint_mirror_indices and joint_mirror_signs
        - velocity_commands, lin_vel_cmd: [vx, vy, vz/wz] -> [vx, -vy, -vz/wz]
        - base_lin_vel, root_lin_vel: [vx, vy, vz] -> [vx, -vy, vz]
        - base_ang_vel, root_angle_vel: [wx, wy, wz] -> [-wx, wy, -wz]
        - projected_gravity, proj_grav: [gx, gy, gz] -> [gx, -gy, gz]
    """
    obs_mirrored = obs.clone()
    device = obs.device
    n_joints = len(joint_mirror_indices)

    mirror_idx = torch.tensor(joint_mirror_indices, device=device, dtype=torch.long)
    mirror_sgn = torch.tensor(joint_mirror_signs, device=device, dtype=obs.dtype)

    idx = 0
    for key in keys:
        if key in ("lin_vel_cmd", "velocity_commands"):
            obs_mirrored[:, idx:idx + 3] = obs[:, idx:idx + 3] * torch.tensor(
                [1, -1, -1], device=device, dtype=obs.dtype,
            )
            idx += 3
        elif key in ("dof_pos", "dof_vel"):
            obs_mirrored[:, idx:idx + n_joints] = obs[:, idx:idx + n_joints][:, mirror_idx] * mirror_sgn
            idx += n_joints
        elif key in ("root_lin_vel", "base_lin_vel"):
            obs_mirrored[:, idx:idx + 3] = obs[:, idx:idx + 3] * torch.tensor(
                [1, -1, 1], device=device, dtype=obs.dtype,
            )
            idx += 3
        elif key in ("root_angle_vel", "base_ang_vel"):
            obs_mirrored[:, idx:idx + 3] = obs[:, idx:idx + 3] * torch.tensor(
                [-1, 1, -1], device=device, dtype=obs.dtype,
            )
            idx += 3
        elif key in ("proj_grav", "projected_gravity"):
            obs_mirrored[:, idx:idx + 3] = obs[:, idx:idx + 3] * torch.tensor(
                [1, -1, 1], device=device, dtype=obs.dtype,
            )
            idx += 3
        else:
            raise ValueError(f"Unknown key for mirror_obs_by_keys: {key}")

    return obs_mirrored
