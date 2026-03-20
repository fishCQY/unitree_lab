"""AMP Motion Data Loader.

This module provides utilities for loading and preprocessing motion capture data
for AMP (Adversarial Motion Priors) training.

Key features:
- Load motion data from pickle files
- Support for conditional AMP with multiple motion types
- Mirror augmentation for symmetric motions
- Feature extraction matching online observations

Usage:
    ```python
    amp_data = load_amp_motion_data(
        motion_files=["walk.pkl", "run.pkl"],
        keys=["dof_pos", "dof_vel", "root_angle_vel", "proj_grav", "key_points_b"],
        device="cuda:0",
    )

    # For conditional AMP
    amp_data = load_conditional_amp_data(
        motion_conditions={
            "walk": ["walk_01.pkl", "walk_02.pkl"],
            "run": ["run_01.pkl", "run_02.pkl"],
        },
        keys=["dof_pos", "dof_vel", "root_angle_vel", "proj_grav", "key_points_b"],
        device="cuda:0",
    )
    ```
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch


@dataclass
class AMPMotionData:
    """Container for AMP motion data.

    Attributes:
        motion_data: Tensor of shape (total_frames, feature_dim) containing
            concatenated motion features.
        motion_lengths: Tensor of shape (num_motions,) containing the length
            of each motion sequence.
        motion_ids: Tensor mapping frame index to motion index.
        condition_ids: Tensor of shape (num_motions,) containing condition
            labels for conditional AMP. None for unconditional AMP.
        num_conditions: Number of unique conditions.
        feature_dim: Dimension of concatenated features.
        keys: List of feature keys in order.
    """

    motion_data: torch.Tensor
    motion_lengths: torch.Tensor
    motion_ids: torch.Tensor
    condition_ids: torch.Tensor | None = None
    num_conditions: int = 1
    feature_dim: int = 0
    keys: list[str] = field(default_factory=list)

    def sample_frames(
        self,
        batch_size: int,
        num_frames: int = 2,
        condition_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample random frame sequences from motion data.

        Args:
            batch_size: Number of sequences to sample.
            num_frames: Number of consecutive frames per sequence.
            condition_mask: Optional boolean mask to filter by condition.

        Returns:
            Tuple of (sequences, conditions) where:
            - sequences: Tensor of shape (batch_size, num_frames, feature_dim)
            - conditions: Tensor of shape (batch_size,) or None
        """
        device = self.motion_data.device

        # Get valid motion indices (those with enough frames)
        valid_mask = self.motion_lengths >= num_frames
        if condition_mask is not None:
            valid_mask = valid_mask & condition_mask

        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) == 0:
            raise ValueError("No valid motions found with enough frames")

        # Sample random motions
        motion_indices = valid_indices[torch.randint(len(valid_indices), (batch_size,), device=device)]

        # Compute cumulative lengths for indexing
        cum_lengths = torch.cat([torch.zeros(1, device=device), self.motion_lengths.cumsum(0)])

        # Sample random start frames within each motion
        max_starts = self.motion_lengths[motion_indices] - num_frames
        start_offsets = (torch.rand(batch_size, device=device) * max_starts.float()).long()
        start_frames = cum_lengths[motion_indices].long() + start_offsets

        # Extract sequences
        frame_indices = start_frames.unsqueeze(1) + torch.arange(num_frames, device=device)
        sequences = self.motion_data[frame_indices]  # (batch_size, num_frames, feature_dim)

        # Get condition labels
        conditions = None
        if self.condition_ids is not None:
            conditions = self.condition_ids[motion_indices]

        return sequences, conditions

    def sample_by_condition(
        self,
        batch_size: int,
        num_frames: int,
        target_conditions: torch.Tensor,
    ) -> torch.Tensor:
        """Sample frames matching specific conditions.

        Args:
            batch_size: Number of sequences to sample.
            num_frames: Number of consecutive frames per sequence.
            target_conditions: Tensor of shape (batch_size,) with target condition indices.

        Returns:
            Tensor of shape (batch_size, num_frames, feature_dim).
        """
        if self.condition_ids is None:
            raise ValueError("Cannot sample by condition: no condition_ids available")

        device = self.motion_data.device
        sequences = torch.zeros(batch_size, num_frames, self.feature_dim, device=device)

        for cond_idx in range(self.num_conditions):
            mask = target_conditions == cond_idx
            if mask.sum() == 0:
                continue

            cond_motion_mask = self.condition_ids == cond_idx
            cond_sequences, _ = self.sample_frames(
                batch_size=mask.sum().item(),
                num_frames=num_frames,
                condition_mask=cond_motion_mask,
            )
            sequences[mask] = cond_sequences

        return sequences


def _flatten_motion_pkl(raw: Any) -> list[dict]:
    """Normalize PKL payload to a flat list of clip dicts.

    Supports three formats:
      - list[dict]:  already flat → return as-is
      - dict with feature keys (e.g. "dof_pos" at top level): single clip → [raw]
      - dict of dicts ({clip_name: clip_data}): nested → list(raw.values())
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        first_val = next(iter(raw.values()), None)
        if isinstance(first_val, dict):
            return list(raw.values())
        return [raw]
    raise TypeError(f"Unexpected PKL payload type: {type(raw)}")


def load_amp_motion_data(
    motion_files: str | Path | Sequence[str | Path],
    keys: Sequence[str],
    device: str | torch.device = "cpu",
    point_indices: Sequence[int] | None = None,
    mirror: bool = False,
    joint_mirror_indices: Sequence[int] | None = None,
    joint_mirror_signs: Sequence[float] | None = None,
) -> AMPMotionData:
    """Load motion data from pickle files.

    Args:
        motion_files: Single file or list of motion data files.
        keys: List of feature keys to extract (e.g., ["dof_pos", "dof_vel"]).
        device: Device to load data to.
        point_indices: Optional indices to select from key_points_b.
        mirror: Whether to apply mirror augmentation.
        joint_mirror_indices: Joint indices for mirroring.
        joint_mirror_signs: Signs for mirrored joints.

    Returns:
        AMPMotionData containing loaded motion data.
    """
    if isinstance(motion_files, (str, Path)):
        motion_files = [motion_files]

    all_features = []
    motion_lengths = []

    for motion_file in motion_files:
        motion_file = Path(motion_file)
        if not motion_file.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_file}")

        with open(motion_file, "rb") as f:
            raw = pickle.load(f)

        for motion in _flatten_motion_pkl(raw):
            features = _extract_features(motion, keys, point_indices)
            all_features.append(features)
            motion_lengths.append(features.shape[0])

            if mirror and joint_mirror_indices is not None:
                mirrored = _mirror_features(
                    features, keys, motion, joint_mirror_indices, joint_mirror_signs
                )
                all_features.append(mirrored)
                motion_lengths.append(mirrored.shape[0])

    # Concatenate all motions
    motion_data = torch.cat(all_features, dim=0).to(device)
    motion_lengths = torch.tensor(motion_lengths, device=device)

    # Create motion ID mapping
    motion_ids = torch.cat([
        torch.full((length,), i, device=device)
        for i, length in enumerate(motion_lengths)
    ])

    return AMPMotionData(
        motion_data=motion_data,
        motion_lengths=motion_lengths,
        motion_ids=motion_ids,
        condition_ids=None,
        num_conditions=1,
        feature_dim=motion_data.shape[-1],
        keys=list(keys),
    )


def load_conditional_amp_data(
    motion_conditions: dict[str, Sequence[str | Path]],
    keys: Sequence[str],
    device: str | torch.device = "cpu",
    point_indices: Sequence[int] | None = None,
    mirror: bool = False,
    joint_mirror_indices: Sequence[int] | None = None,
    joint_mirror_signs: Sequence[float] | None = None,
) -> AMPMotionData:
    """Load motion data with condition labels for conditional AMP.

    Args:
        motion_conditions: Dictionary mapping condition names to motion files.
            Example: {"walk": ["walk1.pkl", "walk2.pkl"], "run": ["run1.pkl"]}
        keys: List of feature keys to extract.
        device: Device to load data to.
        point_indices: Optional indices to select from key_points_b.
        mirror: Whether to apply mirror augmentation.
        joint_mirror_indices: Joint indices for mirroring.
        joint_mirror_signs: Signs for mirrored joints.

    Returns:
        AMPMotionData with condition_ids set.
    """
    condition_names = list(motion_conditions.keys())
    num_conditions = len(condition_names)

    all_features = []
    motion_lengths = []
    condition_ids = []

    for cond_idx, (cond_name, motion_files) in enumerate(motion_conditions.items()):
        if isinstance(motion_files, (str, Path)):
            motion_files = [motion_files]

        for motion_file in motion_files:
            motion_file = Path(motion_file)
            if not motion_file.exists():
                print(f"Warning: Motion file not found, skipping: {motion_file}")
                continue

            with open(motion_file, "rb") as f:
                raw = pickle.load(f)

            for motion in _flatten_motion_pkl(raw):
                features = _extract_features(motion, keys, point_indices)
                all_features.append(features)
                motion_lengths.append(features.shape[0])
                condition_ids.append(cond_idx)

                if mirror and joint_mirror_indices is not None:
                    mirrored = _mirror_features(
                        features, keys, motion, joint_mirror_indices, joint_mirror_signs
                    )
                    all_features.append(mirrored)
                    motion_lengths.append(mirrored.shape[0])
                    condition_ids.append(cond_idx)

    if not all_features:
        raise ValueError("No valid motion files found")

    # Concatenate all motions
    motion_data = torch.cat(all_features, dim=0).to(device)
    motion_lengths = torch.tensor(motion_lengths, device=device)
    condition_ids = torch.tensor(condition_ids, device=device)

    # Create motion ID mapping
    motion_ids = torch.cat([
        torch.full((length,), i, device=device)
        for i, length in enumerate(motion_lengths)
    ])

    return AMPMotionData(
        motion_data=motion_data,
        motion_lengths=motion_lengths,
        motion_ids=motion_ids,
        condition_ids=condition_ids,
        num_conditions=num_conditions,
        feature_dim=motion_data.shape[-1],
        keys=list(keys),
    )


def _extract_features(
    motion: dict[str, Any],
    keys: Sequence[str],
    point_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Extract and concatenate features from motion data.

    Args:
        motion: Dictionary containing motion data.
        keys: Feature keys to extract.
        point_indices: Optional indices for key_points_b selection.

    Returns:
        Tensor of shape (num_frames, feature_dim).
    """
    features = []

    for key in keys:
        if key not in motion:
            raise KeyError(f"Key '{key}' not found in motion data. Available: {list(motion.keys())}")

        data = motion[key]
        if isinstance(data, torch.Tensor):
            feature = data
        else:
            feature = torch.tensor(data, dtype=torch.float32)

        # Handle key_points_b with point selection
        if key == "key_points_b" and point_indices is not None:
            # Reshape from (frames, num_points, 3) to select points
            if feature.dim() == 3:
                feature = feature[:, point_indices, :].reshape(feature.shape[0], -1)
            elif feature.dim() == 2:
                # Already flattened, select by indices
                num_points = feature.shape[1] // 3
                indices = []
                for idx in point_indices:
                    indices.extend([idx * 3, idx * 3 + 1, idx * 3 + 2])
                feature = feature[:, indices]

        # Flatten if needed
        if feature.dim() > 2:
            feature = feature.reshape(feature.shape[0], -1)

        features.append(feature)

    return torch.cat(features, dim=-1)


def _mirror_features(
    features: torch.Tensor,
    keys: Sequence[str],
    motion: dict[str, Any],
    joint_mirror_indices: Sequence[int],
    joint_mirror_signs: Sequence[float] | None = None,
) -> torch.Tensor:
    """Apply mirror augmentation to features.

    This swaps left/right joints and applies sign changes for symmetric motions.

    Args:
        features: Original feature tensor.
        keys: Feature keys.
        motion: Original motion data (for dimension info).
        joint_mirror_indices: Indices for joint mirroring.
        joint_mirror_signs: Signs for mirrored joints.

    Returns:
        Mirrored feature tensor.
    """
    mirrored = features.clone()

    # Build feature offset map
    offset = 0
    for key in keys:
        if key not in motion:
            continue

        data = motion[key]
        if isinstance(data, torch.Tensor):
            dim = data.shape[-1] if data.dim() == 2 else data[0].numel()
        else:
            dim = len(data[0]) if hasattr(data[0], "__len__") else 1

        if key in ["dof_pos", "dof_vel"]:
            # Mirror joint positions/velocities
            joint_dim = dim
            mirrored_joints = mirrored[:, offset:offset + joint_dim].clone()

            # Apply index permutation
            indices = torch.tensor(joint_mirror_indices)
            mirrored[:, offset:offset + joint_dim] = mirrored_joints[:, indices]

            # Apply sign changes
            if joint_mirror_signs is not None:
                signs = torch.tensor(joint_mirror_signs, dtype=mirrored.dtype)
                mirrored[:, offset:offset + joint_dim] *= signs

        elif key == "root_angle_vel":
            # Mirror angular velocity: negate y and z components
            mirrored[:, offset + 1] *= -1  # wy
            mirrored[:, offset + 2] *= -1  # wz

        elif key == "proj_grav":
            # Mirror projected gravity: negate y component
            mirrored[:, offset + 1] *= -1

        elif key == "key_points_b":
            # Mirror body positions: negate y component for each point
            num_points = dim // 3
            for i in range(num_points):
                y_idx = offset + i * 3 + 1
                mirrored[:, y_idx] *= -1

        offset += dim

    return mirrored


def create_mirror_config(
    left_joint_names: Sequence[str],
    right_joint_names: Sequence[str],
    all_joint_names: Sequence[str],
) -> tuple[list[int], list[float]]:
    """Create mirror configuration from joint names.

    Args:
        left_joint_names: Names of left-side joints.
        right_joint_names: Names of right-side joints (same order as left).
        all_joint_names: All joint names in order.

    Returns:
        Tuple of (mirror_indices, mirror_signs).
    """
    num_joints = len(all_joint_names)
    mirror_indices = list(range(num_joints))
    mirror_signs = [1.0] * num_joints

    # Create left-right swap mapping
    for left_name, right_name in zip(left_joint_names, right_joint_names):
        if left_name in all_joint_names and right_name in all_joint_names:
            left_idx = all_joint_names.index(left_name)
            right_idx = all_joint_names.index(right_name)

            mirror_indices[left_idx] = right_idx
            mirror_indices[right_idx] = left_idx

            # Sign changes for certain joint types
            if any(x in left_name.lower() for x in ["roll", "yaw"]):
                mirror_signs[left_idx] = -1.0
                mirror_signs[right_idx] = -1.0

    return mirror_indices, mirror_signs
