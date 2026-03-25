# Copyright (c) 2024-2026, Light Robotics.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class SymmetryClassifierCfg:
    """Configuration for symmetry classifier training."""

    obs_group: str = MISSING
    """Observation group name used for classifier training."""

    num_frames: int = 2
    """Number of observation frames to use for the classifier."""

    cls_hidden_dims: list[int] = [1024, 512]
    """Hidden layer dimensions for the classifier network."""

    activation: str = "relu"
    """Activation function for the classifier network."""

    obs_normalization: bool = False
    """Whether to use empirical normalization for observations."""

    reward_weight: float = MISSING
    """Weight for the symmetry classifier reward contribution."""

    learning_rate: float = 1e-4
    """Learning rate for the classifier optimizer."""

    lr_scale: float | None = None
    """Scale factor for the learning rate relative to policy learning rate.
    If None, uses the absolute learning_rate value."""

    grad_pen_weight: float = 0.0
    """Weight for the gradient penalty term in classifier training.
    Applied to both policy and mirrored inputs. Set to 0.0 to disable."""

    num_learning_epochs: int = 5
    """Number of epochs to train the classifier per policy update."""

    num_mini_batches: int = 4
    """Number of mini-batches for classifier training."""

    mirror_obs_func: str = "cfg.mirror_obs"
    """Path to the mirror observation function. Supports two formats:
    1. "cfg.mirror_obs" - navigate from environment object (e.g., env.cfg.mirror_obs)
    2. "module.path:function_name" - import from module (e.g., "package.module:flip_function")
    The function should take a tensor of observations and return the mirrored version."""
