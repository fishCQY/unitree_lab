# Copyright (c) 2024-2026, Light Robotics.
# All rights reserved.

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class RslRlGuidanceBaseCfg:
    """Base configuration for guidance-based learning."""

    pure_distillation: bool = False
    """Whether to use pure distillation without PPO surrogate loss. Default is False."""

    guidance_weight: float = 1.0
    """Initial weight for the guidance loss. Default is 1.0."""

    error_type: Literal["normal", "std_weighted"] = "normal"
    """Error type for guidance. 'normal' uses standard error, 'std_weighted' weights by inverse std."""

    loss_fn: Literal["mse", "huber"] = "mse"
    """Loss function for guidance. Options: 'mse' (mean squared error) or 'huber' (Huber loss)."""

    weight_decay_func: str | callable | None = None
    """Function to decay the guidance weight. If None, the weight is not decayed."""

    action_blend_warmup_iters: int | None = None
    """Number of iterations to use pure expert actions before starting blend weight decay.
    If None, action blending is disabled and student actions are used directly."""

    action_blend_decay_iters: int = 0
    """Number of iterations to linearly decay the blend weight from 1.0 to 0.0 after warmup.
    If 0, the blend weight drops immediately to 0 after warmup ends."""


@configclass
class RslRlExpertGuidanceCfg(RslRlGuidanceBaseCfg):
    """Configuration for single-expert guidance imitation learning.

    The expert's required observation keys are auto-derived from the JIT model's
    ``actor_obs_keys`` and ``exteroception_key`` attributes at load time.
    """

    expert_policy_path: str = MISSING
    """Path to the pretrained expert policy JIT file."""
