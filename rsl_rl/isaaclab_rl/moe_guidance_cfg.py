# Copyright (c) 2024-2026, Light Robotics.
# All rights reserved.

"""Configuration for Mixture of Experts (MoE) guidance in multi-task learning."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from .expert_guidance_cfg import RslRlGuidanceBaseCfg


@configclass
class RslRlMoEGuidanceCfg(RslRlGuidanceBaseCfg):
    """Configuration for MoE (Mixture of Experts) guidance in multi-task learning.

    This configuration enables training a single policy using multiple expert policies,
    where each expert supervises a specific subset of environments based on their group ID.

    Each expert's required observation keys are auto-derived from the JIT model's
    ``actor_obs_keys`` and ``exteroception_key`` attributes at load time.
    """

    expert_policy_paths: list[str] = MISSING
    """List of paths to pretrained expert policy JIT files, one per task group."""

    obs_key_overrides: list[dict[str, str] | None] | None = None
    """Optional per-expert mapping from expert obs key name to env obs key name.

    When provided, must have the same length as ``expert_policy_paths``.
    Each entry is either a dict that maps keys differing between the expert's
    training env and the current mixed env, or ``None`` for experts that need
    no remapping. Keys not present in the dict keep their original name.
    Applied to JIT model attributes at load time.

    Example::

        obs_key_overrides=[
            {"privileged": "locomotion_privileged"},
            {"privileged": "fall_recovery_privileged"},
        ]

    If only the first expert needs remapping::

        obs_key_overrides=[
            {"privileged": "locomotion_privileged"},
            None,
        ]
    """

    env_group_obs_name: str = "env_group"
    """Name of the observation group containing environment group IDs. Default is 'env_group'."""

    # Override default: MoE typically uses pure distillation
    pure_distillation: bool = True
    """Whether to use pure distillation without PPO surrogate loss. Default is True for MoE."""
