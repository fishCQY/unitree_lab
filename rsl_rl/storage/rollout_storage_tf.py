# Copyright (c) 2024-2026, Light Robotics.
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause

"""Rollout storage extension for Transformer-based RL policies.

This module provides a ``RolloutStorageTF`` class that adds Transformer-specific
mini-batch generation and cross-rollout prefix buffering on top of the standard
:class:`RolloutStorage`.
"""

from __future__ import annotations

from collections.abc import Generator

import torch
from tensordict import TensorDict

from rsl_rl.storage.rollout_storage import RolloutStorage


class RolloutStorageTF(RolloutStorage):
    """Rollout storage with Transformer-specific sliding-window mini-batch generation.

    Adds two capabilities on top of :class:`RolloutStorage`:

    1. **Prefix buffer** (``save_transformer_prefix``): saves the tail of the current
       rollout so the next rollout can reconstruct context windows that span the
       rollout boundary.
    2. **Transformer mini-batch generator** (``transformer_mini_batch_generator``):
       reconstructs sliding-window observations from the flat single-frame storage,
       respecting episode boundaries and only windowing the observation groups that
       the Transformer policy actually needs.
    """

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
    ) -> None:
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)

        # Prefix buffer for cross-rollout history
        self.obs_prefix: TensorDict | None = None
        self.dones_prefix: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Transformer-specific methods
    # ------------------------------------------------------------------

    def save_transformer_prefix(self, context_len: int, groups_to_window: set[str] | None = None) -> None:
        """Save tail of current rollout as prefix for next rollout's window building.

        Only saves groups that need windowing to minimize memory usage.

        Args:
            context_len: The context window length of the Transformer.
            groups_to_window: Set of obs group names to window. If None, all groups are saved.
        """
        prefix_len = min(context_len - 1, self.num_transitions_per_env)
        if prefix_len <= 0:
            return
        keys = groups_to_window if groups_to_window is not None else set(self.observations.keys())
        self.obs_prefix = TensorDict(
            {key: self.observations[key][-prefix_len:].clone() for key in keys if key in self.observations.keys()},
            batch_size=[prefix_len, self.num_envs],
            device=self.device,
        )
        self.dones_prefix = self.dones[-prefix_len:].clone()

    def transformer_mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int,
        context_len: int,
        groups_to_window: set[str] | None = None,
    ) -> Generator:
        """Mini-batch generator that reconstructs sliding-window observations.

        Only groups in ``groups_to_window`` are expanded to ``(batch, context_len, dim)``.
        All other groups are kept as single-frame ``(batch, dim)``, saving significant GPU memory.

        Args:
            num_mini_batches: Number of mini-batches per epoch.
            num_epochs: Number of epochs.
            context_len: The context window length of the Transformer.
            groups_to_window: Set of obs group names to window. If None, all groups are windowed.
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        T = self.num_transitions_per_env
        N = self.num_envs
        all_keys = set(self.observations.keys())
        window_keys = all_keys if groups_to_window is None else (groups_to_window & all_keys)
        flat_keys = all_keys - window_keys

        # 1. Prepend prefix from previous rollout (if available) — only for windowed groups
        if self.obs_prefix is not None:
            prefix_len = self.obs_prefix.batch_size[0]
            extended_obs = TensorDict(
                {
                    k: torch.cat([self.obs_prefix[k], self.observations[k]], dim=0)
                    for k in window_keys
                    if k in self.obs_prefix.keys()
                },
                batch_size=[prefix_len + T, N],
                device=self.device,
            )
            extended_dones = torch.cat([self.dones_prefix, self.dones], dim=0).squeeze(-1)
        else:
            prefix_len = 0
            extended_obs = TensorDict(
                {k: self.observations[k] for k in window_keys},
                batch_size=[T, N],
                device=self.device,
            )
            extended_dones = self.dones.squeeze(-1)

        ET = prefix_len + T  # extended time dimension

        # 2. Compute episode_start on extended timeline
        episode_start = torch.zeros(ET, N, dtype=torch.long, device=self.device)
        for t in range(1, ET):
            episode_start[t] = torch.where(
                extended_dones[t - 1].bool(), torch.tensor(t, device=self.device), episode_start[t - 1]
            )

        # 3. Build windowed obs for ORIGINAL rollout steps only
        t_indices = torch.arange(T, device=self.device)
        t_ext = t_indices + prefix_len
        ep_start_for_t = episode_start[t_ext]  # (T, N)

        result_obs = {}

        # 3a. Window the groups that need it
        for key in window_keys:
            ext_obs = extended_obs[key]  # (ET, N, D)
            obs_dim = ext_obs.shape[-1]
            windows = torch.zeros(T, N, context_len, obs_dim, device=self.device)
            for k in range(context_len):
                src_t = t_ext - (context_len - 1 - k)
                src_t_clamped = src_t.clamp(min=0)
                valid = (src_t >= 0).unsqueeze(1) & (src_t_clamped.unsqueeze(1) >= ep_start_for_t)
                gathered = ext_obs[src_t_clamped]
                windows[:, :, k] = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
            result_obs[key] = windows

        # 3b. Keep flat groups as single-frame (no windowing)
        for key in flat_keys:
            result_obs[key] = self.observations[key]  # (T, N, D) — single frame

        # 4. Build window validity masks
        window_masks = torch.zeros(T, N, context_len, dtype=torch.bool, device=self.device)
        for k in range(context_len):
            src_t = t_ext - (context_len - 1 - k)
            src_t_clamped = src_t.clamp(min=0)
            valid = (src_t >= 0).unsqueeze(1) & (src_t_clamped.unsqueeze(1) >= ep_start_for_t)
            window_masks[:, :, k] = valid

        result_obs["__window_mask__"] = window_masks
        result_obs = TensorDict(result_obs, batch_size=[T, N], device=self.device)

        # Free extended_obs to release memory before flattening
        del extended_obs

        # 5. Flatten + shuffle
        batch_size = T * N
        observations = result_obs.flatten(0, 1)
        del result_obs  # free the pre-flatten tensor

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        expert_actions = self.expert_actions.flatten(0, 1) if self.expert_actions is not None else None

        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                expert_actions_batch = expert_actions[batch_idx] if expert_actions is not None else None

                yield (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (None, None),
                    None,
                    expert_actions_batch,
                )
