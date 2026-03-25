# Copyright (c) 2024-2026, Light Robotics.
# All rights reserved.

"""Proximal Policy Optimization with Mixture of Experts (PPO-MoE) algorithm.

This module extends the standard PPO algorithm to support multiple expert policies
for different environment groups. Each group is supervised by its corresponding expert.
"""

from __future__ import annotations

import os

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable


class PPO_MoE(PPO):
    """PPO with Mixture of Experts (MoE) for multi-task learning.

    This algorithm extends PPO to support multiple expert policies that supervise
    different subsets of environments based on their group assignment.

    Key features:
    - Multiple expert policies, one for each task group
    - Expert selection based on env_group observation
    - Pure distillation mode for imitation learning

    The main difference from standard PPO expert guidance is in the act() method,
    which selects the appropriate expert based on env_group. The update() method
    reuses the parent class implementation since expert_actions are already
    pre-computed per-sample during act().
    """

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        adv_filtering_ratio: float = 0.0,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # MoE expert guidance parameters
        moe_guidance_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        # Initialize MoE-specific attributes before parent __init__
        self.expert_policies: list[torch.jit.ScriptModule] = []
        self.expert_obs_keys_per_group: list[list[str]] = []
        self.num_experts = 0
        self.env_group_obs_name = "env_group"

        # Per-group action error tracking for monitoring (sample-weighted MSE)
        self._group_error_sums: dict[int, torch.Tensor] = {}
        self._group_error_counts: dict[int, int] = {}

        # Initialize parent without expert_guidance_cfg (we handle it ourselves)
        super().__init__(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            adv_filtering_ratio=adv_filtering_ratio,
            device=device,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            expert_guidance_cfg=None,  # Don't pass to parent, we handle it
            multi_gpu_cfg=multi_gpu_cfg,
        )

        # Initialize MoE expert guidance if provided
        if moe_guidance_cfg is not None:
            self._initialize_moe_experts(moe_guidance_cfg)

    def _initialize_moe_experts(self, moe_guidance_cfg: dict):
        """Initialize multiple expert policies for MoE guidance.

        Args:
            moe_guidance_cfg: Configuration dict containing:
                - expert_policy_paths: List of paths to expert policy JIT files
                - guidance_weight: Initial weight for guidance loss
                - pure_distillation: Whether to use pure distillation mode
                - error_type: Error computation type ('normal' or 'std_weighted')
                - loss_fn: Loss function ('mse' or 'huber')
                - env_group_obs_name: Name of the env_group observation (default: 'env_group')
                - obs_key_overrides: Optional per-expert mapping from expert obs key to env obs key
        """
        self.env_group_obs_name = moe_guidance_cfg.get("env_group_obs_name", "env_group")

        # Load expert policies and auto-derive obs keys from each JIT model
        expert_policy_paths = moe_guidance_cfg["expert_policy_paths"]
        self.num_experts = len(expert_policy_paths)
        self.expert_obs_keys_per_group = []

        obs_key_overrides = moe_guidance_cfg.get("obs_key_overrides")
        if obs_key_overrides is not None and len(obs_key_overrides) != len(expert_policy_paths):
            raise ValueError(
                f"obs_key_overrides length ({len(obs_key_overrides)}) must match "
                f"expert_policy_paths length ({len(expert_policy_paths)})"
            )

        for i, path in enumerate(expert_policy_paths):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Expert policy JIT file not found: {path}")
            expert_policy = torch.jit.load(path, map_location=self.device)
            expert_policy.to(self.device)
            expert_policy.eval()
            self.expert_policies.append(expert_policy)

            # Auto-derive obs keys from JIT model
            if not hasattr(expert_policy, "actor_obs_keys"):
                raise RuntimeError(
                    f"Expert JIT model at '{path}' is missing 'actor_obs_keys' attribute. "
                    "Please re-export the expert policy with the latest exporter."
                )

            # Apply obs key overrides to JIT model attributes before auto-derive
            key_map = (obs_key_overrides[i] or {}) if obs_key_overrides else {}
            if key_map:
                original_actor_keys = list(expert_policy.actor_obs_keys)
                expert_policy.actor_obs_keys = [key_map.get(k, k) for k in original_actor_keys]
                original_ext_key = str(expert_policy.exteroception_key)
                if original_ext_key and original_ext_key in key_map:
                    expert_policy.exteroception_key = key_map[original_ext_key]
                print(
                    f"  obs_key_overrides applied for expert {i}: "
                    f"actor_obs_keys {original_actor_keys} -> {list(expert_policy.actor_obs_keys)}"
                )

            obs_keys: list[str] = list(expert_policy.actor_obs_keys)
            ext_key = str(expert_policy.exteroception_key)
            if ext_key:
                obs_keys.append(ext_key)
            self.expert_obs_keys_per_group.append(obs_keys)
            print(f"Loaded expert policy {i} from: {path}, obs_keys={obs_keys}")

        # Set up parent class expert_guidance to reuse update() logic
        self.expert_guidance = moe_guidance_cfg
        self.guidance_weight = moe_guidance_cfg.get("guidance_weight", 1.0)
        self.guidance_error_type = moe_guidance_cfg.get("error_type", "normal")
        self.guidance_loss_fn = moe_guidance_cfg.get("loss_fn", "mse")

        # Validate configurations
        if self.guidance_error_type not in ["normal", "std_weighted"]:
            raise ValueError(f"Unknown error type: {self.guidance_error_type}. Supported: ['normal', 'std_weighted']")
        if self.guidance_loss_fn not in ["mse", "huber"]:
            raise ValueError(f"Unknown loss function: {self.guidance_loss_fn}. Supported: ['mse', 'huber']")

        # Set up weight decay function
        weight_decay_func = moe_guidance_cfg.get("weight_decay_func")
        if weight_decay_func is not None:
            if isinstance(weight_decay_func, str):
                self.weight_decay_func = string_to_callable(weight_decay_func)
            elif callable(weight_decay_func):
                self.weight_decay_func = weight_decay_func
            else:
                raise ValueError(f"Invalid weight_decay_func: {weight_decay_func}")
            print(f"Using custom weight decay function: {weight_decay_func}")
        else:
            self.weight_decay_func = None

        # Action blending (DAgger-style): blend student and expert actions for env interaction
        self.action_blend_warmup_iters: int | None = moe_guidance_cfg.get("action_blend_warmup_iters")
        self.action_blend_decay_iters: int = moe_guidance_cfg.get("action_blend_decay_iters", 0)
        self.action_blend_weight: float = 1.0 if self.action_blend_warmup_iters is not None else 0.0

        print(
            f"MoE Expert guidance config - Num experts: {self.num_experts}, "
            f"Weight: {self.guidance_weight}, Error type: {self.guidance_error_type}, "
            f"Loss fn: {self.guidance_loss_fn}"
        )
        if self.action_blend_warmup_iters is not None:
            print(
                f"Action blending enabled - Warmup: {self.action_blend_warmup_iters} iters,"
                f" Decay: {self.action_blend_decay_iters} iters"
            )

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        """Initialize rollout storage and expert policy hidden states."""
        # Create rollout storage (same as parent)
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

        # Initialize hidden states for recurrent expert policies
        for expert_policy in self.expert_policies:
            if expert_policy.is_recurrent:
                layers, _, hidden_dim = expert_policy.hidden_state.shape
                expert_policy.hidden_state = torch.zeros(layers, num_envs, hidden_dim, device=self.device)
                if hasattr(expert_policy, "cell_state"):
                    expert_policy.cell_state = torch.zeros(layers, num_envs, hidden_dim, device=self.device)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Compute actions and store expert actions for each environment group.

        For MoE, we compute expert actions based on each environment's group assignment.
        The expert_actions are stored in transition and will be used by the parent
        class update() method for guidance loss computation.
        """
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        # Compute student policy actions
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs

        # Compute MoE expert actions based on env_group
        if self.expert_guidance is not None:
            num_envs = obs.batch_size[0]
            action_dim = self.transition.actions.shape[-1]

            # Get env_group from observations
            env_group = obs[self.env_group_obs_name].squeeze(-1).long()  # Shape: (num_envs,)

            # Initialize expert actions tensor
            expert_actions = torch.zeros(num_envs, action_dim, device=self.device)

            # Compute expert actions for each group
            for group_id in range(self.num_experts):
                group_mask = env_group == group_id
                if not group_mask.any():
                    continue

                # Build Dict[str, Tensor] for this expert from auto-derived keys
                obs_dict: dict[str, torch.Tensor] = {key: obs[key] for key in self.expert_obs_keys_per_group[group_id]}

                # Get expert actions
                with torch.no_grad():
                    try:
                        group_expert_actions = self.expert_policies[group_id](obs_dict)
                    except RuntimeError as e:
                        raise RuntimeError(
                            f"Expert {group_id} failed. obs_keys={self.expert_obs_keys_per_group[group_id]}."
                            f" Original error: {e}"
                        ) from e

                # Get expert actions for environments in this group
                group_expert_actions_masked = group_expert_actions[group_mask]

                # Store expert actions
                expert_actions[group_mask] = group_expert_actions_masked

                # Compute per-group error for monitoring (student action_mean vs expert action)
                # Use a sample-weighted mean so the metric is stable across varying per-step group sizes.
                group_student_actions = self.transition.action_mean[group_mask]
                diff = group_student_actions - group_expert_actions_masked
                group_sse = diff.pow(2).sum()
                group_count = int(diff.numel())

                # Accumulate error statistics
                if group_id not in self._group_error_sums:
                    self._group_error_sums[group_id] = torch.zeros((), device=self.device)
                    self._group_error_counts[group_id] = 0
                self._group_error_sums[group_id] += group_sse
                self._group_error_counts[group_id] += group_count

            self.transition.expert_actions = expert_actions.detach()

            # blend actions and update log-prob
            self._blend_actions()
            self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Process environment step and reset expert policy hidden states on done."""
        # Call parent implementation
        super().process_env_step(obs, rewards, dones, extras)

        # Reset hidden states for MoE expert policies (in addition to parent's reset)
        if self.expert_guidance is not None:
            done_mask = dones.reshape(-1).to(torch.bool)
            for expert_policy in self.expert_policies:
                if expert_policy.is_recurrent:
                    expert_policy.hidden_state[:, done_mask] = 0
                    if hasattr(expert_policy, "cell_state"):
                        expert_policy.cell_state[:, done_mask] = 0

    def update(self) -> dict[str, float]:  # noqa: C901
        loss_dict = super().update()

        # Add per-group action errors to loss_dict for monitoring
        if self.expert_guidance is not None:
            for group_id in range(self.num_experts):
                if group_id in self._group_error_counts and self._group_error_counts[group_id] > 0:
                    avg_error = self._group_error_sums[group_id] / self._group_error_counts[group_id]
                    loss_dict[f"action_mse_group_{group_id}"] = avg_error.item()

            # Reset accumulators for next update cycle
            self._group_error_sums = {}
            self._group_error_counts = {}

        return loss_dict
