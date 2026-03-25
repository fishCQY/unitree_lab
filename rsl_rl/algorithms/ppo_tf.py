# Copyright (c) 2024-2026, Light Robotics.
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause

"""Proximal Policy Optimization for Transformer policies (PPO-TF).

This module extends the standard PPO algorithm with:
- A Transformer-specific mini-batch generator that reconstructs sliding-window
  observation sequences from single-frame rollout storage.
- Gradient accumulation to reduce peak GPU memory during training: each mini-batch
  is split into ``accumulation_steps`` micro-batches, and gradients are accumulated
  before a single optimizer step.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import ActorCriticTransformerBase
from rsl_rl.storage.rollout_storage_tf import RolloutStorageTF


class PPO_TF(PPO):
    """PPO with Transformer policy support and gradient accumulation."""

    def __init__(
        self,
        policy: ActorCriticTransformerBase,
        accumulation_steps: int = 1,
        weight_decay: float = 0.0,
        warmup_iterations: int = 0,
        min_lr: float = 1e-6,
        cosine_max_iterations: int = 10000,
        amp_dtype: str | None = None,
        torch_compile: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(policy=policy, **kwargs)
        self.accumulation_steps = max(1, accumulation_steps)

        # Keep behavior explicit: these options are currently not implemented in PPO_TF.
        if self.symmetry is not None:
            raise NotImplementedError(
                "PPO_TF does not support `symmetry_cfg` yet. Disable `symmetry_cfg` or use `PPO` instead."
            )
        if self.adv_filtering_ratio > 0.0:
            raise NotImplementedError(
                "PPO_TF does not support `adv_filtering_ratio > 0.0` yet because "
                "`RolloutStorageTF.transformer_mini_batch_generator` has no advantage filtering path. "
                "Set `adv_filtering_ratio=0.0` or use `PPO`."
            )

        # Mixed precision setup
        if amp_dtype == "bf16":
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
        elif amp_dtype == "fp16":
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        elif amp_dtype is None:
            self.amp_enabled = False
            self.amp_dtype = torch.float32
        else:
            raise ValueError(f"Unknown amp_dtype: {amp_dtype!r}. Must be 'bf16', 'fp16', or None.")
        # GradScaler is only needed for FP16; for BF16/None it acts as a no-op.
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == "fp16"))
        if self.amp_enabled:
            print(f"PPO_TF: Mixed precision enabled with dtype={amp_dtype}")

        # torch.compile: deferred to first update() call so that checkpoint loading
        # (which happens between __init__ and learn()) sees normal state_dict keys
        # without the ``_orig_mod.`` prefix that torch.compile introduces.
        self._torch_compile_pending = torch_compile

        # Warmup and schedule parameters
        self.warmup_iterations = warmup_iterations
        self.min_lr = min_lr
        self.cosine_max_iterations = cosine_max_iterations
        self.peak_lr = self.learning_rate  # Store peak LR for schedule computation

        # Override optimizer with AdamW when weight decay is enabled
        if weight_decay > 0.0:
            decay_params = []
            no_decay_params = []
            for name, param in self.policy.named_parameters():
                if not param.requires_grad:
                    continue
                # Decay: Transformer encoder linear weights (attention, SwiGLU FFN, GRU gates)
                # No decay: RMSNorm .scale, output heads, critic MLP, normalizers, noise std
                if "_transformer." in name and not name.endswith(".scale"):
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
            self.optimizer = optim.AdamW(
                [
                    {"params": decay_params, "weight_decay": weight_decay},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=self.learning_rate,
            )
            print(
                f"PPO_TF: Using AdamW with weight_decay={weight_decay} "
                f"(decay params: {len(decay_params)}, no-decay params: {len(no_decay_params)})"
            )

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        """Initialize Transformer-aware rollout storage."""
        self.storage = RolloutStorageTF(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )
        if self.expert_policy is not None and self.expert_policy.is_recurrent:
            layers, _, hidden_dim = self.expert_policy.hidden_state.shape
            self.expert_policy.hidden_state = torch.zeros(layers, num_envs, hidden_dim, device=self.device)
            if hasattr(self.expert_policy, "cell_state"):
                self.expert_policy.cell_state = torch.zeros(layers, num_envs, hidden_dim, device=self.device)

    # def act(self, obs: TensorDict) -> torch.Tensor:
    #     super().act(obs)
    #     return (
    #         self.transition.expert_actions
    #         + torch.randn_like(self.transition.actions) * self.transition.action_sigma
    #     )

    def update(self) -> dict[str, float]:  # noqa: C901
        # Deferred torch.compile: apply on first update() so checkpoint loading
        # (which happens between __init__ and learn()) sees normal state_dict keys.
        if self._torch_compile_pending:
            self._apply_torch_compile()
            self._torch_compile_pending = False

        # Swap in compiled submodules for the duration of update().
        originals: dict[str, nn.Module] | None = None
        if getattr(self, "_torch_compiled", False):
            originals = self._swap_in_compiled()

        # Explained variance: how well the value function predicts actual returns.
        with torch.no_grad():
            values_flat = self.storage.values.flatten()
            returns_flat = self.storage.returns.flatten()
            var_returns = returns_flat.var()
            explained_var = (1 - (returns_flat - values_flat).var() / var_returns).item() if var_returns > 0 else 0.0

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_clip_fraction = 0
        mean_approx_kl = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_guidance_loss = 0 if self.expert_guidance else None

        # --- Per-iteration LR update: warmup (any schedule) or cosine ---
        in_warmup = self.warmup_iterations > 0 and self.iteration_count < self.warmup_iterations
        if in_warmup:
            self._apply_warmup_lr()
        elif self.schedule == "cosine":
            self._apply_cosine_lr()

        generator = self.storage.transformer_mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
            self.policy.context_len,
            groups_to_window=self.policy.groups_to_window,
        )

        num_mini_batches = 0
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
            expert_actions_batch,
        ) in generator:
            accum = self.accumulation_steps

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            total_size = obs_batch.batch_size[0]
            micro_size = total_size // accum
            if micro_size == 0:
                accum = 1
                micro_size = total_size

            self.optimizer.zero_grad()
            if self.rnd:
                self.rnd_optimizer.zero_grad()

            mb_value_loss = 0.0
            mb_surrogate_loss = 0.0
            mb_entropy = 0.0
            mb_rnd_loss = 0.0
            mb_guidance_loss = 0.0
            last_mu_batch = None
            last_sigma_batch = None
            last_old_mu_batch = None
            last_old_sigma_batch = None

            for micro_idx in range(accum):
                s = micro_idx * micro_size
                e = (micro_idx + 1) * micro_size if micro_idx < accum - 1 else total_size
                micro_bs = e - s

                m_obs = obs_batch[s:e]
                m_actions = actions_batch[s:e]
                m_target_values = target_values_batch[s:e]
                m_advantages = advantages_batch[s:e]
                m_returns = returns_batch[s:e]
                m_old_log_prob = old_actions_log_prob_batch[s:e]
                m_old_mu = old_mu_batch[s:e]
                m_old_sigma = old_sigma_batch[s:e]
                m_expert_actions = expert_actions_batch[s:e] if expert_actions_batch is not None else None

                # --- Forward pass + loss computation under mixed precision ---
                with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                    self.policy.act(m_obs, masks=masks_batch, hidden_state=hidden_states_batch[0])
                    actions_log_prob = self.policy.get_actions_log_prob(m_actions)
                    value_pred = self.policy.evaluate(m_obs, masks=masks_batch, hidden_state=hidden_states_batch[1])
                    mu = self.policy.action_mean[:micro_bs]
                    sigma = self.policy.action_std[:micro_bs]
                    entropy = self.policy.entropy[:micro_bs]

                    ratio = torch.exp(actions_log_prob - torch.squeeze(m_old_log_prob))
                    if num_mini_batches == 0:
                        print(f"ratio: {ratio.mean()}, {ratio.min()}, {ratio.max()}")
                    num_mini_batches += 1
                    surrogate = -torch.squeeze(m_advantages) * ratio
                    surrogate_clipped = -torch.squeeze(m_advantages) * torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                    if self.use_clipped_value_loss:
                        value_clipped = m_target_values + (value_pred - m_target_values).clamp(
                            -self.clip_param, self.clip_param
                        )
                        value_losses = (value_pred - m_returns).pow(2)
                        value_losses_clipped = (value_clipped - m_returns).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = (m_returns - value_pred).pow(2).mean()

                    if self.expert_guidance and self.expert_guidance["pure_distillation"]:
                        loss = self.value_loss_coef * value_loss
                    else:
                        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                    if self.expert_guidance:
                        guidance_loss = self._compute_guidance_loss(m_expert_actions)
                        loss += self.guidance_weight * guidance_loss
                        mb_guidance_loss += guidance_loss.item()

                # --- Backward (GradScaler is no-op for BF16/disabled) ---
                self.grad_scaler.scale(loss / accum).backward()

                if self.rnd:
                    with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                        rnd_loss = self._compute_rnd_loss(m_obs, micro_bs)
                    self.grad_scaler.scale(rnd_loss / accum).backward()
                    mb_rnd_loss += rnd_loss.item()

                last_mu_batch = mu.float()
                last_sigma_batch = sigma.float()
                last_old_mu_batch = m_old_mu[:micro_bs].float()
                last_old_sigma_batch = m_old_sigma[:micro_bs].float()

                with torch.no_grad():
                    mean_clip_fraction += ((torch.abs(ratio - 1.0) > self.clip_param).float().mean().item()) / accum
                    mean_approx_kl += (((ratio - 1) - torch.log(ratio)).mean().item()) / accum

                mb_value_loss += value_loss.item()
                mb_surrogate_loss += surrogate_loss.item()
                mb_entropy += entropy.mean().item()

            # --- After all micro-batches: optimizer step ---
            if self.is_multi_gpu:
                self.reduce_parameters()

            self.grad_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.grad_scaler.step(self.optimizer)
            if self.rnd_optimizer:
                self.grad_scaler.unscale_(self.rnd_optimizer)
                self.grad_scaler.step(self.rnd_optimizer)
            self.grad_scaler.update()

            # --- Per-mini-batch LR: adaptive KL schedule (only post-warmup) ---
            if not in_warmup and self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(last_sigma_batch / last_old_sigma_batch + 1.0e-5)
                        + (torch.square(last_old_sigma_batch) + torch.square(last_old_mu_batch - last_mu_batch))
                        / (2.0 * torch.square(last_sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            mean_value_loss += mb_value_loss / accum
            mean_surrogate_loss += mb_surrogate_loss / accum
            mean_entropy += mb_entropy / accum
            if mean_rnd_loss is not None:
                mean_rnd_loss += mb_rnd_loss / accum
            if mean_guidance_loss is not None:
                mean_guidance_loss += mb_guidance_loss / accum

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_clip_fraction /= num_updates
        mean_approx_kl /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_guidance_loss is not None:
            mean_guidance_loss /= num_updates

        self.storage.save_transformer_prefix(self.policy.context_len, self.policy.groups_to_window)
        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "clip_fraction": mean_clip_fraction,
            "approx_kl": mean_approx_kl,
            "explained_variance": explained_var,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.expert_guidance:
            loss_dict["guidance"] = mean_guidance_loss
            loss_dict["guidance_weight"] = self.guidance_weight
            if self._action_mse_count > 0:
                loss_dict["action_mse"] = (self._action_mse_sum / self._action_mse_count).item()
            # Reset accumulators for next update cycle
            self._action_mse_sum.zero_()
            self._action_mse_count = 0
        if self.expert_guidance:
            self._update_guidance_weight()
            if self.expert_guidance["pure_distillation"]:
                loss_dict.pop("surrogate")

        # Swap back original (non-compiled) submodules so that save / JIT export
        # sees a clean policy without torch.compile wrappers.
        if originals is not None:
            self._swap_out_compiled(self.policy, originals)

        self.iteration_count += 1
        return loss_dict

    # ------------------------------------------------------------------
    # Learning rate schedule helpers
    # ------------------------------------------------------------------

    def _apply_warmup_lr(self) -> None:
        """Linear warmup: ramp LR from ``min_lr`` to ``peak_lr``.

        Called once per iteration during the warmup phase (iteration < warmup_iterations).
        Independent of the chosen schedule — after warmup the schedule takes over.
        """
        alpha = self.iteration_count / max(1, self.warmup_iterations)
        self.learning_rate = self.min_lr + (self.peak_lr - self.min_lr) * alpha
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def _apply_cosine_lr(self) -> None:
        """Cosine annealing: decay LR from ``peak_lr`` to ``min_lr``.

        Called once per iteration when schedule='cosine' and warmup has ended.
        The decay spans from ``warmup_iterations`` to ``cosine_max_iterations``.
        """
        progress = (self.iteration_count - self.warmup_iterations) / max(
            1, self.cosine_max_iterations - self.warmup_iterations
        )
        progress = min(progress, 1.0)
        self.learning_rate = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    # ------------------------------------------------------------------
    # torch.compile helpers
    # ------------------------------------------------------------------

    def _apply_torch_compile(self) -> None:
        """JIT-compile Transformer encoder submodules for kernel fusion.

        Compiled modules are stored as separate attributes (``_compiled_modules``)
        rather than replacing the originals on the policy.  This keeps the policy's
        ``state_dict()`` and ``torch.jit.script`` export working normally.
        Parameters are shared between original and compiled versions, so
        ``optimizer.step()`` updates both.
        """
        self._compiled_modules: dict[str, torch.nn.Module] = {}
        self._compiled_modules["actor_transformer"] = torch.compile(self.policy.actor_transformer, dynamic=True)
        if hasattr(self.policy, "critic_transformer"):
            self._compiled_modules["critic_transformer"] = torch.compile(self.policy.critic_transformer, dynamic=True)
        self._torch_compiled = True
        print(f"PPO_TF: torch.compile enabled for submodules: {list(self._compiled_modules.keys())}")

    def _swap_in_compiled(self) -> dict[str, nn.Module]:
        """Replace policy submodules with compiled versions. Returns originals."""
        originals: dict[str, nn.Module] = {}
        for attr, compiled in self._compiled_modules.items():
            originals[attr] = getattr(self.policy, attr)
            setattr(self.policy, attr, compiled)
        return originals

    @staticmethod
    def _swap_out_compiled(policy: nn.Module, originals: dict[str, nn.Module]) -> None:
        """Restore original (non-compiled) submodules on the policy."""
        for attr, orig in originals.items():
            setattr(policy, attr, orig)
