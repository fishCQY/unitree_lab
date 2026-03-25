# Copyright (c) 2024-2026, Light Robotics.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable, Generator

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.storage import RolloutStorage


class SymmetryClassifier:
    """Classifier to distinguish original vs mirrored observation sequences."""

    def __init__(
        self,
        obs_dict: TensorDict,
        obs_group: str,
        num_frames: int,
        mirror_obs_func: Callable[[torch.Tensor], torch.Tensor],
        cls_hidden_dims: tuple[int] | list[int] = (1024, 512),
        activation: str = "relu",
        obs_normalization: bool = True,
        reward_weight: float = 1.0,
        learning_rate: float = 1e-4,
        lr_scale: float | None = None,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 4,
        grad_pen_weight: float = 0.0,
        device: str = "cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        if kwargs:
            print(
                "SymmetryClassifier.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Observation configuration
        self.obs_group = obs_group
        self.num_cls_obs = obs_dict[obs_group].shape[-1]
        self.obs_buffer = torch.zeros(
            num_frames - 1, obs_dict[obs_group].shape[0], self.num_cls_obs, device=self.device
        )
        self.rewards_sum = torch.zeros(obs_dict[obs_group].shape[0], device=self.device)

        # Reward variance accumulators
        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0
        self.reward_count = 0

        # Reward delta accumulators (for average change rate)
        self.prev_reward: torch.Tensor | None = None
        self.reward_delta_sum = 0.0
        self.reward_delta_count = 0

        # Mirror function
        self.mirror_obs_func = mirror_obs_func
        if not callable(self.mirror_obs_func):
            raise ValueError("mirror_obs_func must be callable.")

        # Build classifier network
        self.classifier = MLP(self.num_cls_obs * num_frames, 1, cls_hidden_dims, activation).to(self.device)
        print(f"[SymmetryClassifier] Classifier MLP: {self.classifier}")

        # Observation normalization
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.num_cls_obs * num_frames).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()

        # Create optimizer
        self.optimizer = optim.Adam(self.classifier.parameters(), weight_decay=1e-5, lr=learning_rate)

        # Classifier parameters
        self.reward_weight = reward_weight
        self.lr_scale = lr_scale
        self.num_frames = num_frames
        self.learning_rate = learning_rate
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.grad_pen_weight = grad_pen_weight

        self.loss_fn = nn.BCEWithLogitsLoss()

    def reward(self, obs: TensorDict, rl_storage: RolloutStorage) -> torch.Tensor:
        """Compute symmetry reward from classifier uncertainty.

        Available Reward functions:
            reward = torch.clamp(1 - torch.square(probs - 1.0), min=0)
            reward = torch.clamp(1 - 4 * torch.square(probs.clip(max=0.5) - 0.5), min=0)
        Default Reward function:
            reward = torch.clamp(1 - torch.square(probs - 1.0), min=0)
        """
        with torch.no_grad():
            step = rl_storage.step
            rl_obs = rl_storage.observations[self.obs_group][:step]
            obs_seq = torch.cat([self.obs_buffer, rl_obs, obs[self.obs_group].unsqueeze(0)], dim=0).permute(1, 0, 2)
            cls_input = self.obs_normalizer(obs_seq[:, -self.num_frames :].flatten(1, 2))
            self.classifier.eval()
            logits = self.classifier(cls_input).squeeze(-1)
            probs = torch.sigmoid(logits)
            reward = torch.clamp(1 - torch.square(probs - 1.0), min=0)
            # Accumulate for variance calculation
            self.reward_sum += reward.sum().item()
            self.reward_sq_sum += (reward**2).sum().item()
            self.reward_count += reward.numel()
            # Accumulate for average change rate
            if self.prev_reward is not None:
                delta = torch.abs(reward - self.prev_reward)
                self.reward_delta_sum += delta.sum().item()
                self.reward_delta_count += delta.numel()
            self.prev_reward = reward.clone()
            self.classifier.train()
            return reward * self.reward_weight

    def update(
        self,
        rl_storage: RolloutStorage,
        rl_learning_rate: float,
    ) -> dict[str, float]:
        self.obs_buffer = rl_storage.observations[self.obs_group][-self.num_frames + 1 :].clone()
        if self.lr_scale is not None:
            self.optimizer.param_groups[0]["lr"] = rl_learning_rate * self.lr_scale

        _, acc_eval = self.evaluate(rl_storage)

        mean_cls_loss = 0.0
        mean_grad_pen_loss = 0.0

        generator = self._mini_batch_generator(rl_storage)
        for policy_obs_seq_batch, mirror_obs_seq_batch in generator:
            if self.obs_normalization:
                self.obs_normalizer.update(policy_obs_seq_batch)
                self.obs_normalizer.update(mirror_obs_seq_batch)

            # Normalize
            policy_obs_seq_batch = self.obs_normalizer(policy_obs_seq_batch)
            mirror_obs_seq_batch = self.obs_normalizer(mirror_obs_seq_batch)

            # Build classifier inputs
            inputs = torch.cat([policy_obs_seq_batch, mirror_obs_seq_batch], dim=0)
            labels = torch.cat(
                [
                    torch.zeros(policy_obs_seq_batch.size(0), 1, device=self.device),
                    torch.ones(mirror_obs_seq_batch.size(0), 1, device=self.device),
                ],
                dim=0,
            )

            logits = self.classifier(inputs)
            cls_loss = self.loss_fn(logits, labels)

            # Compute gradient penalty on both policy and mirrored inputs
            grad_pen_loss = torch.tensor(0.0, device=self.device)
            if self.grad_pen_weight > 0:
                policy_grad_pen = self.grad_pen(policy_obs_seq_batch)
                mirror_grad_pen = self.grad_pen(mirror_obs_seq_batch)
                grad_pen_loss = 0.5 * self.grad_pen_weight * (policy_grad_pen + mirror_grad_pen)
                total_loss = cls_loss + grad_pen_loss
            else:
                total_loss = cls_loss

            self.optimizer.zero_grad()
            total_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.classifier.parameters(), 1)
            self.optimizer.step()

            mean_cls_loss += cls_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_cls_loss /= num_updates
        mean_grad_pen_loss /= num_updates

        # Compute reward variance: Var(X) = E[X²] - E[X]²
        if self.reward_count > 0:
            mean_reward = self.reward_sum / self.reward_count
            reward_var = self.reward_sq_sum / self.reward_count - mean_reward**2
        else:
            reward_var = 0.0
        # Compute average change rate
        if self.reward_delta_count > 0:
            reward_delta_avg = self.reward_delta_sum / self.reward_delta_count
        else:
            reward_delta_avg = 0.0
        # Reset accumulators
        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0
        self.reward_count = 0
        self.reward_delta_sum = 0.0
        self.reward_delta_count = 0
        self.prev_reward = None

        return {
            "symm_cls_loss": mean_cls_loss,
            "symm_cls_grad_pen_loss": mean_grad_pen_loss,
            "symm_cls_acc_eval": acc_eval,
            "~symm_cls_reward_var": reward_var,
            "~symm_cls_reward_delta": reward_delta_avg,
        }

    def grad_pen(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty on classifier inputs.

        Args:
            obs: Flattened observation sequence tensor of shape (batch, obs_dim * num_frames).

        Returns:
            Gradient penalty: mean squared L2 norm of input gradients.
        """
        obs = obs.clone().detach().requires_grad_(True)
        logits = self.classifier(obs)
        grad = torch.autograd.grad(
            logits,
            obs,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            only_inputs=True,
        )[0]
        return (grad.norm(2, dim=1)).pow(2).mean()

    def evaluate(self, rl_storage: RolloutStorage) -> tuple[float, float]:
        """Evaluate classifier accuracy and loss on stored observations."""
        policy_obs = rl_storage.observations[self.obs_group]
        policy_sequences, num_valid_policy = self._extract_valid_sequences(policy_obs, rl_storage.dones)
        if num_valid_policy == 0:
            raise ValueError("No valid policy sequences found")

        policy_flat, mirror_flat = self._build_flattened_batches(policy_sequences)

        # Switch to eval mode temporarily
        was_training = self.classifier.training
        self.classifier.eval()
        with torch.no_grad():
            if self.obs_normalization:
                policy_flat = self.obs_normalizer(policy_flat)
                mirror_flat = self.obs_normalizer(mirror_flat)

            inputs = torch.cat([policy_flat, mirror_flat], dim=0)
            labels = torch.cat(
                [
                    torch.zeros(policy_flat.size(0), 1, device=self.device),
                    torch.ones(mirror_flat.size(0), 1, device=self.device),
                ],
                dim=0,
            )
            logits = self.classifier(inputs)
            cls_loss = self.loss_fn(logits, labels)
            preds = (logits >= 0).float()
            cls_acc = (preds == labels).float().mean()

        if was_training:
            self.classifier.train()

        return cls_loss.item(), cls_acc.item()

    def _mini_batch_generator(self, rl_storage: RolloutStorage) -> Generator:
        """Generate mini batches of policy and mirrored observation sequences."""
        policy_obs = rl_storage.observations[self.obs_group]
        policy_sequences, num_valid_policy = self._extract_valid_sequences(policy_obs, rl_storage.dones)
        if num_valid_policy == 0:
            raise ValueError("No valid policy sequences found")

        mini_batch_size = num_valid_policy // self.num_mini_batches
        if mini_batch_size == 0:
            raise ValueError("mini_batch_size is 0. Check num_mini_batches and available data.")

        for _ in range(self.num_learning_epochs):
            policy_perm = torch.randperm(num_valid_policy, device=self.device)

            for i in range(self.num_mini_batches):
                start, end = i * mini_batch_size, (i + 1) * mini_batch_size
                policy_batch = policy_sequences[policy_perm[start : min(end, num_valid_policy)]]

                policy_flat, mirror_flat = self._build_flattened_batches(policy_batch)
                yield (policy_flat, mirror_flat)

    def _build_flattened_batches(self, policy_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build flattened policy and mirrored batches for classifier."""
        batch_size, num_frames, obs_dim = policy_batch.shape
        policy_batch_flat = policy_batch.reshape(-1, obs_dim)
        mirror_batch_flat = self.mirror_obs_func(policy_batch_flat)
        mirror_batch = mirror_batch_flat.reshape(batch_size, num_frames, obs_dim)

        return policy_batch.flatten(1, 2), mirror_batch.flatten(1, 2)

    def _extract_valid_sequences(self, data: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Extract valid sequences that don't cross done boundaries."""
        num_steps = data.shape[0]

        if num_steps < self.num_frames:
            return torch.empty(0, self.num_frames, data.shape[2], device=data.device), 0

        # Create sliding windows: (num_windows, num_envs, num_frames, feature_dim)
        data_windows = data.unfold(0, self.num_frames, 1).permute(0, 1, 3, 2)
        num_windows = data_windows.shape[0]

        # Check for dones within sequences (excluding last frame of each window)
        if num_windows > 0 and dones.shape[0] > self.num_frames - 1:
            dones_check = dones[: num_windows + self.num_frames - 2].unfold(0, self.num_frames - 1, 1)
            dones_check = dones_check.squeeze(2).any(dim=2)
        else:
            return torch.empty(0, self.num_frames, data.shape[2], device=data.device), 0

        # Ensure bool dtype to avoid deprecation warning
        valid_mask = (~dones_check).flatten().to(torch.bool)

        data_flat = data_windows.flatten(0, 1)

        if data_flat.shape[0] != valid_mask.shape[0]:
            print(
                f"[SymmetryClassifier] Warning: Shape mismatch: data_flat={data_flat.shape[0]}, "
                f"valid_mask={valid_mask.shape[0]}"
            )
            return torch.empty(0, self.num_frames, data.shape[2], device=data.device), 0

        return data_flat[valid_mask], valid_mask.sum().item()  # type: ignore

    def sync_normalizer_buffers(self) -> None:
        """Synchronize EmpiricalNormalization buffers across all GPUs.

        Broadcasts the running statistics (_mean, _var, _std, count) of the
        SymmetryClassifier observation normalizer from rank 0 to all other ranks.  This
        should be called once per training iteration after :meth:`update`.
        """
        if not self.is_multi_gpu:
            return
        if not self.obs_normalization:
            return

        from rsl_rl.networks import EmpiricalNormalization

        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer.broadcast_buffers(src=0)

    def broadcast_parameters(self) -> None:
        """Broadcast classifier parameters to all GPUs."""
        model_params = [self.classifier.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.classifier.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them."""
        grads = [param.grad.view(-1) for param in self.classifier.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in self.classifier.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
