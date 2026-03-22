"""SymmetryClassifier plugin for enforcing left-right gait symmetry.

Trains a binary classifier to distinguish original vs mirrored observation
sequences. If the classifier cannot tell them apart (output ≈ 0.5), the
policy's behavior is symmetric — and it gets a high reward.

Integrated into AMPPluginRunner alongside AMPPlugin.
"""

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
        **kwargs,
    ) -> None:
        if kwargs:
            print(f"SymmetryClassifier.__init__ got unexpected arguments: {list(kwargs.keys())}")

        self.device = device
        self.obs_group = obs_group
        self.num_cls_obs = obs_dict[obs_group].shape[-1]
        self.obs_buffer = torch.zeros(
            num_frames - 1, obs_dict[obs_group].shape[0], self.num_cls_obs, device=self.device,
        )
        self.rewards_sum = torch.zeros(obs_dict[obs_group].shape[0], device=self.device)

        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0
        self.reward_count = 0
        self.prev_reward: torch.Tensor | None = None
        self.reward_delta_sum = 0.0
        self.reward_delta_count = 0

        self.mirror_obs_func = mirror_obs_func
        if not callable(self.mirror_obs_func):
            raise ValueError("mirror_obs_func must be callable.")

        self.classifier = MLP(self.num_cls_obs * num_frames, 1, cls_hidden_dims, activation).to(self.device)
        print(f"[SymmetryClassifier] Classifier MLP: {self.classifier}")

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.num_cls_obs * num_frames).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()

        self.optimizer = optim.Adam(self.classifier.parameters(), weight_decay=1e-5, lr=learning_rate)

        self.reward_weight = reward_weight
        self.lr_scale = lr_scale
        self.num_frames = num_frames
        self.learning_rate = learning_rate
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.grad_pen_weight = grad_pen_weight

        self.loss_fn = nn.BCEWithLogitsLoss()

    def reward(self, obs: TensorDict, rl_storage: RolloutStorage) -> torch.Tensor:
        """Compute symmetry reward: high when classifier can't distinguish original from mirrored."""
        with torch.no_grad():
            step = rl_storage.step
            rl_obs = rl_storage.observations[self.obs_group][:step]
            obs_seq = torch.cat([self.obs_buffer, rl_obs, obs[self.obs_group].unsqueeze(0)], dim=0).permute(1, 0, 2)
            cls_input = self.obs_normalizer(obs_seq[:, -self.num_frames:].flatten(1, 2))
            self.classifier.eval()
            logits = self.classifier(cls_input).squeeze(-1)
            probs = torch.sigmoid(logits)
            reward = torch.clamp(1 - torch.square(probs - 1.0), min=0)
            self.reward_sum += reward.sum().item()
            self.reward_sq_sum += (reward ** 2).sum().item()
            self.reward_count += reward.numel()
            if self.prev_reward is not None:
                delta = torch.abs(reward - self.prev_reward)
                self.reward_delta_sum += delta.sum().item()
                self.reward_delta_count += delta.numel()
            self.prev_reward = reward.clone()
            self.classifier.train()
            return reward * self.reward_weight

    def update(self, rl_storage: RolloutStorage, rl_learning_rate: float) -> dict[str, float]:
        """Train the classifier on original vs mirrored observation sequences."""
        self.obs_buffer = rl_storage.observations[self.obs_group][-self.num_frames + 1:].clone()
        if self.lr_scale is not None:
            self.optimizer.param_groups[0]["lr"] = rl_learning_rate * self.lr_scale

        _, acc_eval = self.evaluate(rl_storage)

        mean_cls_loss = 0.0
        mean_grad_pen_loss = 0.0

        for policy_flat, mirror_flat in self._mini_batch_generator(rl_storage):
            if self.obs_normalization:
                self.obs_normalizer.update(policy_flat)
                self.obs_normalizer.update(mirror_flat)

            policy_flat = self.obs_normalizer(policy_flat)
            mirror_flat = self.obs_normalizer(mirror_flat)

            inputs = torch.cat([policy_flat, mirror_flat], dim=0)
            labels = torch.cat([
                torch.zeros(policy_flat.size(0), 1, device=self.device),
                torch.ones(mirror_flat.size(0), 1, device=self.device),
            ], dim=0)

            logits = self.classifier(inputs)
            cls_loss = self.loss_fn(logits, labels)

            grad_pen_loss = torch.tensor(0.0, device=self.device)
            if self.grad_pen_weight > 0:
                policy_gp = self._grad_pen(policy_flat)
                mirror_gp = self._grad_pen(mirror_flat)
                grad_pen_loss = 0.5 * self.grad_pen_weight * (policy_gp + mirror_gp)
                total_loss = cls_loss + grad_pen_loss
            else:
                total_loss = cls_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.classifier.parameters(), 1)
            self.optimizer.step()

            mean_cls_loss += cls_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_cls_loss /= max(num_updates, 1)
        mean_grad_pen_loss /= max(num_updates, 1)

        if self.reward_count > 0:
            mean_reward = self.reward_sum / self.reward_count
            reward_var = self.reward_sq_sum / self.reward_count - mean_reward ** 2
        else:
            reward_var = 0.0
        reward_delta_avg = self.reward_delta_sum / max(self.reward_delta_count, 1)

        self.reward_sum = self.reward_sq_sum = 0.0
        self.reward_count = 0
        self.reward_delta_sum = 0.0
        self.reward_delta_count = 0
        self.prev_reward = None

        return {
            "symm_cls_loss": mean_cls_loss,
            "symm_cls_grad_pen": mean_grad_pen_loss,
            "symm_cls_acc": acc_eval,
            "~symm_cls_reward_var": reward_var,
            "~symm_cls_reward_delta": reward_delta_avg,
        }

    def _grad_pen(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.clone().detach().requires_grad_(True)
        logits = self.classifier(obs)
        grad = torch.autograd.grad(
            logits, obs, grad_outputs=torch.ones_like(logits), create_graph=True, only_inputs=True,
        )[0]
        return grad.norm(2, dim=1).pow(2).mean()

    def evaluate(self, rl_storage: RolloutStorage) -> tuple[float, float]:
        policy_obs = rl_storage.observations[self.obs_group]
        seqs, n_valid = self._extract_valid_sequences(policy_obs, rl_storage.dones)
        if n_valid == 0:
            return 0.0, 0.5

        policy_flat, mirror_flat = self._build_flattened_batches(seqs)

        was_training = self.classifier.training
        self.classifier.eval()
        with torch.no_grad():
            if self.obs_normalization:
                policy_flat = self.obs_normalizer(policy_flat)
                mirror_flat = self.obs_normalizer(mirror_flat)
            inputs = torch.cat([policy_flat, mirror_flat], dim=0)
            labels = torch.cat([
                torch.zeros(policy_flat.size(0), 1, device=self.device),
                torch.ones(mirror_flat.size(0), 1, device=self.device),
            ], dim=0)
            logits = self.classifier(inputs)
            loss = self.loss_fn(logits, labels)
            acc = ((logits >= 0).float() == labels).float().mean()
        if was_training:
            self.classifier.train()
        return loss.item(), acc.item()

    def _mini_batch_generator(self, rl_storage: RolloutStorage) -> Generator:
        policy_obs = rl_storage.observations[self.obs_group]
        seqs, n_valid = self._extract_valid_sequences(policy_obs, rl_storage.dones)
        if n_valid == 0:
            return

        mb_size = n_valid // self.num_mini_batches
        if mb_size == 0:
            return

        for _ in range(self.num_learning_epochs):
            perm = torch.randperm(n_valid, device=self.device)
            for i in range(self.num_mini_batches):
                batch = seqs[perm[i * mb_size: (i + 1) * mb_size]]
                yield self._build_flattened_batches(batch)

    def _build_flattened_batches(self, policy_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs, nf, od = policy_batch.shape
        policy_flat = policy_batch.reshape(-1, od)
        mirror_flat = self.mirror_obs_func(policy_flat)
        mirror_batch = mirror_flat.reshape(bs, nf, od)
        return policy_batch.flatten(1, 2), mirror_batch.flatten(1, 2)

    def _extract_valid_sequences(self, data: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, int]:
        num_steps = data.shape[0]
        if num_steps < self.num_frames:
            return torch.empty(0, self.num_frames, data.shape[2], device=data.device), 0

        windows = data.unfold(0, self.num_frames, 1).permute(0, 1, 3, 2)
        num_windows = windows.shape[0]

        if num_windows > 0 and dones.shape[0] > self.num_frames - 1:
            dones_check = dones[:num_windows + self.num_frames - 2].unfold(0, self.num_frames - 1, 1)
            dones_check = dones_check.squeeze(2).any(dim=2)
        else:
            return torch.empty(0, self.num_frames, data.shape[2], device=data.device), 0

        valid_mask = (~dones_check).flatten().to(torch.bool)
        flat = windows.flatten(0, 1)

        if flat.shape[0] != valid_mask.shape[0]:
            return torch.empty(0, self.num_frames, data.shape[2], device=data.device), 0

        return flat[valid_mask], valid_mask.sum().item()
