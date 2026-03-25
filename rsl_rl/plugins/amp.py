"""Adversarial Motion Priors (AMP) Plugin.

Standalone AMP implementation that can be attached to any RL runner.
Standalone AMP plugin that can be attached to any RL runner.

Features:
  - Multiple loss types: GAN, LSGAN, WGAN
  - Linear interpolation (lerp) reward fusion
  - Per-layer weight decay for discriminator
  - Plugin architecture (decoupled from PPO)
  - Conditional AMP with learnable embeddings
  - Training noise for regularization
  - Valid sequence extraction from RolloutStorage
  - Multi-GPU gradient synchronization
  - Learning rate scaling with policy lr

Usage:
    amp = AMPPlugin(cfg, obs_dim=64, num_envs=4096, device="cuda:0")
    amp.set_offline_data(offline_dataset)

    # During rollout:
    style_reward, disc_score = amp.reward(obs, storage, step_dt)
    total_reward = amp.combine_reward(task_reward, style_reward)

    # After PPO update:
    amp_loss_dict = amp.update(storage, rl_learning_rate)
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from tensordict import TensorDict

from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_nn_activation


class LossType(Enum):
    GAN = "GAN"
    LSGAN = "LSGAN"
    WGAN = "WGAN"


class AMPPlugin:
    """Adversarial Motion Priors plugin for style-based reward shaping.

    The plugin manages its own discriminator network, optimizer, observation
    normalizer, and (optionally) condition embedding. It reads single-step
    AMP observations from the rollout storage and constructs multi-frame
    sequences internally.

    Args:
        cfg: AMP configuration dictionary.
        obs_dim: Dimension of single-step AMP observation.
        num_envs: Number of parallel environments.
        device: Torch device.
        multi_gpu_cfg: Optional multi-GPU configuration.
    """

    def __init__(
        self,
        cfg: dict,
        obs_dim: int,
        num_envs: int,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        self.device = device
        self.obs_dim = obs_dim
        self.num_envs = num_envs

        # --- Config ---
        self.obs_group: str = cfg["obs_group"]
        self.condition_obs_group: str | None = cfg.get("condition_obs_group")
        self.num_frames: int = cfg.get("num_frames", 2)
        self.style_reward_scale: float = cfg.get("style_reward_scale", 2.0)
        self.grad_penalty_scale: float = cfg.get("grad_penalty_scale", 10.0)
        self.max_grad_norm: float = cfg.get("disc_max_grad_norm", 0.5)
        self.noise_scale: float | None = cfg.get("noise_scale")
        self.num_learning_epochs: int = cfg.get("disc_num_learning_epochs", 5)
        self.num_mini_batches: int = cfg.get("disc_num_mini_batches", 4)
        self.lr_scale: float | None = cfg.get("lr_scale")
        hidden_dims: list[int] = cfg.get("hidden_dims", [1024, 512])
        activation_name: str = cfg.get("activation", "relu")
        trunk_wd: float = cfg.get("disc_trunk_weight_decay", 1e-4)
        linear_wd: float = cfg.get("disc_linear_weight_decay", 1e-2)
        lr: float = cfg.get("disc_learning_rate", 5e-4)

        # Loss type
        loss_str = cfg.get("loss_type", "LSGAN").upper()
        self.loss_type = LossType(loss_str)

        # --- Conditional AMP ---
        self.conditional = self.condition_obs_group is not None
        self.num_conditions: int = cfg.get("num_conditions", 0) if self.conditional else 0
        embedding_dim: int = cfg.get("condition_embedding_dim", 16) if self.conditional else 0
        self.condition_embedding: nn.Embedding | None = None
        if self.conditional and self.num_conditions > 0:
            self.condition_embedding = nn.Embedding(self.num_conditions, embedding_dim).to(device)

        # --- Build Discriminator ---
        activation = resolve_nn_activation(activation_name)
        disc_input_dim = obs_dim * self.num_frames
        if self.conditional:
            disc_input_dim += embedding_dim

        layers: list[nn.Module] = []
        in_dim = disc_input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(in_dim, hdim))
            layers.append(activation)
            in_dim = hdim
        self.disc_trunk = nn.Sequential(*layers).to(device)
        self.disc_linear = nn.Linear(in_dim, 1).to(device)

        print(f"[AMPPlugin] Discriminator: input={disc_input_dim}, hidden={hidden_dims}, "
              f"loss={self.loss_type.value}, conditional={self.conditional}")

        # --- Normalizers ---
        self.obs_normalizer = EmpiricalNormalization(shape=obs_dim, until=int(1e8)).to(device)
        self.output_normalizer: nn.Module
        if self.loss_type == LossType.WGAN:
            self.output_normalizer = EmpiricalNormalization(shape=1, until=int(1e8)).to(device)
        else:
            self.output_normalizer = nn.Identity().to(device)

        # --- Optimizer ---
        param_groups = [
            {"name": "disc_trunk", "params": self.disc_trunk.parameters(), "weight_decay": trunk_wd},
            {"name": "disc_linear", "params": self.disc_linear.parameters(), "weight_decay": linear_wd},
        ]
        if self.condition_embedding is not None:
            param_groups.append(
                {"name": "cond_emb", "params": self.condition_embedding.parameters(), "weight_decay": 0.0}
            )
        self.optimizer = optim.Adam(param_groups, lr=lr)

        # --- Observation buffer for bridging rollouts ---
        self.obs_buffer = torch.zeros(max(self.num_frames - 1, 1), num_envs, obs_dim, device=device)
        self.cond_buffer: torch.Tensor | None = None
        if self.conditional:
            self.cond_buffer = torch.zeros(max(self.num_frames - 1, 1), num_envs, 1, device=device, dtype=torch.long)

        # --- Offline dataset ---
        self.offline_dataset = None
        self._offline_sequences: torch.Tensor | None = None
        self._offline_cond_ids: torch.Tensor | None = None

        # --- Multi-GPU ---
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg else 1

    # ------------------------------------------------------------------
    # Offline data
    # ------------------------------------------------------------------

    def set_offline_data(self, dataset) -> None:
        """Set and preprocess offline motion dataset.

        Args:
            dataset: An AMPMotionData dataclass with fields:
                motion_data (Tensor): (total_frames, feature_dim)
                motion_lengths (Tensor): (num_motions,)
                motion_ids (Tensor): (total_frames,)
                condition_ids (Tensor | None): (num_motions,) for conditional AMP
        """
        self.offline_dataset = dataset
        data = dataset.motion_data.to(self.device)  # (N, D)
        motion_ids = dataset.motion_ids.to(self.device)  # (N,)

        # Initialize normalizer with offline data
        if self.num_frames > 1:
            self.obs_normalizer.update(data.repeat(1, self.num_frames).view(-1, self.obs_dim))
        else:
            self.obs_normalizer.update(data)

        # Pre-build offline sequences
        dones = torch.zeros(len(data), dtype=torch.bool, device=self.device)
        dones[:-1] = motion_ids[1:] != motion_ids[:-1]
        dones[-1] = True

        seqs = self._extract_sequences_from_flat(data, dones, self.num_frames)
        self._offline_sequences = seqs  # (M, num_frames, obs_dim)

        # Pre-build offline condition IDs per sequence
        if self.conditional and dataset.condition_ids is not None:
            per_frame_cond = dataset.condition_ids[motion_ids].to(self.device)  # (N,)
            # Each sequence's condition = last frame's condition
            cond_seqs = self._extract_sequences_from_flat(
                per_frame_cond.unsqueeze(-1).float(), dones, self.num_frames
            )
            self._offline_cond_ids = cond_seqs[:, -1, 0].long()  # (M,)

        self._cond_index_map = None
        if self._offline_cond_ids is not None:
            self._rebuild_cond_index_map(self._offline_cond_ids)

        print(f"[AMPPlugin] Offline: {len(data)} frames → {len(seqs)} valid sequences")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _disc_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminator forward: trunk → linear."""
        return self.disc_linear(self.disc_trunk(x))

    # ------------------------------------------------------------------
    # Reward computation (called during rollout, no_grad)
    # ------------------------------------------------------------------

    def reward(
        self,
        obs: TensorDict,
        storage: RolloutStorage,
        step_dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute AMP style reward from current observations.

        Constructs a multi-frame sequence from the obs_buffer (tail of
        previous rollout), current rollout storage, and the latest obs.

        Args:
            obs: Current observation TensorDict from environment.
            storage: RolloutStorage with observations accumulated so far.
            step_dt: Simulation time step for reward scaling.

        Returns:
            Tuple of (style_reward, disc_score), each shape (num_envs,).
        """
        with torch.no_grad():
            step = storage.step
            # Gather observation history: buffer + stored + current
            stored_obs = storage.observations[self.obs_group][:step]  # (step, E, D)
            current_obs = obs[self.obs_group].unsqueeze(0)  # (1, E, D)
            full_seq = torch.cat([self.obs_buffer, stored_obs, current_obs], dim=0)  # (T, E, D)
            full_seq = full_seq.permute(1, 0, 2)  # (E, T, D)

            # Take last num_frames
            amp_obs = full_seq[:, -self.num_frames:, :]  # (E, F, D)

            # Normalize each frame independently
            E, F, D = amp_obs.shape
            amp_flat = self.obs_normalizer(amp_obs.reshape(-1, D)).reshape(E, F, D)
            disc_input = amp_flat.flatten(1, 2)  # (E, F*D)

            # Append condition embedding if conditional
            if self.conditional and self.condition_obs_group is not None:
                cond_id = obs[self.condition_obs_group].squeeze(-1).long()  # (E,)
                cond_emb = self.condition_embedding(cond_id)  # (E, emb_dim)
                disc_input = torch.cat([disc_input, cond_emb], dim=-1)

            # Discriminator inference
            was_training = self.disc_trunk.training
            self.disc_trunk.eval()
            self.disc_linear.eval()
            disc_score = self._disc_forward(disc_input).squeeze(-1)  # (E,)
            if was_training:
                self.disc_trunk.train()
                self.disc_linear.train()

            # Compute reward based on loss type
            style_rew = self._score_to_reward(disc_score)
            style_reward = step_dt * self.style_reward_scale * style_rew

            # Update WGAN output normalizer
            if self.loss_type == LossType.WGAN and was_training:
                self.output_normalizer.update(disc_score.unsqueeze(-1))

        return style_reward, disc_score

    def combine_reward(self, task_reward: torch.Tensor, style_reward: torch.Tensor) -> torch.Tensor:
        """Combine task and style rewards additively.

        style_reward already includes step_dt * style_reward_scale from reward().
        """
        return task_reward + style_reward

    # ------------------------------------------------------------------
    # Discriminator update (called after PPO update)
    # ------------------------------------------------------------------

    def update(
        self,
        storage: RolloutStorage,
        rl_learning_rate: float | None = None,
    ) -> dict[str, float]:
        """Train the discriminator using policy rollout and offline data.

        Extracts valid sequences from the rollout storage (avoiding episode
        boundaries) and from the pre-built offline sequences. Trains with
        LSGAN / GAN / WGAN loss plus gradient penalty.

        Must be called AFTER ``alg.update()`` — the storage data is still
        accessible even though ``storage.step`` has been reset to 0.

        Args:
            storage: RolloutStorage (fully filled from the last rollout).
            rl_learning_rate: Current policy learning rate for lr_scale.

        Returns:
            Dictionary of logged AMP metrics.
        """
        # Save obs buffer for next rollout's sequence bridging
        num_steps = storage.num_transitions_per_env
        self.obs_buffer = storage.observations[self.obs_group][
            num_steps - self.num_frames + 1 : num_steps
        ].clone()

        if self.conditional and self.condition_obs_group is not None:
            self.cond_buffer = storage.observations[self.condition_obs_group][
                num_steps - self.num_frames + 1 : num_steps
            ].clone()

        # Adjust learning rate
        if self.lr_scale is not None and rl_learning_rate is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = rl_learning_rate * self.lr_scale

        # Extract policy sequences from rollout storage
        policy_obs = storage.observations[self.obs_group]  # (S, E, D)
        policy_dones = storage.dones  # (S, E, 1)
        policy_seqs = self._extract_sequences_from_rollout(policy_obs, policy_dones)  # (P, F, D)

        # Extract policy condition IDs if conditional
        policy_cond_ids: torch.Tensor | None = None
        if self.conditional and self.condition_obs_group is not None:
            policy_cond = storage.observations[self.condition_obs_group]  # (S, E, 1)
            policy_cond_seqs = self._extract_sequences_from_rollout(policy_cond.float(), policy_dones)
            policy_cond_ids = policy_cond_seqs[:, -1, 0].long()  # (P,)

        # Offline sequences (pre-built)
        offline_seqs = self._offline_sequences  # (M, F, D)
        offline_cond_ids = self._offline_cond_ids  # (M,) or None

        if policy_seqs.shape[0] == 0:
            return {"amp/disc_loss": 0.0, "amp/grad_penalty": 0.0}

        # Mini-batch training
        num_policy = policy_seqs.shape[0]
        num_offline = offline_seqs.shape[0]
        mini_batch_size = min(
            num_policy // self.num_mini_batches,
            num_offline // self.num_mini_batches,
        )
        if mini_batch_size == 0:
            mini_batch_size = min(num_policy, num_offline)

        mean_disc_loss = 0.0
        mean_grad_penalty = 0.0
        mean_disc_score = 0.0
        mean_disc_demo_score = 0.0
        num_updates = 0

        for _ in range(self.num_learning_epochs):
            policy_perm = torch.randperm(num_policy, device=self.device)
            offline_perm = torch.randperm(num_offline, device=self.device)

            for i in range(self.num_mini_batches):
                start = i * mini_batch_size
                end = start + mini_batch_size
                if end > num_policy or end > num_offline:
                    break

                p_idx = policy_perm[start:end]
                o_idx = offline_perm[start:end]

                # If conditional, match offline samples to policy conditions
                if self.conditional and policy_cond_ids is not None and offline_cond_ids is not None:
                    o_idx = self._match_conditions(
                        policy_cond_ids[p_idx], offline_cond_ids, mini_batch_size
                    )

                p_batch = policy_seqs[p_idx]  # (B, F, D)
                o_batch = offline_seqs[o_idx]  # (B, F, D)

                # Update normalizer
                B, F, D = p_batch.shape
                self.obs_normalizer.update(p_batch.reshape(-1, D))
                self.obs_normalizer.update(o_batch.reshape(-1, D))

                # Add training noise
                if self.noise_scale is not None:
                    p_batch = p_batch + (2 * torch.rand_like(p_batch) - 1) * self.noise_scale
                    o_batch = o_batch + (2 * torch.rand_like(o_batch) - 1) * self.noise_scale

                # Normalize
                p_normed = self.obs_normalizer(p_batch.reshape(-1, D)).reshape(B, F, D)
                o_normed = self.obs_normalizer(o_batch.reshape(-1, D)).reshape(B, F, D)

                # Flatten frames
                p_flat = p_normed.flatten(1, 2)  # (B, F*D)
                o_flat = o_normed.flatten(1, 2)

                # Append condition embeddings
                if self.conditional and self.condition_embedding is not None:
                    p_cond_emb = self.condition_embedding(policy_cond_ids[p_idx])
                    o_cond_emb = self.condition_embedding(offline_cond_ids[o_idx] if offline_cond_ids is not None else policy_cond_ids[p_idx])
                    p_flat = torch.cat([p_flat, p_cond_emb], dim=-1)
                    o_flat = torch.cat([o_flat, o_cond_emb], dim=-1)

                # Discriminator forward
                policy_d = self._disc_forward(p_flat)
                offline_d = self._disc_forward(o_flat)

                # Compute loss
                disc_loss = self._compute_disc_loss(policy_d, offline_d)
                grad_pen = self._grad_penalty(o_flat)
                total_loss = disc_loss + self.grad_penalty_scale * grad_pen

                # Backward + step
                self.optimizer.zero_grad()
                total_loss.backward()

                if self.is_multi_gpu:
                    self._reduce_gradients()

                nn.utils.clip_grad_norm_(self._all_parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_disc_loss += disc_loss.item()
                mean_grad_penalty += grad_pen.item()
                mean_disc_score += policy_d.mean().item()
                mean_disc_demo_score += offline_d.mean().item()
                num_updates += 1

        if num_updates > 0:
            mean_disc_loss /= num_updates
            mean_grad_penalty /= num_updates
            mean_disc_score /= num_updates
            mean_disc_demo_score /= num_updates

        return {
            "amp/disc_loss": mean_disc_loss,
            "amp/grad_penalty": mean_grad_penalty,
            "amp/disc_score": mean_disc_score,
            "amp/disc_demo_score": mean_disc_demo_score,
        }

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def _score_to_reward(self, disc_score: torch.Tensor) -> torch.Tensor:
        """Convert discriminator score to style reward."""
        if self.loss_type == LossType.GAN:
            prob = torch.sigmoid(disc_score)
            return -torch.log(torch.clamp(1 - prob, min=1e-6))
        elif self.loss_type == LossType.LSGAN:
            return torch.clamp(1 - 0.25 * (disc_score - 1).pow(2), min=0)
        elif self.loss_type == LossType.WGAN:
            return self.output_normalizer(disc_score.unsqueeze(-1)).squeeze(-1)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _compute_disc_loss(
        self, policy_d: torch.Tensor, offline_d: torch.Tensor
    ) -> torch.Tensor:
        """Compute discriminator loss based on loss type."""
        if self.loss_type == LossType.GAN:
            bce = nn.BCEWithLogitsLoss()
            policy_loss = bce(policy_d, torch.zeros_like(policy_d))
            demo_loss = bce(offline_d, torch.ones_like(offline_d))
            return 0.5 * (policy_loss + demo_loss)
        elif self.loss_type == LossType.LSGAN:
            policy_loss = nn.MSELoss()(policy_d, -torch.ones_like(policy_d))
            demo_loss = nn.MSELoss()(offline_d, torch.ones_like(offline_d))
            return 0.5 * (policy_loss + demo_loss)
        elif self.loss_type == LossType.WGAN:
            return -offline_d.mean() + policy_d.mean()
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _grad_penalty(self, data: torch.Tensor) -> torch.Tensor:
        """Gradient penalty: penalize gradient norm deviating from 0."""
        data = data.clone().detach().requires_grad_(True)
        disc = self._disc_forward(data)
        grad = autograd.grad(
            outputs=disc,
            inputs=data,
            grad_outputs=torch.ones_like(disc),
            create_graph=True,
            only_inputs=True,
        )[0]
        return (grad.norm(2, dim=1)).pow(2).mean()

    # ------------------------------------------------------------------
    # Sequence extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sequences_from_flat(
        data: torch.Tensor,
        dones: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """Extract valid consecutive sequences from flat (N, D) data.

        A sequence is invalid if any done flag is True within the window
        (except possibly the last frame).

        Args:
            data: (N, D) flat tensor of feature frames.
            dones: (N,) boolean tensor, True at episode boundaries.
            num_frames: Window length.

        Returns:
            (M, num_frames, D) tensor of valid sequences.
        """
        N = data.shape[0]
        if N < num_frames:
            return data.unsqueeze(0)[:0]  # empty with correct dims

        sequences = []
        for start in range(N - num_frames + 1):
            # Check that no done occurs before the last frame of the window
            window_dones = dones[start : start + num_frames - 1]
            if not window_dones.any():
                sequences.append(data[start : start + num_frames])

        if not sequences:
            return data.unsqueeze(0)[:0]
        return torch.stack(sequences)

    def _extract_sequences_from_rollout(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Extract valid sequences from rollout storage (S, E, D).

        Builds all valid num_frames-length windows across both the step
        and environment dimensions, filtering out any window that spans
        an episode boundary. Fully vectorized — no Python loops over envs.

        Args:
            obs: (S, E, D) observations from rollout storage.
            dones: (S, E, 1) done flags.

        Returns:
            (M, num_frames, D) tensor of valid sequences.
        """
        S, E, D = obs.shape
        F = self.num_frames
        if S < F:
            return torch.empty(0, F, D, device=self.device)

        dones_2d = dones.squeeze(-1)  # (S, E)

        # Build a (S-F+1, E) mask: True if window [t, t+F) has no done in [t, t+F-1)
        # Use a cumulative sum approach for efficient windowed-any check
        done_cum = dones_2d.float().cumsum(dim=0)  # (S, E)
        # Number of dones in window [t, t+F-1) = cum[t+F-2] - cum[t-1]
        # (with proper boundary handling)
        num_windows = S - F + 1
        end_cum = done_cum[F - 2 : F - 2 + num_windows]  # (num_windows, E)
        if F >= 2:
            # Prepend a zero row for t=0 (no preceding dones)
            start_cum = torch.cat([
                torch.zeros(1, E, device=self.device),
                done_cum[: num_windows - 1]
            ], dim=0)  # (num_windows, E)
        else:
            start_cum = torch.zeros(num_windows, E, device=self.device)
        window_done_count = end_cum - start_cum  # (num_windows, E)
        valid_mask = window_done_count == 0  # (num_windows, E)

        # Gather valid (t, env) pairs
        valid_t, valid_e = torch.nonzero(valid_mask, as_tuple=True)  # each (M,)

        if valid_t.numel() == 0:
            return torch.empty(0, F, D, device=self.device)

        # Build frame indices: for each valid window, gather F consecutive frames
        frame_offsets = torch.arange(F, device=self.device)  # (F,)
        frame_indices = valid_t.unsqueeze(1) + frame_offsets.unsqueeze(0)  # (M, F)
        env_indices = valid_e.unsqueeze(1).expand(-1, F)  # (M, F)

        sequences = obs[frame_indices, env_indices]  # (M, F, D)
        return sequences

    def _match_conditions(
        self,
        target_conds: torch.Tensor,
        offline_conds: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """For conditional AMP: sample offline indices matching target conditions (vectorized)."""
        if not hasattr(self, "_cond_index_map") or self._cond_index_map is None:
            self._rebuild_cond_index_map(offline_conds)

        indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        matched = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for cond_val, pool in self._cond_index_map.items():
            mask = target_conds == cond_val
            count = mask.sum().item()
            if count == 0:
                continue
            indices[mask] = pool[torch.randint(pool.numel(), (count,), device=self.device)]
            matched |= mask

        if not matched.all():
            fallback = torch.arange(len(offline_conds), device=self.device)
            n = (~matched).sum().item()
            indices[~matched] = fallback[torch.randint(fallback.numel(), (n,), device=self.device)]

        return indices

    def _rebuild_cond_index_map(self, offline_conds: torch.Tensor) -> None:
        """Pre-build per-condition index pools for fast vectorized sampling."""
        self._cond_index_map = {}
        for cond_val in offline_conds.unique().tolist():
            self._cond_index_map[cond_val] = torch.nonzero(
                offline_conds == cond_val, as_tuple=False
            ).squeeze(-1)

    # ------------------------------------------------------------------
    # Multi-GPU
    # ------------------------------------------------------------------

    def _all_parameters(self):
        """All trainable parameters."""
        params = list(self.disc_trunk.parameters()) + list(self.disc_linear.parameters())
        if self.condition_embedding is not None:
            params += list(self.condition_embedding.parameters())
        return params

    def broadcast_parameters(self) -> None:
        """Broadcast discriminator parameters from rank 0 to all GPUs."""
        if not self.is_multi_gpu:
            return
        state = [self.state_dict()]
        torch.distributed.broadcast_object_list(state, src=0)
        self.load_state_dict(state[0])

    def _reduce_gradients(self) -> None:
        """All-reduce gradients across GPUs."""
        grads = [p.grad.view(-1) for p in self._all_parameters() if p.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for p in self._all_parameters():
            if p.grad is not None:
                numel = p.numel()
                p.grad.data.copy_(all_grads[offset : offset + numel].view_as(p.grad.data))
                offset += numel

    def sync_normalizer(self) -> None:
        """Synchronize normalizer statistics across GPUs."""
        if not self.is_multi_gpu:
            return
        for buf in [self.obs_normalizer.running_mean, self.obs_normalizer.running_var, self.obs_normalizer.count]:
            torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
            buf /= self.gpu_world_size

    # ------------------------------------------------------------------
    # State dict
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return complete state for checkpointing."""
        state = {
            "disc_trunk": self.disc_trunk.state_dict(),
            "disc_linear": self.disc_linear.state_dict(),
            "obs_normalizer": self.obs_normalizer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.condition_embedding is not None:
            state["condition_embedding"] = self.condition_embedding.state_dict()
        if self.loss_type == LossType.WGAN:
            state["output_normalizer"] = self.output_normalizer.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        self.disc_trunk.load_state_dict(state["disc_trunk"])
        self.disc_linear.load_state_dict(state["disc_linear"])
        self.obs_normalizer.load_state_dict(state["obs_normalizer"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.condition_embedding is not None and "condition_embedding" in state:
            self.condition_embedding.load_state_dict(state["condition_embedding"])
        if self.loss_type == LossType.WGAN and "output_normalizer" in state:
            self.output_normalizer.load_state_dict(state["output_normalizer"])

    # ------------------------------------------------------------------
    # Train / eval mode
    # ------------------------------------------------------------------

    def train(self) -> None:
        self.disc_trunk.train()
        self.disc_linear.train()
        self.obs_normalizer.train()
        if self.condition_embedding is not None:
            self.condition_embedding.train()

    def eval(self) -> None:
        self.disc_trunk.eval()
        self.disc_linear.eval()
        self.obs_normalizer.eval()
        if self.condition_embedding is not None:
            self.condition_embedding.eval()
