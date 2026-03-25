# Copyright (c) 2024-2026, Light Robotics.
# All rights reserved.

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules.actor_critic_transformer_base import ActorCriticTransformerBase
from rsl_rl.networks import MLP
from rsl_rl.networks.temporal_transformer_encoder import TemporalTransformerEncoder


class ActorCriticTransformer(ActorCriticTransformerBase):
    """Stateless sliding-window Transformer actor-critic.

    Designed as a drop-in alternative to ActorCritic (MLP-based).
    Treats the Transformer as a stateless function (like MLP) with richer input:
    a sliding window of observation frames, tokenized and processed by a Transformer.

    - is_recurrent = False (standard PPO flow, no hidden states)
    - is_transformer = True (triggers dedicated mini-batch generator in PPO)
    - During inference: maintains internal obs buffer (sliding window)
    - During training: receives pre-windowed observations from transformer_mini_batch_generator
    """

    # ------------------------------------------------------------------
    # Encoder construction (called by base class __init__)
    # ------------------------------------------------------------------

    def _build_encoders(
        self,
        obs: TensorDict,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 512,
        num_layers: int = 3,
        activation: str = "elu",
        token_groups: list[list[str]] | None = None,
        gradient_checkpointing: bool = False,
        use_gru_gating: bool = False,
        gru_bias: float = 2.0,
        critic_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        **kwargs,
    ) -> None:
        if kwargs:
            print(
                "ActorCriticTransformer._build_encoders got unexpected arguments, which will be ignored: "
                + str(list(kwargs))
            )

        self.token_groups = token_groups

        # Resolve token_groups into dimension lists
        actor_token_group_dims = None
        critic_token_group_dims = None
        if token_groups is not None:
            # Validate that token_groups cover all policy obs groups
            all_groups_in_tokens = [g for tg in token_groups for g in tg]
            assert set(all_groups_in_tokens) == set(self.obs_groups["policy"]), (
                "token_groups must cover exactly the policy obs groups. "
                f"Got {all_groups_in_tokens}, expected {self.obs_groups['policy']}"
            )
            # Compute dimensions for each token group
            actor_token_group_dims = []
            for tg in token_groups:
                dim = sum(obs[g].shape[-1] for g in tg)
                actor_token_group_dims.append(dim)
            # For critic with transformer: use same token_groups if groups match, else single token
            if self.use_transformer_critic:
                critic_groups_set = set(self.obs_groups["critic"])
                token_groups_set = set(all_groups_in_tokens)
                if critic_groups_set == token_groups_set:
                    critic_token_group_dims = actor_token_group_dims
                # Otherwise, critic uses default single-token mode

        # Actor Transformer
        self.actor_transformer = TemporalTransformerEncoder(
            obs_dim=num_actor_obs,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            frames_per_token=self.frames_per_token,
            token_group_dims=actor_token_group_dims,
            gradient_checkpointing=gradient_checkpointing,
            use_gru_gating=use_gru_gating,
            gru_bias=gru_bias,
        )
        self.actor_output = nn.Linear(embed_dim, num_actions)
        print(
            f"Actor Transformer: obs_dim={num_actor_obs}, embed_dim={embed_dim}, "
            f"heads={num_heads}, layers={num_layers}, context_len={self.context_len}, "
            f"token_types={self.actor_transformer.num_token_types}, "
            f"frames_per_token={self.frames_per_token}, "
            f"gru_gating={use_gru_gating}"
        )

        # Critic
        if self.use_transformer_critic:
            self.critic_transformer = TemporalTransformerEncoder(
                obs_dim=num_critic_obs,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_layers=num_layers,
                frames_per_token=self.frames_per_token,
                token_group_dims=critic_token_group_dims,
                gradient_checkpointing=gradient_checkpointing,
                use_gru_gating=use_gru_gating,
                gru_bias=gru_bias,
            )
            self.critic_output = nn.Linear(embed_dim, 1)
            print(
                f"Critic Transformer: obs_dim={num_critic_obs}, embed_dim={embed_dim}, "
                f"heads={num_heads}, layers={num_layers}, gru_gating={use_gru_gating}"
            )
        else:
            self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
            print(f"Critic MLP: {self.critic}")

        # Internal observation buffers for inference (initialized lazily)
        self._actor_buffer: torch.Tensor | None = None
        self._critic_buffer: torch.Tensor | None = None
        # Per-env count of valid (non-stale) frames in the buffer.
        # Reset to 0 on episode done so stale zeros get masked out.
        self._actor_valid_len: torch.Tensor | None = None
        self._critic_valid_len: torch.Tensor | None = None

        # Print parameter counts
        actor_params = sum(p.numel() for p in self.actor_transformer.parameters()) + sum(
            p.numel() for p in self.actor_output.parameters()
        )
        if self.use_transformer_critic:
            critic_params = sum(p.numel() for p in self.critic_transformer.parameters()) + sum(
                p.numel() for p in self.critic_output.parameters()
            )
        else:
            critic_params = sum(p.numel() for p in self.critic.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Actor params: {actor_params:,} | Critic params: {critic_params:,} | Total params: {total_params:,}")

    # --------------------------------------------------------------------------
    # Internal buffer management (inference only)
    # --------------------------------------------------------------------------

    def _update_buffer(self, obs_flat: torch.Tensor, buffer_attr: str) -> None:
        """Append single-frame obs to internal sliding window buffer.

        Also maintains a per-env valid-length counter so that stale zeros
        (from episode resets) can be masked out later.

        Args:
            obs_flat: (num_envs, obs_dim) single-frame observation.
            buffer_attr: attribute name, either '_actor_buffer' or '_critic_buffer'.
        """
        buf = getattr(self, buffer_attr)
        valid_attr = buffer_attr.replace("_buffer", "_valid_len")
        frame = obs_flat.unsqueeze(1)  # (num_envs, 1, obs_dim)
        if buf is None:
            setattr(self, buffer_attr, frame)
            setattr(self, valid_attr, torch.ones(obs_flat.shape[0], dtype=torch.long, device=obs_flat.device))
        else:
            buf = torch.cat([buf, frame], dim=1)
            if buf.shape[1] > self.context_len:
                buf = buf[:, -self.context_len :]
            setattr(self, buffer_attr, buf)
            vlen = getattr(self, valid_attr)
            setattr(self, valid_attr, (vlen + 1).clamp(max=buf.shape[1]))

    def _get_buffer(self, buffer_attr: str) -> torch.Tensor:
        """Get current buffer contents."""
        return getattr(self, buffer_attr)

    def _get_obs_window(
        self, obs: TensorDict, obs_getter: str, buffer_attr: str
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get observation window and optional mask, handling both inference and training modes.

        Args:
            obs: Observation TensorDict.
            obs_getter: 'get_actor_obs' or 'get_critic_obs'.
            buffer_attr: '_actor_buffer' or '_critic_buffer'.

        Returns:
            obs_window: (batch, seq_len, obs_dim)
            window_mask: (batch, seq_len) bool or None
        """
        obs_flat = getattr(self, obs_getter)(obs)

        if obs_flat.ndim == 2:
            # Inference mode: single frame → update variable-length buffer
            self._update_buffer(obs_flat, buffer_attr)
            obs_window = self._get_buffer(buffer_attr)
            # Build validity mask from per-env valid_len.
            # Valid frames are the rightmost valid_len entries (most recent).
            # After flip(1) in the encoder, these become the leftmost positions.
            valid_attr = buffer_attr.replace("_buffer", "_valid_len")
            valid_len = getattr(self, valid_attr)  # (num_envs,)
            seq_len = obs_window.shape[1]
            # When all envs are fully valid, skip mask to enable Flash Attention
            if (valid_len == seq_len).all():
                return obs_window, None
            indices = torch.arange(seq_len, device=obs_window.device)  # (seq_len,)
            window_mask = indices.unsqueeze(0) >= (seq_len - valid_len.unsqueeze(1))  # (num_envs, seq_len)
            return obs_window, window_mask
        elif obs_flat.ndim == 3:
            # Training mode: already windowed (batch, context_len, obs_dim)
            # Check for __window_mask__ in TensorDict
            window_mask = None
            if "__window_mask__" in obs.keys():
                wm = obs["__window_mask__"].bool()
                # When all positions are valid, skip mask to enable Flash Attention
                if not wm.all():
                    window_mask = wm
            return obs_flat, window_mask
        else:
            raise ValueError(f"Unexpected obs dimensionality: {obs_flat.ndim}")

    # --------------------------------------------------------------------------
    # Abstract method implementations
    # --------------------------------------------------------------------------

    def _actor_forward(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        obs_window, window_mask = self._get_obs_window(obs, "get_actor_obs", "_actor_buffer")
        obs_window = self.actor_obs_normalizer(obs_window)
        transformer_output = self.actor_transformer(obs_window, window_mask)
        return self.actor_output(transformer_output)

    def _critic_forward(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        if self.use_transformer_critic:
            obs_window, window_mask = self._get_obs_window(obs, "get_critic_obs", "_critic_buffer")
            obs_window = self.critic_obs_normalizer(obs_window)
            transformer_output = self.critic_transformer(obs_window, window_mask)
            return self.critic_output(transformer_output)
        else:
            # MLP critic: use flat obs (handles mixed windowed/single-frame groups)
            obs_flat = self.get_critic_obs_flat(obs)
            obs_flat = self.critic_obs_normalizer(obs_flat)
            return self.critic(obs_flat)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset internal buffers for done environments."""
        if dones is not None and (self._actor_buffer is not None or self._critic_buffer is not None):
            done_mask = dones.reshape(-1).bool()
            if done_mask.any():
                if self._actor_buffer is not None:
                    self._actor_buffer[done_mask] = 0.0
                if self._actor_valid_len is not None:
                    self._actor_valid_len[done_mask] = 0
                if self._critic_buffer is not None:
                    self._critic_buffer[done_mask] = 0.0
                if self._critic_valid_len is not None:
                    self._critic_valid_len[done_mask] = 0
