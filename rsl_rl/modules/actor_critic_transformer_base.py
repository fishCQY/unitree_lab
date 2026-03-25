# Copyright (c) 2024-2026, Light Robotics.
# All rights reserved.

"""Abstract base class for all Transformer-based actor-critics.

Provides the shared interface and logic common to all Transformer actor-critic
variants (sliding-window, TXL, etc.):

- Observation group handling and dimension computation
- Observation normalization setup and updates
- Action noise (scalar / log) and Normal distribution management
- Template methods ``act`` / ``act_inference`` / ``evaluate`` that delegate
  to variant-specific ``_actor_forward`` / ``_critic_forward``

Does **NOT** handle buffer/memory management, encoder architecture, or output
heads — each variant implements these in ``_build_encoders``,
``_actor_forward``, ``_critic_forward``, and ``reset``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization


class ActorCriticTransformerBase(nn.Module, ABC):
    """Abstract base for all Transformer-based actor-critics.

    Handles: distribution, noise, normalization, obs groups, interface.
    Does NOT handle: buffers, memory, encoder architecture, output heads.
    """

    is_recurrent: bool = False
    is_transformer: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        context_len: int = 32,
        frames_per_token: int = 1,
        use_transformer_critic: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.obs_groups = obs_groups
        self.context_len = context_len
        self.use_transformer_critic = use_transformer_critic

        assert frames_per_token >= 1, f"frames_per_token must be >= 1, got {frames_per_token}"
        assert context_len % frames_per_token == 0, (
            f"context_len ({context_len}) must be divisible by frames_per_token ({frames_per_token})"
        )
        self.frames_per_token = frames_per_token

        # Determine which obs groups need windowing to save GPU memory.
        # - Policy groups always need windowing (actor Transformer input).
        # - Critic groups need windowing only if using a Transformer critic.
        # - All other groups (amp, symmetry, etc.) are kept as single-frame.
        _groups_to_window = set(obs_groups["policy"])
        if use_transformer_critic:
            _groups_to_window |= set(obs_groups["critic"])
        self.groups_to_window: set[str] = _groups_to_window

        # Compute observation dimensions
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticTransformer module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticTransformer module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in _update_distribution)
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

        # Build variant-specific encoders and output heads
        self._build_encoders(obs, num_actor_obs, num_critic_obs, num_actions, **kwargs)

    # --------------------------------------------------------------------------
    # Abstract methods — each variant must implement
    # --------------------------------------------------------------------------

    @abstractmethod
    def _build_encoders(
        self,
        obs: TensorDict,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        **kwargs,
    ) -> None:
        """Build encoders and output heads. Called at the end of ``__init__``.

        Args:
            obs: Sample observation TensorDict (for resolving token groups, etc.).
            num_actor_obs: Total actor observation dimension.
            num_critic_obs: Total critic observation dimension.
            num_actions: Number of actions.
            **kwargs: Variant-specific configuration (embed_dim, num_heads, etc.).
        """

    @abstractmethod
    def _actor_forward(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        """Full actor pipeline: obs -> encoder -> output head -> action mean.

        Each variant handles its own buffer/memory management and normalization
        application internally.

        Returns:
            (batch, num_actions) action mean tensor.
        """

    @abstractmethod
    def _critic_forward(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        """Full critic pipeline: obs -> encoder -> output head -> value.

        Each variant handles its own buffer/memory management and normalization
        application internally.

        Returns:
            (batch, 1) value prediction tensor.
        """

    @abstractmethod
    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset internal state for done environments.

        Args:
            dones: Boolean/float tensor indicating which environments are done.
        """

    # --------------------------------------------------------------------------
    # Observation helpers
    # --------------------------------------------------------------------------

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate all policy obs groups along last dim."""
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate all critic obs groups along last dim.

        Handles mixed-dimension TensorDicts where some groups are windowed (3D)
        and others are single-frame (2D) — used when only policy groups are windowed.
        """
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs_flat(self, obs: TensorDict) -> torch.Tensor:
        """Get single-frame critic obs, extracting last frame from any windowed groups."""
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            o = obs[obs_group]
            if o.ndim == 3:
                o = o[:, -1, :]  # take last frame from windowed group
            obs_list.append(o)
        return torch.cat(obs_list, dim=-1)

    # --------------------------------------------------------------------------
    # Distribution and noise
    # --------------------------------------------------------------------------

    def _get_std(self, mean: torch.Tensor) -> torch.Tensor:
        """Compute standard deviation based on noise_std_type."""
        if self.noise_std_type == "scalar":
            return self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            return torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

    def _update_distribution(self, mean: torch.Tensor) -> None:
        """Create Normal distribution from action mean."""
        std = self._get_std(mean)
        self.distribution = Normal(mean, std)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    # --------------------------------------------------------------------------
    # Core template methods
    # --------------------------------------------------------------------------

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        mean = self._actor_forward(obs, **kwargs)
        self._update_distribution(mean)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self._actor_forward(obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        return self._critic_forward(obs, **kwargs)

    # --------------------------------------------------------------------------
    # Normalization
    # --------------------------------------------------------------------------

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    # --------------------------------------------------------------------------
    # State dict
    # --------------------------------------------------------------------------

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        .. note:: **Checkpoint compatibility (RoPE refactor)**

            Checkpoints saved before the RoPE refactor use ``nn.MultiheadAttention``
            with combined ``attention.in_proj_weight/bias`` and a separate
            ``rope.inv_freq`` buffer.  The current architecture uses
            ``RoPEMultiHeadAttention`` with split ``q_proj/k_proj/v_proj`` weights
            and pre-cached ``cos_cache/sin_cache`` buffers.  Old checkpoints are
            **not** compatible and will raise on ``strict=True``.  Since the old
            RoPE application was incorrect (applied before Q/K projection), retraining
            from scratch is recommended.

        Returns:
            Whether this training resumes a previous training.
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
