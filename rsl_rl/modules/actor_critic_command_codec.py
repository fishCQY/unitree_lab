from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


def _straight_through_round(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through gradients."""
    return x + (torch.round(x) - x).detach()


class FSQQuantizer(nn.Module):
    """Finite Scalar Quantization (FSQ) with straight-through estimator.

    This implementation quantizes each dimension independently to a fixed number of
    uniformly spaced levels in [-1, 1].
    """

    def __init__(self, levels: int = 8) -> None:
        super().__init__()
        if levels < 2:
            raise ValueError("FSQ levels must be >= 2")
        self.levels = int(levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map x to [-1, 1] (tanh bounds), then to [0, levels-1], round, and map back.
        x_bounded = torch.tanh(x)
        scale = (self.levels - 1) / 2.0
        x_scaled = (x_bounded + 1.0) * scale  # [0, levels-1]
        x_q = _straight_through_round(x_scaled)
        x_dequant = x_q / scale - 1.0
        return x_dequant


class RFSQQuantizer(nn.Module):
    """Residual FSQ: apply FSQ on residual multiple times and sum."""

    def __init__(self, levels: int = 8, num_stages: int = 2) -> None:
        super().__init__()
        if num_stages < 1:
            raise ValueError("RFSQ num_stages must be >= 1")
        self.num_stages = int(num_stages)
        self.stages = nn.ModuleList([FSQQuantizer(levels=levels) for _ in range(self.num_stages)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        r = x
        for q in self.stages:
            q_r = q(r)
            y = y + q_r
            r = r - q_r
        return y


class VectorQuantizer(nn.Module):
    """VQ codebook with straight-through estimator (simple, gradient-updated)."""

    def __init__(self, codebook_size: int, dim: int, commitment_cost: float = 0.25) -> None:
        super().__init__()
        if codebook_size < 2:
            raise ValueError("codebook_size must be >= 2")
        self.codebook_size = int(codebook_size)
        self.dim = int(dim)
        self.commitment_cost = float(commitment_cost)
        self.embedding = nn.Embedding(self.codebook_size, self.dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Flatten to (B, D)
        z = z_e.view(-1, self.dim)
        # Compute squared L2 distance to embeddings: ||z - e||^2
        e = self.embedding.weight
        # (B, 1) + (1, K) - 2 (B, K)
        z2 = torch.sum(z * z, dim=1, keepdim=True)
        e2 = torch.sum(e * e, dim=1).unsqueeze(0)
        ze = torch.matmul(z, e.t())
        dist = z2 + e2 - 2.0 * ze
        indices = torch.argmin(dist, dim=1)
        z_q = self.embedding(indices).view_as(z_e)
        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        # VQ losses (to be added externally)
        loss = torch.mean((z_q.detach() - z_e) ** 2) + self.commitment_cost * torch.mean((z_q - z_e.detach()) ** 2)
        return z_q_st, loss


class _ActorCriticBase(nn.Module):
    """Actor-critic base compatible with rsl_rl runners (non-recurrent, 1D obs)."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                f"{self.__class__.__name__}.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups = obs_groups

        # Compute input dims (implemented by subclasses)
        num_actor_obs, num_critic_obs = self._resolve_obs_dims(obs)

        # Actor / Critic MLPs
        self.state_dependent_std = state_dependent_std
        self.noise_std_type = noise_std_type

        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)

        # Normalizers
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()

        # Action noise params
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            mean = self.actor(obs)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        self._update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        if self.state_dependent_std:
            return self.actor(actor_obs)[..., 0, :]
        return self.actor(actor_obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Match rsl_rl runner contract (returns resumed_training flag)."""
        super().load_state_dict(state_dict, strict=strict)
        return True

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def _resolve_obs_dims(self, obs: TensorDict) -> tuple[int, int]:
        raise NotImplementedError

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[group] for group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[group] for group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)


@dataclass
class AuxLossInfo:
    total: torch.Tensor
    recon: torch.Tensor | None = None
    codebook: torch.Tensor | None = None


class ActorCriticCommandCodec(_ActorCriticBase):
    """ActorCritic that replaces a specific observation group with a learned latent.

    It supports optional auxiliary losses (reconstruction / codebook) that PPO can
    add to the overall objective.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        command_obs_key: str = "command",
        command_latent_dim: int = 64,
        command_encoder_hidden_dims: tuple[int] | list[int] = (256, 256),
        command_decoder_hidden_dims: tuple[int] | list[int] = (256, 256),
        aux_loss_coef: float = 0.0,
        **kwargs: dict[str, Any],
    ) -> None:
        self.command_obs_key = str(command_obs_key)
        self.command_latent_dim = int(command_latent_dim)
        self.aux_loss_coef = float(aux_loss_coef)

        if self.command_obs_key not in obs.keys():
            raise KeyError(f"Observation group '{self.command_obs_key}' not found in env observations.")
        self.command_in_dim = int(obs[self.command_obs_key].shape[-1])

        # command encoder/decoder (initialized before base to allow dim resolution)
        self.command_encoder = MLP(self.command_in_dim, self.command_latent_dim, command_encoder_hidden_dims, "elu")
        self.command_decoder = MLP(self.command_latent_dim, self.command_in_dim, command_decoder_hidden_dims, "elu")
        self._last_aux: AuxLossInfo | None = None

        super().__init__(obs, obs_groups, num_actions, **kwargs)

    # encoding path (override for quantized variants)
    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, AuxLossInfo | None]:
        z = self.command_encoder(x)
        if self.aux_loss_coef > 0.0:
            x_hat = self.command_decoder(z)
            recon = torch.mean((x_hat - x) ** 2)
            return z, AuxLossInfo(total=recon, recon=recon)
        return z, None

    def auxiliary_loss(self, obs: TensorDict) -> torch.Tensor | None:
        if self.aux_loss_coef <= 0.0:
            return None
        x = obs[self.command_obs_key]
        _, info = self._encode(x)
        if info is None:
            return None
        return info.total

    def _resolve_obs_dims(self, obs: TensorDict) -> tuple[int, int]:
        num_actor_obs = 0
        for group in self.obs_groups["policy"]:
            assert len(obs[group].shape) == 2, "Only 1D observations supported."
            num_actor_obs += self.command_latent_dim if group == self.command_obs_key else obs[group].shape[-1]
        num_critic_obs = 0
        for group in self.obs_groups["critic"]:
            assert len(obs[group].shape) == 2, "Only 1D observations supported."
            num_critic_obs += self.command_latent_dim if group == self.command_obs_key else obs[group].shape[-1]
        return int(num_actor_obs), int(num_critic_obs)

    def _replace_group_with_latent(self, obs: TensorDict, groups: list[str]) -> torch.Tensor:
        obs_list: list[torch.Tensor] = []
        for group in groups:
            if group == self.command_obs_key:
                z, _ = self._encode(obs[group])
                obs_list.append(z)
            else:
                obs_list.append(obs[group])
        return torch.cat(obs_list, dim=-1)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._replace_group_with_latent(obs, self.obs_groups["policy"])

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._replace_group_with_latent(obs, self.obs_groups["critic"])


class ActorCriticCommandFSQ(ActorCriticCommandCodec):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        fsq_levels: int = 8,
        **kwargs: dict[str, Any],
    ) -> None:
        self.fsq = FSQQuantizer(levels=fsq_levels)
        super().__init__(obs, obs_groups, num_actions, **kwargs)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, AuxLossInfo | None]:
        z_e = self.command_encoder(x)
        z_q = self.fsq(z_e)
        if self.aux_loss_coef > 0.0:
            x_hat = self.command_decoder(z_q)
            recon = torch.mean((x_hat - x) ** 2)
            return z_q, AuxLossInfo(total=recon, recon=recon)
        return z_q, None


class ActorCriticCommandRFSQ(ActorCriticCommandCodec):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        fsq_levels: int = 8,
        rfsq_stages: int = 2,
        **kwargs: dict[str, Any],
    ) -> None:
        self.rfsq = RFSQQuantizer(levels=fsq_levels, num_stages=rfsq_stages)
        super().__init__(obs, obs_groups, num_actions, **kwargs)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, AuxLossInfo | None]:
        z_e = self.command_encoder(x)
        z_q = self.rfsq(z_e)
        if self.aux_loss_coef > 0.0:
            x_hat = self.command_decoder(z_q)
            recon = torch.mean((x_hat - x) ** 2)
            return z_q, AuxLossInfo(total=recon, recon=recon)
        return z_q, None


class ActorCriticCommandVQVAE(ActorCriticCommandCodec):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        vq_codebook_size: int = 512,
        vq_commitment_cost: float = 0.25,
        **kwargs: dict[str, Any],
    ) -> None:
        # VectorQuantizer expects latent dim; we create it after command_latent_dim known
        self._vq_codebook_size = int(vq_codebook_size)
        self._vq_commitment_cost = float(vq_commitment_cost)
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        self.vq = VectorQuantizer(self._vq_codebook_size, self.command_latent_dim, self._vq_commitment_cost)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, AuxLossInfo | None]:
        z_e = self.command_encoder(x)
        z_q, vq_loss = self.vq(z_e)
        if self.aux_loss_coef > 0.0:
            x_hat = self.command_decoder(z_q)
            recon = torch.mean((x_hat - x) ** 2)
            total = recon + vq_loss
            return z_q, AuxLossInfo(total=total, recon=recon, codebook=vq_loss)
        return z_q, AuxLossInfo(total=vq_loss, codebook=vq_loss)

    def auxiliary_loss(self, obs: TensorDict) -> torch.Tensor | None:
        if self.aux_loss_coef <= 0.0:
            return None
        x = obs[self.command_obs_key]
        _, info = self._encode(x)
        if info is None:
            return None
        return info.total

