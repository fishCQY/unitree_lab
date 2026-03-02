"""PPO runner configurations for locomotion tasks (velocity tracking + AMP)."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg as _RslRlPpoAlgorithmCfg,
    RslRlPpoActorCriticRecurrentCfg,
)


@configclass
class ActorCriticDepthCfg(RslRlPpoActorCriticCfg):
    """Actor-critic with a depth image encoder branch.

    Extends the standard MLP actor-critic by adding a CNN/MLP encoder that
    processes depth images and outputs a latent vector concatenated with
    the proprioceptive observation before entering the actor MLP.
    """

    class_name: str = "ActorCriticDepth"

    depth_encoder_type: Literal["standard", "cnn"] = "standard"
    """Type of depth encoder. 'standard' uses an MLP, 'cnn' uses a ConvNet."""

    depth_latent_dim: int = 64
    """Dimension of the latent vector produced by the depth encoder."""

    depth_encoder_hidden_dims: list[int] = MISSING
    """Hidden layer dimensions for the depth encoder network."""


@configclass
class AMPDiscriminatorCfg:
    """Sub-config passed as ``amp_discriminator`` to PPOAMP."""
    hidden_dims: list[int] = MISSING
    activation: str = "relu"
    style_reward_scale: float = 1.0
    task_style_lerp: float = 0.5


@configclass
class AMPCfg:
    """AMP configuration consumed by rsl_rl.algorithms.PPOAMP.

    Fields ``disc_obs_dim``, ``disc_obs_steps``, and ``step_dt`` are
    auto-resolved by ``resolve_amp_config`` at runtime — do NOT set them here.
    """
    loss_type: str = "LSGAN"
    disc_learning_rate: float = 5e-4
    disc_trunk_weight_decay: float = 1e-4
    disc_linear_weight_decay: float = 1e-2
    disc_max_grad_norm: float = 0.5
    grad_penalty_scale: float = 10.0
    disc_obs_buffer_size: int = 200
    amp_discriminator: AMPDiscriminatorCfg = MISSING


@configclass
class RslRlPpoAlgorithmCfg(_RslRlPpoAlgorithmCfg):
    amp_cfg: AMPCfg | None = None


# =========================================================================
# AMP Runner Configs (use rsl_rl AMPRunner)
# =========================================================================


@configclass
class UnitreeG1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "unitree_g1_rough"
    empirical_normalization = True
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc_agent"],
        "discriminator_demonstration": ["disc_demo"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=AMPCfg(
            loss_type="LSGAN",
            disc_learning_rate=5e-4,
            disc_trunk_weight_decay=1e-4,
            disc_linear_weight_decay=1e-2,
            disc_max_grad_norm=0.5,
            grad_penalty_scale=10.0,
            disc_obs_buffer_size=200,
            amp_discriminator=AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="relu",
                style_reward_scale=1.0,
                task_style_lerp=0.5,
            ),
        ),
    )


@configclass
class UnitreeG1FlatPPORunnerCfg(UnitreeG1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 60000
        self.experiment_name = "unitree_g1_flat"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class UnitreeG1RoughPPORunnerGRUCfg(UnitreeG1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_g1_rough_gru"
        self.policy = RslRlPpoActorCriticRecurrentCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            rnn_type="gru",
            rnn_hidden_dim=512,
            rnn_num_layers=1,
        )


@configclass
class UnitreeG1RoughDepthPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO Runner with depth image encoder for visual locomotion."""

    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "unitree_g1_rough_depth"
    empirical_normalization = True
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "image": ["image"],
        "discriminator": ["disc_agent"],
        "discriminator_demonstration": ["disc_demo"],
    }
    policy = ActorCriticDepthCfg(
        init_noise_std=1.0,
        depth_encoder_type="standard",
        depth_latent_dim=64,
        depth_encoder_hidden_dims=[32, 64, 128],
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=AMPCfg(
            loss_type="LSGAN",
            disc_learning_rate=5e-4,
            disc_trunk_weight_decay=1e-4,
            disc_linear_weight_decay=1e-2,
            disc_max_grad_norm=0.5,
            grad_penalty_scale=10.0,
            disc_obs_buffer_size=200,
            amp_discriminator=AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="relu",
                style_reward_scale=1.0,
                task_style_lerp=0.5,
            ),
        ),
    )
