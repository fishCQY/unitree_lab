"""PPO runner configurations for locomotion tasks (velocity tracking + AMP).

Uses the plugin-based AMPPluginRunner approach: PPO stays vanilla,
AMP discriminator is a standalone plugin that manages its own training loop.
Data is loaded via env.cfg.load_amp_data() (conditional LAFAN walk/run).
"""

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
    """Actor-critic with a depth image encoder branch."""

    class_name: str = "ActorCriticDepth"
    depth_encoder_type: Literal["standard", "cnn"] = "standard"
    depth_latent_dim: int = 64
    depth_encoder_hidden_dims: list[int] = MISSING


# =========================================================================
# AMPPlugin configs (standalone discriminator plugin for AMPPluginRunner)
# =========================================================================


@configclass
class AMPPluginCfg:
    """Configuration for the AMPPlugin (standalone discriminator plugin).

    Used with AMPPluginRunner. The plugin manages its own discriminator,
    optimizer, and training loop — PPO stays unmodified.
    """

    # Observation groups
    obs_group: str = "amp"
    """Key in the observation TensorDict for AMP features."""

    condition_obs_group: str | None = None
    """Key for conditional AMP observation (None = unconditional)."""

    # Sequence
    num_frames: int = 2
    """Number of consecutive frames for discriminator input."""

    # Reward
    loss_type: str = "LSGAN"
    """Discriminator loss type: 'GAN', 'LSGAN', or 'WGAN'."""

    style_reward_scale: float = 2.0
    """Scaling factor for style reward. total = task + scale * style * dt."""

    # Discriminator architecture
    hidden_dims: list[int] = MISSING
    """Hidden layer dimensions for discriminator MLP."""

    activation: str = "relu"
    """Activation function name."""

    # Optimizer
    disc_learning_rate: float = 5e-4
    """Discriminator learning rate."""

    lr_scale: float | None = None
    """If set, disc_lr = policy_lr * lr_scale (overrides disc_learning_rate)."""

    disc_trunk_weight_decay: float = 1e-4
    """L2 regularization for discriminator trunk layers."""

    disc_linear_weight_decay: float = 1e-2
    """L2 regularization for discriminator output layer."""

    disc_max_grad_norm: float = 0.5
    """Max gradient norm for discriminator."""

    # Training
    disc_num_learning_epochs: int = 5
    """Discriminator training epochs per policy update."""

    disc_num_mini_batches: int = 4
    """Number of mini-batches for discriminator training."""

    grad_penalty_scale: float = 10.0
    """Gradient penalty weight."""

    noise_scale: float | None = None
    """If set, adds uniform noise to observations during discriminator training."""

    # Conditional AMP
    num_conditions: int = 2
    """Number of discrete conditions (e.g., walk=0, run=1)."""

    condition_embedding_dim: int = 16
    """Embedding dimension for condition IDs."""

    # Data
    motion_files: list[str] | None = None
    """Motion PKL files for offline data (alternative to env.cfg.load_amp_data)."""

    motion_keys: list[str] | None = None
    """Feature keys to extract from motion files."""

    mirror: bool = False
    """Whether to apply mirror augmentation to offline data."""

    joint_mirror_indices: list[int] | None = None
    """Joint index permutation for mirroring."""

    joint_mirror_signs: list[float] | None = None
    """Sign flips for mirrored joints."""

    point_indices: list[int] | None = None
    """Indices to select from key_points_b in motion data."""


# =========================================================================
# Plugin-based AMP Runner Configs (recommended, use AMPPluginRunner)
# =========================================================================


@configclass
class UnitreeG1RoughPluginRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Plugin-based AMP runner: PPO stays vanilla, AMP is a standalone plugin.

    Uses single-step 'amp' observation group; the plugin builds multi-frame
    sequences internally from the rollout storage. Supports conditional AMP,
    key-point observations, mirror augmentation, training noise, and multi-GPU.
    """

    class_name = "AMPPluginRunner"
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 500
    experiment_name = "unitree_g1_rough_plugin"
    empirical_normalization = True
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "amp": ["amp"],
        "amp_condition": ["amp_condition"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = _RslRlPpoAlgorithmCfg(
        class_name="PPO",
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
    )
    amp_cfg = AMPPluginCfg(
        obs_group="amp",
        condition_obs_group="amp_condition",
        num_frames=2,
        loss_type="LSGAN",
        style_reward_scale=2.0,
        hidden_dims=[1024, 512],
        activation="relu",
        disc_learning_rate=5e-4,
        disc_trunk_weight_decay=1e-4,
        disc_linear_weight_decay=1e-2,
        disc_max_grad_norm=0.5,
        disc_num_learning_epochs=5,
        disc_num_mini_batches=4,
        grad_penalty_scale=10.0,
        noise_scale=None,
        num_conditions=2,
        condition_embedding_dim=16,
        mirror=False,
    )



@configclass
class UnitreeG1FlatPluginRunnerCfg(UnitreeG1RoughPluginRunnerCfg):
    """Flat terrain variant of the plugin-based AMP runner."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 60000
        self.experiment_name = "unitree_g1_flat_plugin"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class UnitreeG1RoughPluginGRURunnerCfg(UnitreeG1RoughPluginRunnerCfg):
    """GRU variant of the plugin-based AMP runner."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_g1_rough_plugin_gru"
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
