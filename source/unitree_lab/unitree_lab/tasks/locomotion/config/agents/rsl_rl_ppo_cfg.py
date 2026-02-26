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
class AMPCfg:
    amp_reward_weight: float = MISSING
    amp_lambda: float = MISSING
    obs_history_len: int = MISSING
    hidden_dims: list[int] = MISSING
    partial_ids: list[int] = MISSING
    offline_dataset_path: str = MISSING
    lr_scale: float = MISSING
    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    add_noise: float = MISSING
    noise_scale: list[float] | None = MISSING
    activation: str = MISSING
    normalization: bool = MISSING


@configclass
class RslRlGuidanceCfg:
    experts_paths: list[str] = MISSING
    loss_type: str = MISSING
    loss_coeff: float = MISSING


@configclass
class RslRlPpoAlgorithmCfg(_RslRlPpoAlgorithmCfg):
    amp_cfg: AMPCfg | None = None
    guidance_cfg: RslRlGuidanceCfg | None = None


@configclass
class UnitreeG1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 1000
    experiment_name = "unitree_g1_rough"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
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
            amp_reward_weight=0.5,
            amp_lambda=10,
            obs_history_len=2,
            hidden_dims=[1024, 512],
            partial_ids=list(range(0, 6)) + list(range(9, 67)),
            offline_dataset_path=[
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B4_-_Stand_to_Walk_backwards_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B15_-__Walk_turn_around_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B10_-__Walk_turn_left_45_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B5_-__Walk_backwards_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B16_-_Walk_turn_change_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B23_-_Side_step_right_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B22_-_Side_step_left_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B11_-__Walk_turn_left_135_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B22_-__side_step_left_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B4_-_Stand_to_Walk_Back_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B10_-_Walk_turn_left_45_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B4_-_Stand_to_Walk_backwards_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B23_-__side_step_right_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B13_-_Walk_turn_right_45_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B13_-__Walk_turn_right_90_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B9_-__Walk_turn_left_90_stageii.pkl",
            ],
            lr_scale=0.5,
            num_learning_epochs=1,
            num_mini_batches=10,
            add_noise=True,
            noise_scale=[0.2] * 3 + [0.05] * 3 + [0.01] * 29 + [1.5] * 29,
            activation="relu",
            normalization=True,
        ),
    )


@configclass
class UnitreeG1FlatPPORunnerCfg(UnitreeG1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 60000
        self.experiment_name = "unitree_g1_flat"
        self.algorithm.amp_cfg.offline_dataset_path = [
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B4_-_Stand_to_Walk_backwards_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B15_-__Walk_turn_around_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B10_-__Walk_turn_left_45_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B5_-__Walk_backwards_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B16_-_Walk_turn_change_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B23_-_Side_step_right_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B22_-_Side_step_left_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B11_-__Walk_turn_left_135_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B22_-__side_step_left_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B4_-_Stand_to_Walk_Back_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B10_-_Walk_turn_left_45_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B4_-_Stand_to_Walk_backwards_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B23_-__side_step_right_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B13_-_Walk_turn_right_45_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B13_-__Walk_turn_right_90_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B9_-__Walk_turn_left_90_stageii.pkl",
        ]
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

    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 1000
    experiment_name = "unitree_g1_rough_depth"
    empirical_normalization = True
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "image": ["image"],
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
            amp_reward_weight=0.5,
            amp_lambda=10,
            obs_history_len=2,
            hidden_dims=[1024, 512],
            partial_ids=list(range(0, 6)) + list(range(9, 67)),
            offline_dataset_path=[
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B4_-_Stand_to_Walk_backwards_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B15_-__Walk_turn_around_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B10_-__Walk_turn_left_45_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B5_-__Walk_backwards_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B16_-_Walk_turn_change_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B23_-_Side_step_right_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B22_-_Side_step_left_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B11_-__Walk_turn_left_135_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B22_-__side_step_left_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B4_-_Stand_to_Walk_Back_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B10_-_Walk_turn_left_45_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B4_-_Stand_to_Walk_backwards_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B23_-__side_step_right_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/Walk_B13_-_Walk_turn_right_45_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B13_-__Walk_turn_right_90_stageii.pkl",
                "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run/B9_-__Walk_turn_left_90_stageii.pkl",
            ],
            lr_scale=0.5,
            num_learning_epochs=1,
            num_mini_batches=10,
            add_noise=True,
            noise_scale=[0.2] * 3 + [0.05] * 3 + [0.01] * 29 + [1.5] * 29,
            activation="relu",
            normalization=True,
        ),
    )
