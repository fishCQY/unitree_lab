"""PPO runner configurations for motion tracking tasks."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ActorCriticCommandAECfg(RslRlPpoActorCriticCfg):
    """Actor-critic with an AE command bottleneck (reconstruction aux loss)."""

    class_name: str = "ActorCriticCommandCodec"
    command_obs_key: str = "command"
    command_latent_dim: int = 64
    command_encoder_hidden_dims: list[int] = [256, 256]
    command_decoder_hidden_dims: list[int] = [256, 256]
    aux_loss_coef: float = 0.1


@configclass
class ActorCriticCommandVQVAECfg(RslRlPpoActorCriticCfg):
    """Actor-critic with VQ-VAE codebook on command latent."""

    class_name: str = "ActorCriticCommandVQVAE"
    command_obs_key: str = "command"
    command_latent_dim: int = 64
    command_encoder_hidden_dims: list[int] = [256, 256]
    command_decoder_hidden_dims: list[int] = [256, 256]
    vq_codebook_size: int = 512
    vq_commitment_cost: float = 0.25
    aux_loss_coef: float = 0.1


@configclass
class ActorCriticCommandFSQCfg(RslRlPpoActorCriticCfg):
    """Actor-critic with FSQ quantization on command latent."""

    class_name: str = "ActorCriticCommandFSQ"
    command_obs_key: str = "fsq_command"
    command_latent_dim: int = 64
    command_encoder_hidden_dims: list[int] = [256, 256]
    command_decoder_hidden_dims: list[int] = [256, 256]
    fsq_levels: int = 8
    aux_loss_coef: float = 0.1


@configclass
class ActorCriticCommandRFSQCfg(RslRlPpoActorCriticCfg):
    """Actor-critic with residual FSQ (RFSQ) quantization on command latent."""

    class_name: str = "ActorCriticCommandRFSQ"
    command_obs_key: str = "fsq_command"
    command_latent_dim: int = 64
    command_encoder_hidden_dims: list[int] = [256, 256]
    command_decoder_hidden_dims: list[int] = [256, 256]
    fsq_levels: int = 8
    rfsq_stages: int = 2
    aux_loss_coef: float = 0.1


@configclass
class UnitreeG1TrackingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 1000
    experiment_name = "unitree_g1_tracking"
    obs_groups = {
        "policy": ["command", "policy"],
        "critic": ["command", "critic"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[2048, 1024, 512],
        critic_hidden_dims=[2048, 1024, 512],
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        activation="elu",
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
    )


@configclass
class UnitreeG1TrackingPRPPORunnerCfg(UnitreeG1TrackingPPORunnerCfg):
    experiment_name = "unitree_g1_tracking_pr"


@configclass
class UnitreeG1TrackingABPPORunnerCfg(UnitreeG1TrackingPPORunnerCfg):
    experiment_name = "unitree_g1_tracking_ab"


@configclass
class UnitreeG1TrackingAEPPORunnerCfg(UnitreeG1TrackingPPORunnerCfg):
    """AutoEncoder variant (same env/obs; different runner tag)."""

    experiment_name = "unitree_g1_tracking_ae"
    policy = ActorCriticCommandAECfg(
        init_noise_std=1.0,
        actor_hidden_dims=[2048, 1024, 512],
        critic_hidden_dims=[2048, 1024, 512],
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        activation="elu",
        command_obs_key="command",
        command_latent_dim=64,
        aux_loss_coef=0.1,
    )


@configclass
class UnitreeG1TrackingFSQPPORunnerCfg(UnitreeG1TrackingPPORunnerCfg):
    """FSQ variant: adds future/current/next command observations."""

    experiment_name = "unitree_g1_tracking_fsq"
    obs_groups = {
        "policy": ["fsq_command", "policy"],
        "critic": ["fsq_command", "critic"],
    }
    policy = ActorCriticCommandFSQCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[2048, 1024, 512],
        critic_hidden_dims=[2048, 1024, 512],
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        activation="elu",
        command_obs_key="fsq_command",
        command_latent_dim=64,
        fsq_levels=8,
        aux_loss_coef=0.1,
    )


@configclass
class UnitreeG1TrackingRFSQPPORunnerCfg(UnitreeG1TrackingFSQPPORunnerCfg):
    """RFSQ variant (same obs groups as FSQ)."""

    experiment_name = "unitree_g1_tracking_rfsq"
    policy = ActorCriticCommandRFSQCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[2048, 1024, 512],
        critic_hidden_dims=[2048, 1024, 512],
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        activation="elu",
        command_obs_key="fsq_command",
        command_latent_dim=64,
        fsq_levels=8,
        rfsq_stages=2,
        aux_loss_coef=0.1,
    )


@configclass
class UnitreeG1TrackingVQVAEPPORunnerCfg(UnitreeG1TrackingPPORunnerCfg):
    """VQ-VAE variant (same env/obs; different runner tag)."""

    experiment_name = "unitree_g1_tracking_vqvae"
    policy = ActorCriticCommandVQVAECfg(
        init_noise_std=1.0,
        actor_hidden_dims=[2048, 1024, 512],
        critic_hidden_dims=[2048, 1024, 512],
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        activation="elu",
        command_obs_key="command",
        command_latent_dim=64,
        vq_codebook_size=512,
        vq_commitment_cost=0.25,
        aux_loss_coef=0.1,
    )


@configclass
class RslRlDistillationPolicyCfg:
    class_name: str = "StudentTeacher"
    student_obs_normalization: bool = True
    teacher_obs_normalization: bool = True
    student_hidden_dims: list[int] = [2048, 1024, 512]
    teacher_hidden_dims: list[int] = [2048, 1024, 512]
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"


@configclass
class RslRlDistillationAlgorithmCfg:
    class_name: str = "Distillation"
    num_learning_epochs: int = 5
    gradient_length: int = 15
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    loss_type: str = "mse"
    optimizer: str = "adam"


@configclass
class UnitreeG1TrackingDistillationFSQRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Distillation variant (FSQ student); teacher uses privileged critic observations."""

    class_name: str = "DistillationRunner"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 1000
    experiment_name = "unitree_g1_tracking_distill_fsq"
    obs_groups = {
        "policy": ["fsq_command", "policy"],
        "critic": ["command", "critic"],
        "teacher": ["command", "critic"],
    }
    policy = RslRlDistillationPolicyCfg()
    algorithm = RslRlDistillationAlgorithmCfg()


@configclass
class UnitreeG1TrackingDistillationVQVAERunnerCfg(UnitreeG1TrackingDistillationFSQRunnerCfg):
    """Distillation variant (VQ-VAE student tag)."""

    experiment_name = "unitree_g1_tracking_distill_vqvae"

