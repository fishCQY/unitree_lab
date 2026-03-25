# Copyright (c) 2024-2026, Light Robotics.
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause

"""Transformer-specific configuration classes for RSL-RL."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from .rl_cfg import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

#########################
# Policy configurations #
#########################


@configclass
class RslRlPpoActorCriticTransformerBaseCfg(RslRlPpoActorCriticCfg):
    """Base configuration for all Transformer-based actor-critic policies.

    Contains parameters shared across all Transformer variants (sliding-window,
    TXL, etc.). Variant-specific parameters are defined in subclasses.
    """

    embed_dim: int = MISSING
    """The embedding dimension for the Transformer."""

    num_heads: int = MISSING
    """The number of attention heads."""

    ff_dim: int = MISSING
    """The feed-forward dimension (SwiGLU hidden size)."""

    num_layers: int = MISSING
    """The number of Transformer blocks."""

    context_len: int = MISSING
    """The context window length (number of observation frames)."""

    frames_per_token: int = 1
    """Number of consecutive observation frames grouped into a single token. Default is 1.

    When > 1, every ``frames_per_token`` consecutive frames in the sliding window are
    concatenated along the feature dimension before being projected into the embedding
    space. This reduces the Transformer sequence length by a factor of ``frames_per_token``
    while preserving temporal information within each token.

    Example: with ``context_len=30`` and ``frames_per_token=3``, the Transformer processes
    10 tokens (each covering 3 frames) instead of 30 tokens.

    ``context_len`` must be divisible by ``frames_per_token``.
    """

    use_transformer_critic: bool = False
    """Whether to use a Transformer critic. If False, uses an MLP critic. Default is False."""

    gradient_checkpointing: bool = False
    """Whether to use gradient checkpointing to reduce GPU memory at the cost of ~30% more compute.

    When enabled, intermediate activations in Transformer blocks are not stored during the forward
    pass but recomputed during the backward pass. This typically saves 50-70% of activation memory.
    Recommended when context_len or num_steps_per_env is large.
    """

    def __post_init__(self):
        super().__post_init__()
        delattr(self, "actor_hidden_dims")


@configclass
class RslRlPpoActorCriticTransformerCfg(RslRlPpoActorCriticTransformerBaseCfg):
    """Configuration for the PPO actor-critic networks with sliding-window Transformer architecture.

    The Transformer policy processes temporal observation sequences using a sliding window
    of past observations. It behaves like an MLP (stateless forward pass, standard PPO flow)
    but with richer input: a context window of observation frames tokenized and processed
    by a Transformer encoder.
    """

    class_name: str = "ActorCriticTransformer"
    """The policy class name. Default is ActorCriticTransformer."""

    token_groups: list[list[str]] | None = None
    """Optional multi-token observation splitting per timestep. Default is None.

    If None, all obs_groups["policy"] are concatenated into one token per timestep.
    If provided, each sub-list defines a token type with its own projection.
    Example: [["proprioception"], ["commands", "terrain"]] creates 2 tokens per timestep.
    """

    use_gru_gating: bool = False
    """Whether to use GRU-type gating layers instead of residual connections (GTrXL paper).

    When enabled, each TransformerBlock replaces its two residual connections (after attention
    and after FFN) with GRU gating units. This stabilizes training and improves performance
    in RL, at the cost of ~50% more parameters per block. Default is False.
    """

    gru_bias: float = 2.0
    """Gate bias for GRU gating identity initialization. Only used when use_gru_gating=True.

    A positive bias initializes the gate close to the identity map (z ~ sigmoid(-bg) ~ 0),
    so the network starts as an approximately Markovian policy and gradually learns to use
    attention. Higher values = stronger identity initialization. Default is 2.0.
    """


############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoTFAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO-TF (Transformer) algorithm.

    Extends the standard PPO algorithm with Transformer-specific mini-batch
    generation and gradient accumulation to reduce peak GPU memory.
    """

    class_name: str = "PPO_TF"
    """The algorithm class name. Default is PPO_TF."""

    accumulation_steps: int = 1
    """Number of gradient accumulation steps per mini-batch. Default is 1 (no accumulation).

    Each mini-batch is split into ``accumulation_steps`` micro-batches. Gradients are
    accumulated across micro-batches before a single optimizer step, reducing peak GPU
    memory at the cost of slightly more compute. Set to 2-8 for large context_len or
    num_steps_per_env values.
    """

    weight_decay: float = 1e-4
    """Weight decay coefficient for AdamW optimizer. Default is 1e-4.

    When > 0, the optimizer is switched from Adam to AdamW with proper parameter groups:
    only Transformer encoder weights (attention, FFN, gating) receive weight decay, while
    normalization parameters, output heads, critic MLP, and noise std are excluded.
    Recommended range for Transformer RL: 1e-4 ~ 1e-3.
    """

    warmup_iterations: int = 0
    """Number of linear warmup iterations. Applies to all schedules. Default is 0.

    During warmup, the learning rate linearly increases from ``min_lr`` to ``learning_rate``,
    regardless of the chosen schedule. After warmup, the selected schedule takes full control.
    """

    min_lr: float = 1e-6
    """Minimum learning rate. Default is 1e-6.

    Used as the starting LR during warmup (ramps up to ``learning_rate``), and as the
    floor LR for cosine decay (decays down to ``min_lr``).
    """

    cosine_max_iterations: int = 10000
    """Total training iterations for cosine decay calculation. Only used when schedule='cosine'.

    Should match the runner's ``max_iterations``. The cosine decay spans from the end of
    warmup to this iteration count. Default is 10000.
    """

    amp_dtype: str | None = None
    """Mixed precision dtype for training acceleration. Default is None.

    Options:

    - ``"bf16"``: BF16 mixed precision. No GradScaler needed, numerically stable.
      Recommended for GPUs with native BF16 support (A100, H100, L20, RTX 4090).
    - ``"fp16"``: FP16 mixed precision with dynamic loss scaling (GradScaler).
      Use on older GPUs without efficient BF16 (e.g. V100, RTX 3090).
    - None: No mixed precision (FP32 only, equivalent to previous behavior).
    """

    torch_compile: bool = False
    """Whether to use ``torch.compile`` to JIT-compile Transformer encoder submodules.

    When enabled, the ``TemporalTransformerEncoder`` modules (actor and optionally critic)
    are compiled with ``torch.compile(module, dynamic=True)``. This fuses operations
    (attention projections, SwiGLU FFN, RMSNorm, RoPE) into optimized CUDA kernels,
    reducing kernel launch overhead and improving throughput.

    Works well in combination with ``amp_dtype`` (mixed precision). The first few
    training iterations will be slower due to compilation warmup.

    Default is False.
    """
