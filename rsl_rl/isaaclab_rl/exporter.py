# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2024-2026, unitree_lab contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os

import torch


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Supports MLP, recurrent, and Transformer-based policies. Transformer
    policies are detected via ``policy.is_transformer`` and exported with
    a built-in sliding-window observation buffer.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    if getattr(policy, "is_transformer", False):
        policy_exporter = _TorchTransformerPolicyExporter(policy, normalizer)
    else:
        policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Supports MLP, recurrent, and Transformer-based policies. Transformer
    policies are detected via ``policy.is_transformer`` and exported with
    explicit observation-buffer inputs/outputs for stateless inference.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if getattr(policy, "is_transformer", False):
        policy_exporter = _OnnxTransformerPolicyExporter(policy, normalizer, verbose)
    else:
        policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _DummyEncoder(torch.nn.Module):
    """No-op encoder placeholder so TorchScript can resolve attribute types."""

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        return x.new_zeros(x.shape[0], 0)


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file.

    Accepts ``Dict[str, Tensor]`` input (JIT-compatible proxy for TensorDict),
    replicating the full ``act_inference`` pipeline including obs-group
    concatenation, normalization, optional encoder, optional RNN, and actor.

    Uses dynamic method binding (``self.forward = self.forward_lstm``, etc.)
    so that ``torch.jit.script`` only sees the branch relevant to this
    particular policy, avoiding TorchScript cross-branch type conflicts.
    """

    actor_obs_keys: list[str]
    is_recurrent: bool
    has_encoder: bool
    state_dependent_std: bool
    exteroception_key: str

    def __init__(self, policy, normalizer=None):
        super().__init__()

        self.actor_obs_keys = list(policy.obs_groups["policy"])

        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
        else:
            raise ValueError("Policy does not have an actor/student module.")

        self.state_dependent_std = getattr(policy, "state_dependent_std", False)

        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

        self.has_encoder = hasattr(policy, "exteroception_encoder") and policy.exteroception_encoder is not None
        if self.has_encoder:
            self.encoder = copy.deepcopy(policy.exteroception_encoder)
            self.exteroception_key = getattr(policy, "exteroception_key", "exteroception")
        else:
            self.encoder = _DummyEncoder()
            self.exteroception_key = ""

        self.is_recurrent = policy.is_recurrent
        if self.is_recurrent:
            if hasattr(policy, "memory_a"):
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
            elif hasattr(policy, "memory_s"):
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
            else:
                raise ValueError("Recurrent policy has no memory module.")
            self.rnn.cpu()
            rnn_type = type(self.rnn).__name__.lower()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            if rnn_type == "lstm":
                self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.forward = self._forward_lstm
                self.reset = self._reset_memory_lstm
            elif rnn_type == "gru":
                self.forward = self._forward_gru
                self.reset = self._reset_memory_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {rnn_type}")

    def _get_actor_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for key in self.actor_obs_keys:
            parts.append(obs[key])
        return torch.cat(parts, dim=-1)

    def _encode(self, obs: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        if self.has_encoder:
            latent = self.encoder(obs[self.exteroception_key], condition=x)
            return torch.cat([x, latent], dim=-1)
        return x

    def _actor_out(self, x: torch.Tensor) -> torch.Tensor:
        if self.state_dependent_std:
            return self.actor(x)[..., 0, :]
        return self.actor(x)

    def _forward_lstm(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self._get_actor_obs(obs)
        x = self.normalizer(x)
        x = self._encode(obs, x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self._actor_out(x)

    def _forward_gru(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self._get_actor_obs(obs)
        x = self.normalizer(x)
        x = self._encode(obs, x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        x = x.squeeze(0)
        return self._actor_out(x)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self._get_actor_obs(obs)
        x = self.normalizer(x)
        x = self._encode(obs, x)
        return self._actor_out(x)

    def _reset_memory_lstm(self) -> None:
        self.hidden_state.zero_()
        self.cell_state.zero_()

    def _reset_memory_gru(self) -> None:
        self.hidden_state.zero_()

    @torch.jit.export
    def reset(self) -> None:
        pass

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        self.eval()
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            self.base_obs_dim = self.actor[0].in_features
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
                self.base_obs_dim = self.rnn.input_size
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            self.base_obs_dim = self.actor[0].in_features
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
                self.base_obs_dim = self.rnn.input_size
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # copy exteroception encoder if exists
        if hasattr(policy, "exteroception_encoder") and policy.exteroception_encoder is not None:
            self.encoder = copy.deepcopy(policy.exteroception_encoder)
            self.has_encoder = True
            self.base_obs_dim -= self.encoder.latent_dim
            self.exteroception_shape = policy.exteroception_shape
        else:
            self.encoder = None
            self.has_encoder = False
            self.exteroception_shape = [1]
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in, exteroception):
        x_in = self.normalizer(x_in)
        if self.has_encoder and self.encoder is not None:
            latent = self.encoder(exteroception, condition=x_in)
            x_in = torch.cat([x_in, latent], dim=-1)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward_gru(self, x_in, h_in, exteroception):
        x_in = self.normalizer(x_in)
        if self.has_encoder and self.encoder is not None:
            latent = self.encoder(exteroception, condition=x_in)
            x_in = torch.cat([x_in, latent], dim=-1)
        x, h = self.rnn(x_in.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return self.actor(x), h

    def forward(self, x, exteroception):
        x = self.normalizer(x)
        if self.has_encoder and self.encoder is not None:
            latent = self.encoder(exteroception, condition=x)
            x = torch.cat([x, latent], dim=-1)
        return self.actor(x)

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18  # was 11, but it caused problems with linux-aarch, and 18 worked well across all systems.

        # Create exteroception tensor based on encoder input shape (or [1] for models without encoder)
        exteroception = torch.zeros(1, *self.exteroception_shape)

        obs = torch.zeros(1, self.base_obs_dim)
        if self.is_recurrent:
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)

            if self.rnn_type == "lstm":
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in, exteroception),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=opset_version,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in", "exteroception"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in, exteroception),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=opset_version,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "exteroception"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            torch.onnx.export(
                self,
                (obs, exteroception),
                os.path.join(path, filename),
                export_params=True,
                opset_version=opset_version,
                verbose=self.verbose,
                input_names=["obs", "exteroception"],
                output_names=["actions"],
                dynamic_axes={},
            )


class _TorchTransformerPolicyExporter(torch.nn.Module):
    """Exporter of Transformer actor-critic into JIT file.

    Maintains an internal sliding-window observation buffer for stateful
    single-frame inference.  Each ``forward(obs)`` call appends the new
    observation to the buffer, builds a validity mask, normalizes, runs the
    Transformer encoder, and returns action predictions.

    Call ``reset()`` to clear the buffer (e.g., on episode boundaries).
    """

    context_len: int  # TorchScript requires class-level type annotations

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.actor_transformer = copy.deepcopy(policy.actor_transformer)
        self.actor_output = copy.deepcopy(policy.actor_output)
        self.context_len = policy.context_len
        obs_dim: int = policy.actor_transformer.obs_dim

        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

        # Pre-allocated sliding-window buffer (1 env for deployment)
        self.register_buffer("obs_buffer", torch.zeros(1, policy.context_len, obs_dim))
        self.register_buffer("valid_len", torch.zeros(1, dtype=torch.long))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Run one inference step.

        Args:
            obs: (1, obs_dim) observation for the current timestep.

        Returns:
            (1, num_actions) action predictions.
        """
        # Shift buffer left by one position and insert new obs at the end
        self.obs_buffer[:, :-1] = self.obs_buffer[:, 1:].clone()
        self.obs_buffer[:, -1] = obs
        self.valid_len.add_(1).clamp_(max=self.context_len)

        # Build validity mask: True = valid, False = padding
        seq_len: int = self.context_len
        valid: int = int(self.valid_len.item())
        window_mask: torch.Tensor | None = None
        if valid < seq_len:
            indices = torch.arange(seq_len, device=obs.device)
            window_mask = indices.unsqueeze(0) >= (seq_len - valid)

        # Normalize -> Transformer -> output head
        obs_window = self.normalizer(self.obs_buffer)
        feature = self.actor_transformer(obs_window, window_mask)
        return self.actor_output(feature)

    @torch.jit.export
    def reset(self) -> None:
        """Reset the observation buffer and valid-frame counter."""
        self.obs_buffer.zero_()
        self.valid_len.zero_()

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        self.to("cpu")
        scripted = torch.jit.script(self)
        scripted.save(filepath)


class _OnnxTransformerPolicyExporter(torch.nn.Module):
    """Exporter of Transformer actor-critic into ONNX file.

    Stateless design: the sliding-window buffer and valid-frame counter are
    passed as inputs and returned (updated) as outputs so that the calling
    runtime (e.g., ONNX Runtime on a robot) manages the state externally.
    """

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor_transformer = copy.deepcopy(policy.actor_transformer)
        self.actor_output = copy.deepcopy(policy.actor_output)
        self.context_len: int = policy.context_len
        self.obs_dim: int = policy.actor_transformer.obs_dim

        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(
        self,
        obs: torch.Tensor,
        obs_buffer: torch.Tensor,
        valid_len: torch.Tensor,
    ) -> tuple:
        """Process one observation with explicit state management.

        Args:
            obs: (1, obs_dim) current observation.
            obs_buffer: (1, context_len, obs_dim) sliding-window buffer.
            valid_len: (1,) number of valid frames currently in the buffer.

        Returns:
            Tuple of (actions, updated_obs_buffer, updated_valid_len).
        """
        # Shift buffer left and insert new observation at the end
        new_buffer = torch.cat([obs_buffer[:, 1:, :], obs.unsqueeze(1)], dim=1)
        new_valid_len = (valid_len + 1).clamp(max=self.context_len)

        # Always build mask (ONNX tracing requires deterministic control flow)
        seq_len = new_buffer.shape[1]
        indices = torch.arange(seq_len, device=obs.device)
        window_mask = indices.unsqueeze(0) >= (seq_len - new_valid_len.unsqueeze(1))

        # Normalize -> Transformer -> output head
        obs_window = self.normalizer(new_buffer)
        feature = self.actor_transformer(obs_window, window_mask)
        actions = self.actor_output(feature)

        return actions, new_buffer, new_valid_len

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        self.to("cpu")
        self.eval()

        obs = torch.zeros(1, self.obs_dim)
        obs_buffer = torch.zeros(1, self.context_len, self.obs_dim)
        valid_len = torch.zeros(1, dtype=torch.long)

        torch.onnx.export(
            self,
            (obs, obs_buffer, valid_len),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=self.verbose,
            input_names=["obs", "obs_buffer", "valid_len"],
            output_names=["actions", "obs_buffer_out", "valid_len_out"],
            dynamic_axes={},
        )
