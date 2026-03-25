"""ONNX utilities for loading policies and parsing metadata.

This module handles:
1. Loading ONNX models with onnxruntime
2. Parsing IsaacLab metadata from ONNX files (returns plain dict)
3. Detecting policy type (feedforward / recurrent / transformer)
4. Initializing hidden states for recurrent policies
5. Managing inference with hidden states (GRU/LSTM/Transformer)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import onnx as _onnx_lib
except ImportError:
    _onnx_lib = None


def get_onnx_config(onnx_path: str | Path) -> dict:
    """Extract configuration from ONNX model metadata.

    Parsing priority:
    1. ``metadata_json`` field (complete JSON dump)
    2. Individual fields with JSON / CSV / string parsing

    Args:
        onnx_path: Path to ONNX model file.

    Returns:
        Dictionary containing parsed configuration values.
    """
    onnx_path = str(onnx_path)

    if _onnx_lib is not None:
        model = _onnx_lib.load(onnx_path)
        meta = model.metadata_props
    elif ort is not None:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        meta_map = session.get_modelmeta().custom_metadata_map

        class _FakeProp:
            def __init__(self, k, v):
                self.key = k
                self.value = v

        meta = [_FakeProp(k, v) for k, v in meta_map.items()]
    else:
        raise ImportError("Neither onnx nor onnxruntime is installed")

    for p in meta:
        if p.key == "metadata_json":
            try:
                config = json.loads(p.value)
                return config
            except json.JSONDecodeError:
                break

    config: dict[str, Any] = {}
    for p in meta:
        key = p.key
        value = p.value
        if key == "metadata_json":
            continue
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError:
            if "," in value:
                try:
                    config[key] = [float(x) for x in value.split(",")]
                except ValueError:
                    config[key] = value.split(",")
            else:
                config[key] = value

    return config


def detect_policy_type(ort_session: "ort.InferenceSession") -> str:
    """Detect the policy architecture from ONNX model inputs/outputs.

    Returns:
        One of ``"transformer"``, ``"recurrent"`` (GRU/LSTM), or ``"feedforward"``.
    """
    input_names = {inp.name for inp in ort_session.get_inputs()}
    if "obs_buffer" in input_names and "valid_len" in input_names:
        return "transformer"
    output_names = [out.name for out in ort_session.get_outputs()]
    if len(output_names) > 1:
        return "recurrent"
    return "feedforward"


def init_hidden_states(
    ort_session: "ort.InferenceSession",
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """Initialize hidden states for recurrent policies.

    Transformer policies are detected separately via :func:`detect_policy_type`
    and managed by the simulator.

    Returns:
        Tuple of (hidden_state, cell_state, is_recurrent).
    """
    policy_type = detect_policy_type(ort_session)

    hidden_state = None
    cell_state = None
    is_recurrent = policy_type == "recurrent"

    if is_recurrent:
        for input_info in ort_session.get_inputs()[1:]:
            if "h_in" in input_info.name:
                hidden_state = np.zeros(input_info.shape, dtype=np.float32)
            elif "c_in" in input_info.name:
                cell_state = np.zeros(input_info.shape, dtype=np.float32)

    return hidden_state, cell_state, is_recurrent


# ---------------------------------------------------------------------------
# Legacy OnnxConfig dataclass — kept for backward compatibility.
# New code should use ``get_onnx_config()`` which returns a plain dict.
# ---------------------------------------------------------------------------

@dataclass
class OnnxConfig:
    """Configuration extracted from ONNX metadata (DEPRECATED — use dict)."""

    input_dim: int = 0
    output_dim: int = 0
    joint_names: list[str] = field(default_factory=list)
    action_scale: list[float] = field(default_factory=list)
    action_offset: list[float] = field(default_factory=list)
    default_joint_pos: list[float] = field(default_factory=list)
    joint_stiffness: list[float] = field(default_factory=list)
    joint_damping: list[float] = field(default_factory=list)
    joint_armature: list[float] = field(default_factory=list)
    tau_limits: list[float] = field(default_factory=list)
    observation_names: list[str] = field(default_factory=list)
    observation_dims: list[int] = field(default_factory=list)
    observation_scales: dict[str, float] = field(default_factory=dict)
    history_length: int = 1
    single_frame_dims: dict[str, int] = field(default_factory=dict)
    history_newest_first: bool = False
    height_scan_size: tuple[float, float] | None = None
    height_scan_resolution: float = 0.1
    height_scan_offset: float = 0.5
    armature: dict[str, float] = field(default_factory=dict)
    damping: dict[str, float] = field(default_factory=dict)
    friction: dict[str, float] = field(default_factory=dict)
    policy_dt: float = 0.02
    decimation: int = 4
    sim_dt: float = 0.005
    has_hidden_state: bool = False
    hidden_state_dim: int = 0
    raw_metadata: dict = field(default_factory=dict)


def get_onnx_config_dataclass(onnx_path: str | Path) -> OnnxConfig:
    """Legacy helper — returns an :class:`OnnxConfig` dataclass.

    Prefer :func:`get_onnx_config` (returns dict) for new code.
    """
    raw = get_onnx_config(onnx_path)
    cfg = OnnxConfig()
    cfg.raw_metadata = raw

    if ort is not None:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if inputs:
            cfg.input_dim = inputs[0].shape[-1] if inputs[0].shape else 0
            cfg.has_hidden_state = len(inputs) > 1
            if cfg.has_hidden_state and len(inputs) > 1:
                cfg.hidden_state_dim = inputs[1].shape[-1] if inputs[1].shape else 0
        if outputs:
            cfg.output_dim = outputs[0].shape[-1] if outputs[0].shape else 0

    _parse_metadata_dict_into_dataclass(cfg, raw)
    return cfg


def _parse_metadata_dict_into_dataclass(config: OnnxConfig, meta: dict) -> None:
    """Populate an OnnxConfig dataclass from a metadata dict."""
    if "joint_names" in meta:
        config.joint_names = meta["joint_names"]
    if "action_scale" in meta:
        config.action_scale = meta["action_scale"]
    if "action_offset" in meta:
        config.action_offset = meta["action_offset"]
    if "default_joint_pos" in meta:
        config.default_joint_pos = meta["default_joint_pos"]
    if "joint_stiffness" in meta:
        config.joint_stiffness = meta["joint_stiffness"]
    if "joint_damping" in meta:
        config.joint_damping = meta["joint_damping"]
    if "tau_limits" in meta:
        config.tau_limits = meta["tau_limits"]
    if "observation_names" in meta:
        config.observation_names = meta["observation_names"]
    if "observation_dims" in meta:
        config.observation_dims = meta["observation_dims"]
    if "observation_scales" in meta:
        config.observation_scales = meta["observation_scales"]
    if "history_length" in meta:
        config.history_length = meta["history_length"]
    if "single_frame_dims" in meta:
        config.single_frame_dims = meta["single_frame_dims"]
    if "history_newest_first" in meta:
        config.history_newest_first = bool(meta["history_newest_first"])
    if "height_scan_size" in meta:
        size = meta["height_scan_size"]
        config.height_scan_size = (size[0], size[1]) if isinstance(size, list) else size
    if "height_scan_resolution" in meta:
        config.height_scan_resolution = meta["height_scan_resolution"]
    if "height_scan_offset" in meta:
        config.height_scan_offset = meta["height_scan_offset"]
    if "armature" in meta:
        config.armature = meta["armature"]
    if "damping" in meta:
        config.damping = meta["damping"]
    if "friction" in meta:
        config.friction = meta["friction"]
    if "policy_dt" in meta:
        config.policy_dt = meta["policy_dt"]
    if "decimation" in meta:
        config.decimation = meta["decimation"]
    if "sim_dt" in meta:
        config.sim_dt = meta["sim_dt"]
    if "num_actions" in meta:
        config.output_dim = meta["num_actions"]
    if "total_obs_dim" in meta:
        config.input_dim = meta["total_obs_dim"]


# ---------------------------------------------------------------------------
# ONNX inference helpers
# ---------------------------------------------------------------------------

def load_onnx_model(onnx_path: str | Path, device: str = "cpu") -> "ort.InferenceSession":
    """Load ONNX model for inference."""
    if ort is None:
        raise ImportError("onnxruntime not installed")
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)


class OnnxInference:
    """ONNX inference wrapper supporting feedforward / GRU / LSTM / Transformer."""

    def __init__(self, onnx_path: str | Path, device: str = "cpu"):
        self.session = load_onnx_model(onnx_path, device)
        self.config = get_onnx_config(onnx_path)

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        self.policy_type = detect_policy_type(self.session)
        self.is_recurrent = self.policy_type == "recurrent"
        self.is_transformer = self.policy_type == "transformer"
        self.has_exteroception_input = "exteroception" in self.input_names

        self._hidden_state: np.ndarray | None = None
        self._cell_state: np.ndarray | None = None
        self._tf_obs_buffer: np.ndarray | None = None
        self._tf_valid_len: np.ndarray | None = None

        if self.is_recurrent:
            self._hidden_state, self._cell_state, _ = init_hidden_states(self.session)
        elif self.is_transformer:
            for inp in self.session.get_inputs():
                if inp.name == "obs_buffer":
                    self._tf_obs_buffer = np.zeros(inp.shape, dtype=np.float32)
                elif inp.name == "valid_len":
                    self._tf_valid_len = np.zeros(inp.shape, dtype=np.int64)

    def reset_hidden_state(self) -> None:
        """Reset all hidden / recurrent / transformer states to zero."""
        if self._hidden_state is not None:
            self._hidden_state.fill(0)
        if self._cell_state is not None:
            self._cell_state.fill(0)
        if self._tf_obs_buffer is not None:
            self._tf_obs_buffer.fill(0)
        if self._tf_valid_len is not None:
            self._tf_valid_len.fill(0)

    def infer(
        self,
        obs: np.ndarray,
        exteroception: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run inference.

        Args:
            obs: Observation array of shape ``(obs_dim,)`` or ``(1, obs_dim)``.
            exteroception: Optional exteroception tensor.

        Returns:
            Action array of shape ``(action_dim,)``.
        """
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        obs = obs.astype(np.float32)

        if self.is_transformer:
            ort_inputs = {
                "obs": obs,
                "obs_buffer": self._tf_obs_buffer,
                "valid_len": self._tf_valid_len,
            }
            ort_outputs = self.session.run(None, ort_inputs)
            action = ort_outputs[0][0]
            self._tf_obs_buffer[:] = ort_outputs[1]
            self._tf_valid_len[:] = ort_outputs[2]
        elif self.is_recurrent:
            if self._cell_state is not None:  # LSTM
                ort_inputs = {
                    "obs": obs,
                    "h_in": self._hidden_state,
                    "c_in": self._cell_state,
                }
                if self.has_exteroception_input and exteroception is not None:
                    ort_inputs["exteroception"] = exteroception.reshape(1, *exteroception.shape)
                ort_outputs = self.session.run(None, ort_inputs)
                action = ort_outputs[0][0]
                self._hidden_state[:] = ort_outputs[1]
                self._cell_state[:] = ort_outputs[2]
            else:  # GRU
                ort_inputs = {"obs": obs, "h_in": self._hidden_state}
                if self.has_exteroception_input and exteroception is not None:
                    ort_inputs["exteroception"] = exteroception.reshape(1, *exteroception.shape)
                ort_outputs = self.session.run(None, ort_inputs)
                action = ort_outputs[0][0]
                self._hidden_state[:] = ort_outputs[1]
        else:
            ort_inputs = {"obs": obs}
            if self.has_exteroception_input and exteroception is not None:
                ort_inputs["exteroception"] = exteroception.reshape(1, *exteroception.shape)
            action = self.session.run(None, ort_inputs)[0][0]

        return np.clip(action, -100.0, 100.0)

    def __call__(self, obs: np.ndarray, exteroception: np.ndarray | None = None) -> np.ndarray:
        return self.infer(obs, exteroception)
