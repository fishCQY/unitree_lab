"""ONNX utilities for loading policies and parsing metadata.

This module handles:
1. Loading ONNX models with onnxruntime
2. Parsing IsaacLab metadata from ONNX files
3. Managing inference with hidden states (GRU/LSTM)

The metadata_json in ONNX contains critical information:
- joint_names: Action output order
- action_scale: Per-joint action scaling
- default_joint_pos: Default positions for offset
- observation_names/dims: Obs structure for alignment
- height_scan_size/resolution: Exteroception config
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


@dataclass
class OnnxConfig:
    """Configuration extracted from ONNX metadata."""
    
    # Core policy info
    input_dim: int = 0
    output_dim: int = 0
    
    # Joint configuration (action order)
    joint_names: list[str] = field(default_factory=list)
    action_scale: list[float] = field(default_factory=list)
    action_offset: list[float] = field(default_factory=list)
    default_joint_pos: list[float] = field(default_factory=list)
    
    # PD gains
    joint_stiffness: list[float] = field(default_factory=list)
    joint_damping: list[float] = field(default_factory=list)
    joint_armature: list[float] = field(default_factory=list)
    tau_limits: list[float] = field(default_factory=list)
    
    # Observation structure
    observation_names: list[str] = field(default_factory=list)
    observation_dims: list[int] = field(default_factory=list)
    observation_scales: dict[str, float] = field(default_factory=dict)
    
    # History stacking (for obs with temporal context)
    history_length: int = 1  # Number of stacked frames
    single_frame_dims: dict[str, int] = field(default_factory=dict)  # Single-frame dim per term
    # IsaacLab (ManagerBased env) history stacking order is oldest-first along the feature dimension.
    # We keep an override knob for compatibility with other exporters.
    history_newest_first: bool = False  # Stack order for history terms
    
    # Height scan config
    height_scan_size: tuple[float, float] | None = None
    height_scan_resolution: float = 0.1
    height_scan_offset: float = 0.5
    
    # Physics params
    armature: dict[str, float] = field(default_factory=dict)
    damping: dict[str, float] = field(default_factory=dict)
    friction: dict[str, float] = field(default_factory=dict)
    
    # Timing
    policy_dt: float = 0.02  # decimation * sim_dt
    decimation: int = 4
    sim_dt: float = 0.005
    
    # Hidden state (RNN)
    has_hidden_state: bool = False
    hidden_state_dim: int = 0
    
    # Raw metadata for extensibility
    raw_metadata: dict = field(default_factory=dict)


def get_onnx_config(onnx_path: str | Path) -> OnnxConfig:
    """Extract configuration from ONNX model metadata.
    
    Args:
        onnx_path: Path to ONNX model file
        
    Returns:
        OnnxConfig with parsed metadata
    """
    if ort is None:
        raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")
    
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    
    config = OnnxConfig()
    
    # Get input/output dims
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    if inputs:
        config.input_dim = inputs[0].shape[-1] if inputs[0].shape else 0
        config.has_hidden_state = len(inputs) > 1
        if config.has_hidden_state and len(inputs) > 1:
            config.hidden_state_dim = inputs[1].shape[-1] if inputs[1].shape else 0
    
    if outputs:
        config.output_dim = outputs[0].shape[-1] if outputs[0].shape else 0
    
    # Parse metadata
    metadata = session.get_modelmeta().custom_metadata_map
    
    if "metadata_json" in metadata:
        try:
            meta_dict = json.loads(metadata["metadata_json"])
            config.raw_metadata = meta_dict
            _parse_metadata_dict(config, meta_dict)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse metadata_json: {e}")
    
    # Also check deploy.yaml style keys (individual metadata fields)
    for key, value in metadata.items():
        if key == "metadata_json":
            continue
        try:
            parsed = json.loads(value)
            if key == "joint_names":
                config.joint_names = parsed
            elif key == "action_scale":
                config.action_scale = parsed
            elif key == "default_joint_pos":
                config.default_joint_pos = parsed
            elif key == "observation_names":
                config.observation_names = parsed
            elif key == "observation_dims":
                config.observation_dims = parsed
            elif key == "history_length":
                config.history_length = parsed
            elif key == "single_frame_dims":
                config.single_frame_dims = parsed
            elif key == "num_actions":
                config.output_dim = parsed
            elif key == "total_obs_dim":
                config.input_dim = parsed
        except (json.JSONDecodeError, TypeError):
            pass
    
    return config


def _parse_metadata_dict(config: OnnxConfig, meta: dict) -> None:
    """Parse metadata dictionary into config."""
    # Joint configuration
    if "joint_names" in meta:
        config.joint_names = meta["joint_names"]
    if "action_scale" in meta:
        config.action_scale = meta["action_scale"]
    if "action_offset" in meta:
        config.action_offset = meta["action_offset"]
    if "default_joint_pos" in meta:
        config.default_joint_pos = meta["default_joint_pos"]
    
    # PD gains
    if "joint_stiffness" in meta:
        config.joint_stiffness = meta["joint_stiffness"]
    if "joint_damping" in meta:
        config.joint_damping = meta["joint_damping"]
    if "tau_limits" in meta:
        config.tau_limits = meta["tau_limits"]
    
    # Observation structure
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
        # Allow explicit override via metadata for non-IsaacLab exporters.
        config.history_newest_first = bool(meta["history_newest_first"])
    
    # Height scan
    if "height_scan_size" in meta:
        size = meta["height_scan_size"]
        config.height_scan_size = (size[0], size[1]) if isinstance(size, list) else size
    if "height_scan_resolution" in meta:
        config.height_scan_resolution = meta["height_scan_resolution"]
    if "height_scan_offset" in meta:
        config.height_scan_offset = meta["height_scan_offset"]
    
    # Physics
    if "armature" in meta:
        config.armature = meta["armature"]
    if "damping" in meta:
        config.damping = meta["damping"]
    if "friction" in meta:
        config.friction = meta["friction"]
    
    # Timing
    if "policy_dt" in meta:
        config.policy_dt = meta["policy_dt"]
    if "decimation" in meta:
        config.decimation = meta["decimation"]
    if "sim_dt" in meta:
        config.sim_dt = meta["sim_dt"]


def load_onnx_model(onnx_path: str | Path, device: str = "cpu") -> ort.InferenceSession:
    """Load ONNX model for inference.
    
    Args:
        onnx_path: Path to ONNX file
        device: Device for inference ("cpu" or "cuda")
        
    Returns:
        ONNX inference session
    """
    if ort is None:
        raise ImportError("onnxruntime not installed")
    
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    return ort.InferenceSession(str(onnx_path), providers=providers)


class OnnxInference:
    """ONNX inference wrapper with hidden state management.
    
    Supports both feedforward and recurrent (GRU/LSTM) policies.
    """
    
    def __init__(self, onnx_path: str | Path, device: str = "cpu"):
        """Initialize inference session.
        
        Args:
            onnx_path: Path to ONNX model
            device: Inference device
        """
        self.session = load_onnx_model(onnx_path, device)
        self.config = get_onnx_config(onnx_path)
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Initialize hidden state if needed
        self._hidden_state: np.ndarray | None = None
        if self.config.has_hidden_state:
            self.reset_hidden_state()
    
    def reset_hidden_state(self) -> None:
        """Reset hidden state to zeros."""
        if self.config.has_hidden_state and self.config.hidden_state_dim > 0:
            self._hidden_state = np.zeros(
                (1, self.config.hidden_state_dim), 
                dtype=np.float32
            )
    
    def infer(self, obs: np.ndarray) -> np.ndarray:
        """Run inference on observation.
        
        Args:
            obs: Observation array of shape (obs_dim,) or (1, obs_dim)
            
        Returns:
            Action array of shape (action_dim,)
        """
        # Ensure batch dimension
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        obs = obs.astype(np.float32)
        
        # Build input dict
        inputs = {self.input_names[0]: obs}
        
        # Add hidden state if recurrent
        if self.config.has_hidden_state and self._hidden_state is not None:
            if len(self.input_names) > 1:
                inputs[self.input_names[1]] = self._hidden_state
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        
        # Update hidden state if output
        if len(outputs) > 1 and self.config.has_hidden_state:
            self._hidden_state = outputs[1]
        
        # Return action (remove batch dim)
        return outputs[0].squeeze(0)
    
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Alias for infer."""
        return self.infer(obs)
