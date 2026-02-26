"""ONNX utilities for policy export with metadata.

This module provides:
1. Attaching IsaacLab metadata to ONNX models
2. Building observation specification for sim2sim
3. Exporting complete deployment configuration

The metadata enables MuJoCo sim2sim by providing:
- Joint names and order
- Action scale and offset
- Observation structure
- PD gains
- Physics parameters
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def build_obs_spec(env) -> dict:
    """Build observation specification from environment.
    
    Args:
        env: IsaacLab ManagerBasedRLEnv
        
    Returns:
        Dictionary with observation structure
    """
    obs_mgr = env.observation_manager
    obs_names = obs_mgr.active_terms.get("policy", [])
    obs_cfgs = obs_mgr._group_obs_term_cfgs.get("policy", [])
    
    obs_spec: dict[str, Any] = {}
    obs_dims: list[int] = []
    obs_scales: dict[str, Any] = {}
    term_history: dict[str, int] = {}

    # Prefer IsaacLab's aggregated per-term dims for the group if available.
    # This reflects the *actual* vector fed to the policy (after history stacking etc).
    group_term_dim = None
    if hasattr(obs_mgr, "group_obs_term_dim"):
        try:
            group_term_dim = obs_mgr.group_obs_term_dim.get("policy", None)
        except Exception:
            group_term_dim = None
    
    for obs_name, obs_cfg in zip(obs_names, obs_cfgs):
        # Determine term dim
        # - If group_obs_term_dim is available, trust it (already includes stacking).
        # - Otherwise, fall back to calling the term function (often returns single-frame),
        #   and multiply by history_length when configured.
        dim: int
        if isinstance(group_term_dim, dict) and obs_name in group_term_dim:
            dim = int(group_term_dim[obs_name])
            # Best-effort: infer single-frame dim when history_length is known
            obs_sample = obs_cfg.func(env, **obs_cfg.params)
            single_dim = (
                int(obs_sample.shape[-1])
                if getattr(obs_sample, "dim", lambda: 1)() > 1
                else int(obs_sample.numel() // env.num_envs)
            )
        else:
            obs_sample = obs_cfg.func(env, **obs_cfg.params)
            single_dim = (
                int(obs_sample.shape[-1])
                if getattr(obs_sample, "dim", lambda: 1)() > 1
                else int(obs_sample.numel() // env.num_envs)
            )
            h_tmp = int(obs_cfg.history_length) if obs_cfg.history_length else 1
            dim = int(single_dim * h_tmp)
        
        obs_dims.append(int(dim))
        
        # Get scale
        if obs_cfg.scale is not None:
            scale = obs_cfg.scale
            if hasattr(scale, 'tolist'):
                scale = scale.tolist()
            obs_scales[obs_name] = scale
        
        # Record history length for this term (if configured)
        h = int(obs_cfg.history_length) if obs_cfg.history_length else 1
        term_history[obs_name] = int(h)

        # Build spec entry
        obs_spec[obs_name] = {
            "dim": int(dim),
            "scale": obs_scales.get(obs_name, 1.0),
            "history_length": int(h),
            "single_dim": int(single_dim),
        }
    
    # Derive a global history length (best-effort). Many locomotion policies use a shared frame-stack.
    global_history = max(term_history.values(), default=1)
    single_frame_dims: dict[str, int] = {}
    if global_history > 1:
        for name, dim in zip(obs_names, obs_dims):
            # Only provide single-frame dims for terms that match the global history length.
            # (MuJoCo Sim2Sim currently assumes a single global history length.)
            if term_history.get(name, 1) == global_history and dim % global_history == 0:
                single_frame_dims[name] = int(dim // global_history)

    return {
        "observation_names": obs_names,
        "observation_dims": obs_dims,
        "observation_scales": obs_scales,
        "obs_spec": obs_spec,
        "history_length": int(global_history),
        "single_frame_dims": single_frame_dims,
        "total_obs_dim": int(sum(obs_dims)),
    }


def build_action_spec(env) -> dict:
    """Build action specification from environment.
    
    Args:
        env: IsaacLab ManagerBasedRLEnv
        
    Returns:
        Dictionary with action structure
    """
    action_names = env.action_manager.active_terms
    
    joint_names = []
    action_scale = []
    action_offset = []
    
    for action_name, action_term in zip(action_names, env.action_manager._terms.values()):
        # Get joint names from action term
        if hasattr(action_term, '_joint_names'):
            joint_names.extend(action_term._joint_names)
        elif hasattr(action_term.cfg, 'joint_names'):
            joint_names.extend(action_term.cfg.joint_names)
        
        # Get scale
        if hasattr(action_term, '_scale'):
            # IsaacLab versions differ: _scale can be tensor shaped (num_envs, dim),
            # tensor shaped (dim,), python list, or a scalar float.
            s = getattr(action_term, "_scale")
            if isinstance(s, (float, int)):
                action_scale.extend([float(s)] * action_term.action_dim)
            elif hasattr(s, "detach"):
                arr = s.detach().cpu().numpy()
                # Accept (num_envs, dim) or (dim,)
                if arr.ndim == 2:
                    arr = arr[0]
                action_scale.extend(arr.tolist())
            else:
                # Fallback for list/tuple/np array
                try:
                    if isinstance(s, (list, tuple)) and len(s) == action_term.action_dim:
                        action_scale.extend([float(x) for x in s])
                    elif isinstance(s, (list, tuple)) and len(s) > 0:
                        action_scale.extend([float(s[0])] * action_term.action_dim)
                    else:
                        action_scale.extend([1.0] * action_term.action_dim)
                except Exception:
                    action_scale.extend([1.0] * action_term.action_dim)
        elif hasattr(action_term.cfg, 'scale'):
            if isinstance(action_term.cfg.scale, float):
                action_scale.extend([action_term.cfg.scale] * action_term.action_dim)
            else:
                action_scale.extend(action_term.cfg.scale)
        
        # Get offset
        if hasattr(action_term, "_offset"):
            o = getattr(action_term, "_offset")
            if isinstance(o, (float, int)):
                action_offset.extend([float(o)] * action_term.action_dim)
            elif hasattr(o, "detach"):
                arr = o.detach().cpu().numpy()
                if arr.ndim == 2:
                    arr = arr[0]
                action_offset.extend(arr.tolist())
            else:
                try:
                    if isinstance(o, (list, tuple)) and len(o) == action_term.action_dim:
                        action_offset.extend([float(x) for x in o])
                    elif isinstance(o, (list, tuple)) and len(o) > 0:
                        action_offset.extend([float(o[0])] * action_term.action_dim)
                    else:
                        action_offset.extend([0.0] * action_term.action_dim)
                except Exception:
                    action_offset.extend([0.0] * action_term.action_dim)
        else:
            action_offset.extend([0.0] * action_term.action_dim)
    
    return {
        "joint_names": joint_names,
        "action_scale": action_scale,
        "action_offset": action_offset,
        "num_actions": len(joint_names),
    }


def build_physics_spec(env) -> dict:
    """Build physics specification from environment.
    
    Args:
        env: IsaacLab ManagerBasedRLEnv
        
    Returns:
        Dictionary with physics parameters
    """
    asset = env.scene["robot"]
    
    # Get PD gains
    stiffness = asset.data.default_joint_stiffness[0].detach().cpu().numpy().tolist()
    damping = asset.data.default_joint_damping[0].detach().cpu().numpy().tolist()
    
    # Get default positions
    default_joint_pos = asset.data.default_joint_pos[0].detach().cpu().numpy().tolist()
    
    # Timing
    sim_dt = env.cfg.sim.dt
    decimation = env.cfg.decimation
    policy_dt = sim_dt * decimation
    
    return {
        "joint_stiffness": stiffness,
        "joint_damping": damping,
        "default_joint_pos": default_joint_pos,
        "sim_dt": sim_dt,
        "decimation": decimation,
        "policy_dt": policy_dt,
    }


def build_onnx_metadata(env) -> dict:
    """Build complete metadata for ONNX export.
    
    Args:
        env: IsaacLab ManagerBasedRLEnv
        
    Returns:
        Complete metadata dictionary
    """
    metadata = {}
    
    # Observation spec
    obs_spec = build_obs_spec(env)
    metadata.update(obs_spec)
    
    # Action spec
    action_spec = build_action_spec(env)
    metadata.update(action_spec)
    
    # Physics spec
    physics_spec = build_physics_spec(env)
    metadata.update(physics_spec)
    
    # Height scan config (if available)
    if hasattr(env.scene, 'height_scanner'):
        scanner = env.scene.height_scanner
        if hasattr(scanner.cfg, 'pattern_cfg'):
            pattern = scanner.cfg.pattern_cfg
            metadata["height_scan_size"] = list(pattern.size)
            metadata["height_scan_resolution"] = pattern.resolution
    
    return metadata


def attach_onnx_metadata(
    onnx_path: str | Path,
    metadata: dict,
    output_path: str | Path | None = None,
) -> str:
    """Attach metadata to ONNX model.
    
    Args:
        onnx_path: Path to input ONNX file
        metadata: Metadata dictionary to attach
        output_path: Output path (default: overwrite input)
        
    Returns:
        Path to output ONNX file
    """
    try:
        import onnx
    except ImportError:
        print("[Warning] onnx package not installed, skipping metadata attachment")
        return str(onnx_path)
    
    onnx_path = Path(onnx_path)
    if output_path is None:
        output_path = onnx_path
    output_path = Path(output_path)
    
    # Load model
    model = onnx.load(str(onnx_path))
    
    # Add metadata
    metadata_json = json.dumps(metadata)
    
    # Remove existing metadata_json if present
    for prop in list(model.metadata_props):
        if prop.key == "metadata_json":
            model.metadata_props.remove(prop)
    
    # Add new metadata
    meta = model.metadata_props.add()
    meta.key = "metadata_json"
    meta.value = metadata_json
    
    # Save
    onnx.save(model, str(output_path))
    
    print(f"[ONNX] Attached metadata to {output_path}")
    print(f"       Keys: {list(metadata.keys())}")
    
    return str(output_path)


def export_onnx_with_metadata(
    env,
    policy,
    output_path: str | Path,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
) -> str:
    """Export ONNX model with IsaacLab metadata.
    
    Args:
        env: IsaacLab environment
        policy: Policy network (torch.nn.Module)
        output_path: Output ONNX path
        input_names: Input tensor names
        output_names: Output tensor names
        
    Returns:
        Path to exported ONNX file
    """
    import torch
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get observation dimension
    obs_dim = env.observation_manager.group_obs_dim.get("policy", 0)
    if obs_dim == 0:
        obs_dim = sum(build_obs_spec(env)["observation_dims"])
    
    # Create dummy input
    dummy_input = torch.zeros(1, obs_dim, device=next(policy.parameters()).device)
    
    # Default names
    if input_names is None:
        input_names = ["obs"]
    if output_names is None:
        output_names = ["action"]
    
    # Export to ONNX
    policy.eval()
    torch.onnx.export(
        policy,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )
    
    print(f"[ONNX] Exported policy to {output_path}")
    
    # Build and attach metadata
    metadata = build_onnx_metadata(env)
    attach_onnx_metadata(output_path, metadata)
    
    return str(output_path)
