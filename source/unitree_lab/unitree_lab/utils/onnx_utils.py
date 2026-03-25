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
import os
from pathlib import Path
from typing import Any

import numpy as np


def _list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x)
        for x in arr  # numbers → format, strings → as-is
    )


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


def build_physics_spec(env, action_joint_indices: list[int] | None = None) -> dict:
    """Build physics specification from environment.
    
    Args:
        env: IsaacLab ManagerBasedRLEnv
        action_joint_indices: Optional joint indices to reorder physics params to match
            action joint order. If None, returns params in asset (USD) order.
        
    Returns:
        Dictionary with physics parameters
    """
    asset = env.scene["robot"]
    
    # Get raw data in USD/asset order
    stiffness_raw = asset.data.default_joint_stiffness[0].detach().cpu().numpy()
    damping_raw = asset.data.default_joint_damping[0].detach().cpu().numpy()
    default_pos_raw = asset.data.default_joint_pos[0].detach().cpu().numpy()
    armature_raw = asset.data.default_joint_armature[0].detach().cpu().numpy()
    
    # Reorder to action joint order if indices provided
    if action_joint_indices is not None:
        idx = action_joint_indices
        stiffness = stiffness_raw[idx].tolist()
        damping = damping_raw[idx].tolist()
        default_joint_pos = default_pos_raw[idx].tolist()
        armature = armature_raw[idx].tolist()
    else:
        stiffness = stiffness_raw.tolist()
        damping = damping_raw.tolist()
        default_joint_pos = default_pos_raw.tolist()
        armature = armature_raw.tolist()
    
    # Timing
    sim_dt = env.cfg.sim.dt
    decimation = env.cfg.decimation
    policy_dt = sim_dt * decimation

    return {
        "joint_stiffness": stiffness,
        "joint_damping": damping,
        "joint_armature": armature,
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
    
    # Action spec (get joint names and their indices in asset order)
    action_spec = build_action_spec(env)
    metadata.update(action_spec)
    
    # Build mapping from action joint names to asset joint indices
    # This ensures physics params (stiffness, damping, default_pos) are in
    # the same order as action outputs, which sim2sim expects.
    action_joint_indices = None
    if action_spec.get("joint_names"):
        asset = env.scene["robot"]
        asset_joint_names = asset.joint_names  # USD/asset order
        action_joint_names = action_spec["joint_names"]
        try:
            action_joint_indices = [asset_joint_names.index(name) for name in action_joint_names]
        except ValueError:
            # Fallback: try without _joint suffix matching
            action_joint_indices = []
            for aname in action_joint_names:
                found = False
                for i, uname in enumerate(asset_joint_names):
                    if aname == uname or aname.replace("_joint", "") == uname.replace("_joint", ""):
                        action_joint_indices.append(i)
                        found = True
                        break
                if not found:
                    print(f"[Warning] Could not find joint index for: {aname}")
                    action_joint_indices = None
                    break
    
    # Physics spec (reordered to match action joint order)
    physics_spec = build_physics_spec(env, action_joint_indices)
    metadata.update(physics_spec)
    
    # Height scan config (if available)
    if hasattr(env.scene, 'height_scanner'):
        scanner = env.scene.height_scanner
        if hasattr(scanner.cfg, 'pattern_cfg'):
            pattern = scanner.cfg.pattern_cfg
            metadata["height_scan_size"] = list(pattern.size)
            metadata["height_scan_resolution"] = pattern.resolution
    
    return metadata


def _attach_onnx_metadata_from_training_env(env, run_path: str, path: str, filename: str = "policy.onnx") -> None:
    """Attach env.onnx_metadata + run_path to an exported ONNX (training env API)."""
    try:
        import onnx
    except ImportError:
        print("[Warning] onnx package not installed, skipping metadata attachment")
        return

    onnx_path = os.path.join(path, filename)
    model = onnx.load(onnx_path)

    entry = onnx.StringStringEntryProto()
    entry.key = "run_path"
    entry.value = run_path
    model.metadata_props.append(entry)

    onnx_meta = getattr(env, "onnx_metadata", None) or {}
    full_metadata = {"run_path": run_path, **onnx_meta}

    entry = onnx.StringStringEntryProto()
    entry.key = "metadata_json"
    entry.value = json.dumps(full_metadata, ensure_ascii=False)
    model.metadata_props.append(entry)

    for k, v in onnx_meta.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        if v is None:
            entry.value = "null"
        elif isinstance(v, list):
            entry.value = _list_to_csv_str(v)
        elif isinstance(v, dict):
            entry.value = json.dumps(v)
        else:
            entry.value = str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
    print(f"[ONNX] Attached training metadata to {onnx_path}")


def attach_onnx_metadata(
    onnx_path_or_env: str | Path | Any,
    metadata_or_run_path: dict | str | None = None,
    output_path: str | Path | None = None,
    *,
    path: str | None = None,
    filename: str | None = None,
) -> str | None:
    """Attach metadata to ONNX model.

    Supports two call styles:

    1. Path style (original): ``attach_onnx_metadata(onnx_path, metadata_dict, output_path=None)``
    2. Training env style: ``attach_onnx_metadata(env, run_path, path=..., filename=...)``
    """
    if path is not None:
        if filename is None:
            filename = "policy.onnx"
        run_path = metadata_or_run_path if isinstance(metadata_or_run_path, str) else ""
        _attach_onnx_metadata_from_training_env(onnx_path_or_env, run_path, path, filename)
        return os.path.join(path, filename)

    if metadata_or_run_path is None or not isinstance(metadata_or_run_path, dict):
        print("[Warning] attach_onnx_metadata: expected metadata dict for path-style call")
        return str(onnx_path_or_env)

    onnx_path = Path(onnx_path_or_env)
    metadata = metadata_or_run_path
    if output_path is None:
        output_path = onnx_path
    output_path = Path(output_path)

    try:
        import onnx
    except ImportError:
        print("[Warning] onnx package not installed, skipping metadata attachment")
        return str(onnx_path)

    model = onnx.load(str(onnx_path))

    metadata_json = json.dumps(metadata)

    for prop in list(model.metadata_props):
        if prop.key == "metadata_json":
            model.metadata_props.remove(prop)

    meta = model.metadata_props.add()
    meta.key = "metadata_json"
    meta.value = metadata_json

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
