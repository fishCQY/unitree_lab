#!/usr/bin/env python3
"""Long-horizon sim2sim diagnostic: compares ONNX vs checkpoint and logs
fall timing, observation divergence, and per-joint behaviour.

Usage:
    python scripts/mujoco_eval/compare_onnx_vs_checkpoint.py \
        --onnx  logs/rsl_rl/.../export/policy_iter_3000.onnx \
        --ckpt  logs/rsl_rl/.../model_3000.pt \
        [--deploy-yaml logs/rsl_rl/.../export/deploy_latest.yaml] \
        [--steps 50000] [--velocity 0.5 0.0 0.0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_source_pkg = PROJECT_ROOT / "source" / "unitree_lab"
for _p in [str(PROJECT_ROOT), str(_source_pkg)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import types
_ul = types.ModuleType("unitree_lab")
_ul.__path__ = [str(_source_pkg / "unitree_lab")]
_ul.__package__ = "unitree_lab"
sys.modules.setdefault("unitree_lab", _ul)

import numpy as np
import torch


def _load_pytorch_policy(ckpt_path: str, obs_dim: int, act_dim: int, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]

    mean = sd["actor_obs_normalizer._mean"]
    var_ = sd["actor_obs_normalizer._var"]
    std_ = sd.get("actor_obs_normalizer._std", torch.sqrt(var_))
    eps = 1e-2

    actor_keys = sorted([k for k in sd if k.startswith("actor.") and ".weight" in k])
    layers: list[torch.nn.Module] = []
    for wk in actor_keys:
        bk = wk.replace(".weight", ".bias")
        W, b = sd[wk], sd.get(bk)
        lin = torch.nn.Linear(W.shape[1], W.shape[0])
        lin.weight.data.copy_(W)
        lin.bias.data.copy_(b if b is not None else torch.zeros(W.shape[0]))
        layers.append(lin)
        if wk != actor_keys[-1]:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).to(device).eval()

    state_dep_std = layers[-1].out_features == 2 * act_dim if isinstance(layers[-1], torch.nn.Linear) else False
    mean, std_ = mean.to(device), std_.to(device)

    @torch.no_grad()
    def policy_fn(obs_np: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
        out = actor((obs_t - mean) / (std_ + eps))
        if state_dep_std:
            out = out[..., 0, :]
        return out.squeeze(0).cpu().numpy()
    return policy_fn


def _load_deploy_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_override(deploy: dict) -> dict:
    override = {}
    for dk, ok in {"stiffness": "joint_stiffness", "damping": "joint_damping", "armature": "joint_armature"}.items():
        if dk in deploy:
            override[ok] = deploy[dk]
    for k in ("joint_names", "default_joint_pos", "sim_dt", "decimation"):
        if k in deploy:
            override[k] = deploy[k]
    jpa = deploy.get("actions", {}).get("JointPositionAction", {})
    if "scale" in jpa:
        override["action_scale"] = jpa["scale"]
    if "offset" in jpa:
        override["action_offset"] = jpa["offset"]
    return override


def _run_long_test(sim, policy_fn, label: str, steps: int, vel: tuple):
    """Run one trajectory, return stats dict."""
    sim.reset()
    sim.set_velocity_command(*vel)
    fell_step = None
    heights = []
    tilts = []
    for s in range(steps):
        obs = sim.build_observation()
        act = policy_fn(obs)
        sim.step(act)
        h = float(sim.base_pos[2])
        heights.append(h)
        R = sim._quat_to_rotation_matrix(sim.base_quat)
        tilt = float(R @ np.array([0, 0, 1]))[2] if False else float((R @ np.array([0, 0, 1]))[2])
        tilts.append(tilt)
        if sim._check_termination() and fell_step is None:
            fell_step = s + 1
            break
    return {
        "label": label,
        "steps": len(heights),
        "fell": fell_step is not None,
        "fell_step": fell_step,
        "min_height": min(heights),
        "min_tilt": min(tilts),
        "final_dist": float(np.linalg.norm(sim.base_pos[:2])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--xml", default=None)
    parser.add_argument("--deploy-yaml", default=None)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--velocity", type=float, nargs=3, default=[0.5, 0.0, 0.0])
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    ckpt_path = Path(args.ckpt)

    deploy_yaml_path = args.deploy_yaml
    if not deploy_yaml_path:
        for cand in ("deploy.yaml", "deploy_latest.yaml"):
            p = onnx_path.parent / cand
            if p.exists():
                deploy_yaml_path = str(p)
                break

    xml_path = args.xml or str(PROJECT_ROOT / "source" / "unitree_lab" / "unitree_lab"
                                / "assets" / "robots_xml" / "g1" / "scene_29dof.xml")

    override = {}
    if deploy_yaml_path:
        override = _build_override(_load_deploy_yaml(deploy_yaml_path))
        print(f"[INFO] deploy.yaml: {deploy_yaml_path}")

    from unitree_lab.mujoco_utils import BaseMujocoSimulator
    sim = BaseMujocoSimulator(xml_path=xml_path, onnx_path=str(onnx_path), config_override=override or None)

    obs_dim = int(sim.onnx_config.input_dim)
    act_dim = int(sim.onnx_config.output_dim)
    print(f"[INFO] obs_dim={obs_dim}  act_dim={act_dim}  steps={args.steps}")

    pt_policy = _load_pytorch_policy(str(ckpt_path), obs_dim, act_dim)
    print(f"[INFO] Loaded checkpoint: {ckpt_path.name}")

    vel = tuple(args.velocity)

    # Phase 1: action-level comparison (same obs → same actions?)
    print(f"\n{'='*70}")
    print("  Phase 1: Action fidelity (same obs -> compare actions)")
    print(f"{'='*70}")
    sim.reset()
    sim.set_velocity_command(*vel)
    max_diff = 0.0
    for s in range(min(args.steps, 1000)):
        obs = sim.build_observation()
        a_onnx = sim.policy(obs)
        a_pt = pt_policy(obs)
        d = float(np.max(np.abs(a_onnx - a_pt)))
        max_diff = max(max_diff, d)
        sim.step(a_onnx)
    print(f"  max per-element action diff (1000 steps): {max_diff:.2e}")
    print(f"  Verdict: {'MATCH' if max_diff < 1e-4 else 'MISMATCH'}")

    # Phase 2: long-horizon survival
    print(f"\n{'='*70}")
    print(f"  Phase 2: Long-horizon survival ({args.episodes} episodes x {args.steps} steps)")
    print(f"  velocity = {vel}")
    print(f"{'='*70}")

    scenarios = [
        ("zero_vel", (0.0, 0.0, 0.0)),
        ("slow_fwd", (0.3, 0.0, 0.0)),
        ("med_fwd", (0.5, 0.0, 0.0)),
        ("fast_fwd", (1.0, 0.0, 0.0)),
        ("lateral", (0.0, 0.3, 0.0)),
        ("turn", (0.0, 0.0, 0.5)),
    ]

    all_results = []
    for label, v in scenarios:
        ep_results = []
        for ep in range(args.episodes):
            t0 = time.time()
            r = _run_long_test(sim, sim.policy, f"{label}_ep{ep}", args.steps, v)
            dt = time.time() - t0
            ep_results.append(r)
            status = f"FELL@{r['fell_step']}" if r['fell'] else f"SURVIVED {r['steps']} steps"
            print(f"  [{label}] ep{ep}: {status}  dist={r['final_dist']:.1f}m"
                  f"  min_h={r['min_height']:.3f}  ({dt:.1f}s)")
        survived = sum(1 for r in ep_results if not r['fell'])
        all_results.append((label, v, survived, args.episodes, ep_results))

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for label, v, survived, total, ep_results in all_results:
        fell_steps = [r['fell_step'] for r in ep_results if r['fell']]
        avg_fell = f"avg_fell@{int(np.mean(fell_steps))}" if fell_steps else ""
        print(f"  {label:12s} vel={v}  survival={survived}/{total}  {avg_fell}")

    total_survived = sum(s for _, _, s, _, _ in all_results)
    total_episodes = sum(t for _, _, _, t, _ in all_results)
    print(f"\n  Overall survival: {total_survived}/{total_episodes}"
          f" ({100*total_survived/total_episodes:.0f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
