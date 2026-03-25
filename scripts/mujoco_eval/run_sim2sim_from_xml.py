#!/usr/bin/env python3
"""Run sim2sim by directly loading a given MuJoCo XML + ONNX.

This is meant for your use-case: you already have an XML that contains the
terrain setup, so we **do not inject/overwrite** the heightfield data.

Example:
  python scripts/mujoco_eval/run_sim2sim_from_xml.py \\
    --onnx logs/.../policy.onnx \\
    --xml  source/unitree_lab/unitree_lab/assets/robots_xml/g1/scene_29dof_terrain.xml \\
    --deploy-yaml logs/.../deploy_latest.yaml \\
    --render --follow --teleop keyboard
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


def _bootstrap_paths() -> None:
    project_root = Path(__file__).resolve().parents[2]
    source_pkg = project_root / "source" / "unitree_lab"
    for p in (str(project_root), str(source_pkg), str(Path(__file__).resolve().parent)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Provide a lightweight `unitree_lab` package root (avoid importing unitree_lab.__init__)
    import types

    ul = types.ModuleType("unitree_lab")
    ul.__path__ = [str(source_pkg / "unitree_lab")]
    ul.__package__ = "unitree_lab"
    sys.modules.setdefault("unitree_lab", ul)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, type=str)
    p.add_argument("--xml", required=True, type=str)
    p.add_argument("--deploy-yaml", type=str, default=None)
    p.add_argument("--config-override", type=str, default=None, help="JSON string override")

    p.add_argument("--task", type=str, default="flat_stand", help="Only used for velocity profile defaults")
    p.add_argument("--render", action="store_true")
    p.add_argument("--follow", action="store_true")
    p.add_argument("--teleop", type=str, default="keyboard", choices=["keyboard", "off"])
    p.add_argument("--velocity", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"))
    p.add_argument("--max-steps", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    _bootstrap_paths()
    args = parse_args()

    onnx = Path(args.onnx)
    xml = Path(args.xml)
    if not onnx.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx}")
    if not xml.exists():
        raise FileNotFoundError(f"XML not found: {xml}")

    from unitree_lab.mujoco_utils.evaluation.eval_task import TERRAIN_FLAT, get_eval_task
    from run_sim2sim_locomotion import (
        _deploy_yaml_to_config_override,
        _load_deploy_yaml,
        run_interactive,
        run_headless,
    )

    # Do not inject terrain; use XML terrain as-is (copy so we never mutate registry tasks).
    task = replace(get_eval_task(args.task), terrain=TERRAIN_FLAT)

    override = {}
    if args.deploy_yaml:
        override.update(_deploy_yaml_to_config_override(_load_deploy_yaml(args.deploy_yaml)))
    if args.config_override:
        override.update(json.loads(args.config_override))

    from unitree_lab.mujoco_utils.simulation.locomotion_simulator import LocomotionMujocoSimulator

    simulator = LocomotionMujocoSimulator(
        onnx_path=str(onnx),
        mujoco_model_path=str(xml),
    )

    velocity = tuple(args.velocity) if args.velocity else None

    if args.render:
        run_interactive(
            simulator=simulator,
            task=task,
            teleop=args.teleop,
            follow=args.follow,
            velocity=velocity,
            max_steps_per_episode=args.max_steps,
        )
    else:
        run_headless(
            simulator=simulator,
            task=task,
            num_episodes=1,
            save_video=False,
            output_dir="eval_results",
            video_steps=args.max_steps,
            velocity=velocity,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()

