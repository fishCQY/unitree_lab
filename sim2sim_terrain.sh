#!/usr/bin/env bash
set -euo pipefail

# Sim2sim on terrain — uses scene_29dof_terrain.xml with runtime heightfield
# injection (the XML defines an empty hfield that gets populated at startup
# from the task's terrain config, e.g. rough_forward -> mixed terrain).
#
# This should produce identical results to `sim2sim.sh` with TASK=rough_forward,
# but lets you override the XML path for custom terrain scenes.
#
# Usage:
#   bash sim2sim_terrain.sh
#   TASK=stairs_up bash sim2sim_terrain.sh
#   XML_PATH=.../scene_29dof_terrain2.xml bash sim2sim_terrain.sh

ROBOT="${ROBOT:-g1}"
TASK="${TASK:-rough_forward}"
ONNX_PATH="${ONNX_PATH:-logs/rsl_rl/unitree_g1_rough/2026-03-01_10-21-45/export/policy_iter_7000.onnx}"
XML_PATH="${XML_PATH:-source/unitree_lab/unitree_lab/assets/robots_xml/g1/scene_29dof_terrain.xml}"
DEPLOY_YAML="${DEPLOY_YAML:-}"
TELEOP="${TELEOP:-keyboard}"

cmd=(python scripts/mujoco_eval/run_sim2sim_locomotion.py
  --robot "${ROBOT}"
  --onnx "${ONNX_PATH}"
  --xml "${XML_PATH}"
  --task "${TASK}"
  --render --deploy --follow
  --teleop "${TELEOP}"
)

# Auto-pick deploy.yaml next to the ONNX if present.
if [[ -z "${DEPLOY_YAML}" ]]; then
  onnx_dir="$(dirname "${ONNX_PATH}")"
  if [[ -f "${onnx_dir}/deploy.yaml" ]]; then
    DEPLOY_YAML="${onnx_dir}/deploy.yaml"
  elif [[ -f "${onnx_dir}/deploy_latest.yaml" ]]; then
    DEPLOY_YAML="${onnx_dir}/deploy_latest.yaml"
  fi
fi
if [[ -n "${DEPLOY_YAML}" ]]; then
  cmd+=(--deploy-yaml "${DEPLOY_YAML}")
fi

cmd+=(--velocity 1.0 0.0 0.0)

exec "${cmd[@]}" "$@"
