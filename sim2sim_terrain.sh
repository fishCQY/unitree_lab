#!/usr/bin/env bash
set -euo pipefail

# =========================================================================
# G1 MuJoCo Sim2Sim on Terrain
# =========================================================================
#
# Uses scene_29dof_terrain.xml with runtime heightfield injection.
#
# Usage:
#   bash sim2sim_terrain.sh
#   TASK=stairs_up bash sim2sim_terrain.sh
#   XML_PATH=.../scene_29dof_terrain2.xml bash sim2sim_terrain.sh
# =========================================================================

ROBOT="${ROBOT:-g1}"
TASK="${TASK:-rough_forward}"
ONNX_PATH="${ONNX_PATH:-logs/rsl_rl/unitree_g1_rough_plugin/latest/export/policy_latest.onnx}"
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
