#!/usr/bin/env bash
set -euo pipefail

# =========================================================================
# G1 MuJoCo Sim2Sim Evaluation
# =========================================================================
#
# Runs a trained ONNX policy in MuJoCo with interactive viewer.
#
# Usage:
#   bash sim2sim.sh
#   ONNX_PATH=logs/.../policy.onnx bash sim2sim.sh
#   TASK=mixed_terrain bash sim2sim.sh
#   STAND=1 bash sim2sim.sh                  # standing balance test
#   TELEOP=off bash sim2sim.sh               # fixed velocity, no keyboard
#
# Keyboard teleop:
#   UP/DOWN    = forward/backward (vx)
#   LEFT/RIGHT = turn (wz)
#   PgUp/PgDn  = lateral (vy)
#   Backspace  = zero velocity
# =========================================================================

ROBOT="${ROBOT:-g1}"
TASK="${TASK:-rough_forward}"
ONNX_PATH="${ONNX_PATH:-logs/rsl_rl/unitree_g1_rough_plugin/latest/export/policy_latest.onnx}"
DEPLOY_YAML="${DEPLOY_YAML:-}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results}"
TELEOP="${TELEOP:-keyboard}"

cmd=(python scripts/mujoco_eval/run_sim2sim_locomotion.py
  --robot "${ROBOT}"
  --onnx "${ONNX_PATH}"
  --task "${TASK}"
  --render --deploy --follow
  --teleop "${TELEOP}"
  --save-video
  --output-dir "${OUTPUT_DIR}"
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

if [[ "${STAND:-0}" == "1" ]]; then
  cmd=("${cmd[@]/--teleop keyboard/--teleop off}")
  cmd+=(--velocity 0.0 0.0 0.0)
  cmd=("${cmd[@]/--follow/}")
  cmd=("${cmd[@]/--render/}")
else
  cmd+=(--velocity 1.0 0.0 0.0)
fi

exec "${cmd[@]}" "$@"
