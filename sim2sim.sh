#!/usr/bin/env bash
set -euo pipefail

# Quick sim2sim launcher with MuJoCo viewer.
#
# Stand/debug mode:
#   STAND=1 bash sim2sim.sh
# This forces command velocity to (0,0,0) for checking whether the robot can stand still.
#
# Video: --save-video records from start until you close the viewer; saved to OUTPUT_DIR
#        as <task>_sim2sim.mp4 (e.g. rough_forward_sim2sim.mp4).

ROBOT="${ROBOT:-g1}"
TASK="${TASK:-rough_forward}"
ONNX_PATH="${ONNX_PATH:-logs/rsl_rl/unitree_g1_rough/2026-03-06_02-06-24/export/policy_iter_15000.onnx}"
DEPLOY_YAML="${DEPLOY_YAML:-}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results}"

# Teleop:
# - "keyboard": UP/DOWN=vx, LEFT/RIGHT=wz, PgUp/PgDn=vy, Backspace=zero
# - "off": fixed command (from task or --velocity)
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

# Auto-pick deploy.yaml next to the ONNX if present (for correct Kp/Kd, offsets, etc.).
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

# Standing test: zero velocity + no teleop so robot must balance in place.
if [[ "${STAND:-0}" == "1" ]]; then
  cmd=("${cmd[@]/--teleop keyboard/--teleop off}")
  cmd+=(--velocity 0.0 0.0 0.0)
  cmd=("${cmd[@]/--follow/}")
  cmd=("${cmd[@]/--render/}")
else
  # Default forward velocity 1.0 m/s (keyboard teleop uses this as starting value).
  cmd+=(--velocity 1.0 0.0 0.0)
fi

exec "${cmd[@]}" "$@"