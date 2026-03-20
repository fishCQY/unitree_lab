#!/usr/bin/env bash
set -euo pipefail

# =========================================================================
# G1 AMP Locomotion Playback
# =========================================================================
#
# Loads a trained checkpoint and runs the policy in IsaacLab viewer.
#
# Usage:
#   bash play.sh
#   CHECKPOINT=logs/.../model_50000.pt bash play.sh
#   TASK=unitree_lab-Isaac-Velocity-Flat-Unitree-G1-AMP-Play-v0 bash play.sh
# =========================================================================

TASK="${TASK:-unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-Play-v0}"
CHECKPOINT="${CHECKPOINT:-logs/rsl_rl/unitree_g1_rough_plugin/latest/model_latest.pt}"

python scripts/rsl_rl/play.py \
  --task "${TASK}" \
  --checkpoint "${CHECKPOINT}"

# --- Export ONNX only (no viewer) ---
# python scripts/rsl_rl/play.py \
#   --task "${TASK}" \
#   --checkpoint "${CHECKPOINT}" \
#   --export-only
