#!/usr/bin/env bash
set -euo pipefail

# =========================================================================
# G1 AMP Locomotion Training
# =========================================================================
#
# Environment:  UnitreeG1RoughEnvCfg  (rough terrain + LAFAN AMP)
# Algorithm:    AMPPluginRunner       (PPO + AMP discriminator plugin)
# AMP data:     data/AMP/lafan_walk_clips.pkl, lafan_run_clips.pkl
#
# Variants:
#   TASK=unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-v0       (rough, MLP)
#   TASK=unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-GRU-v0   (rough, GRU)
#   TASK=unitree_lab-Isaac-Velocity-Flat-Unitree-G1-AMP-v0        (flat,  MLP)
# =========================================================================

TASK="${TASK:-unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-v0}"

# Basic training + wandb logging + sim2sim video upload
python scripts/rsl_rl/train.py \
  --task "${TASK}" \
  --headless --logger wandb --log_project_name unitree_g1 \
  --sim2sim --sim2sim_duration 20.0

# --- Alternative commands (uncomment as needed) ---

# Training without sim2sim:
# python scripts/rsl_rl/train.py \
#   --task "${TASK}" \
#   --headless --logger wandb --log_project_name unitree_g1

# Sim2sim every 2 checkpoints (save resources):
# python scripts/rsl_rl/train.py \
#   --task "${TASK}" \
#   --headless --logger wandb --log_project_name unitree_g1 \
#   --sim2sim --sim2sim_every 2

# Flat terrain:
# TASK=unitree_lab-Isaac-Velocity-Flat-Unitree-G1-AMP-v0 bash train.sh

# GRU policy:
# TASK=unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-GRU-v0 bash train.sh
