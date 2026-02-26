# 基本训练 + wandb 日志
# python scripts/rsl_rl/train.py \
#   --task unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-v0 \
#   --headless --logger wandb --log_project_name unitree_g1

# 训练 + wandb + 每次保存 checkpoint 都跑 5 秒 sim2sim
python scripts/rsl_rl/train.py \
  --task unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-v0 \
  --headless --logger wandb --log_project_name unitree_g1 \
  --sim2sim --sim2sim_duration 5.0

# 每 2 次 checkpoint 才跑一次 sim2sim（节省资源）
# python scripts/rsl_rl/train.py \
#   --task unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-v0 \
#   --headless --logger wandb --log_project_name unitree_g1 \
#   --sim2sim --sim2sim_every 2