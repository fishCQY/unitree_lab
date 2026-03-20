# G1 AMP Locomotion 配置详解

本文档详细说明 unitree_lab 中 G1 机器人 AMP Locomotion 训练的完整 MDP 环境配置、
算法参数、数据流、以及训练/评估脚本使用方式。

---

## 1. 配置继承层级

```
LocomotionEnvCfg (base_env_cfg.py)            ← 通用 locomotion 基类
    └── UnitreeG1RoughEnvCfg (rough_env_cfg.py)   ← G1 粗糙地形 + AMP
        ├── UnitreeG1RoughEnvCfg_PLAY             ← 评估变体
        └── UnitreeG1FlatEnvCfg (flat_env_cfg.py)  ← 平地变体
            └── UnitreeG1FlatEnvCfg_PLAY
```

---

## 2. MDP 环境配置

### 2.1 Scene（场景）

| 组件 | 配置 | 说明 |
|------|------|------|
| terrain | `ROUGH_TERRAINS_CFG` | 粗糙地形生成器，课程学习 |
| robot | `UNITREE_G1_CFG` (29dof) | 5 组 actuator: legs/feet/shoulders/arms/wrist |
| height_scanner | RayCaster | 0.1m 分辨率，1.6x1.0m，挂 torso_link |
| depth_camera | 32x24 px | σ=0.03 噪声，2 步延迟 |
| contact_forces | 全身 | history=3，track_air_time=True |
| imu | DelayedImu | 挂 pelvis |

### 2.2 Actions（动作）

```
JointPositionAction:
    joint_names = [".*"]        → 29 个关节
    scale = 0.25                → action ∈ [-1,1] × 0.25 = [-0.25, 0.25] rad
    use_default_offset = True   → target = default_pos + scale × action
    clip = [-100, 100]          → 不实际限制（IsaacLab 由关节限位处理）
```

### 2.3 Commands（指令）

```
UniformVelocityCommand:
    lin_vel_x  = (-1.0, 1.0)    m/s    （课程后扩展到 ±2.0）
    lin_vel_y  = (-0.5, 0.5)    m/s
    ang_vel_z  = (-2.0, 2.0)    rad/s
    resampling = (4.0, 6.0) s
    standing   = 10% 环境给零速
    heading    = True, stiffness=2.0
```

### 2.4 Observations（观测）

**4 个活跃观测组**，通过 `obs_groups` 映射给 Runner：

```
obs_groups = {
    "policy":        ["policy"],           → Actor 输入
    "critic":        ["critic"],           → Critic 输入（特权信息）
    "amp":           ["amp"],              → AMP 判别器特征
    "amp_condition": ["amp_condition"],    → AMP 条件 ID
}
```

#### PolicyCfg（Actor 输入，96 维）

| 观测项 | 维度 | 噪声 | 说明 |
|--------|------|------|------|
| `imu_ang_vel` | 3 | ±0.2 | body frame 角速度 |
| `imu_projected_gravity` | 3 | ±0.05 | body frame 重力方向 |
| `velocity_commands` | 3 | — | [vx, vy, wz] 目标速度 |
| `joint_pos_rel` | 29 | ±0.01 | 关节位置偏差 |
| `joint_vel_rel` | 29 | ±1.5 | 关节速度偏差 |
| `last_action` | 29 | — | 上一步动作反馈 |

注意：**不包含 height_scan 和 base_lin_vel**，即 blind locomotion。

#### CriticCfg（Critic 输入，特权观测）

在 PolicyCfg 基础上额外包含：

| 观测项 | 说明 |
|--------|------|
| `base_lin_vel` | 基座线速度（策略看不到） |
| `height_scan` | 地形高度图（±0.1 噪声） |
| `joint_torques` / `joint_accs` | 力矩和加速度 |
| `feet_lin_vel` / `feet_contact_force` | 脚部速度和接触力 |
| `base_mass_rel` / `rigid_body_material` / `base_com` | 物理属性 |
| `action_delay_*` | 各 actuator 组的延迟 |
| `push_force` / `push_torque` | 外部扰动力 |
| `contact_information` | 全身 26 个 body 的接触信息 |

#### AmpCfg（AMP 判别器，76 维/帧）

| 观测项 | 维度 | 说明 |
|--------|------|------|
| `amp_joint_pos` | 29 | 绝对关节位置 |
| `amp_joint_vel` | 29 | 绝对关节速度 |
| `amp_base_ang_vel` | 3 | body frame 角速度 |
| `amp_projected_gravity` | 3 | body frame 重力 |
| `amp_body_pos_b` | 12 | 4 个 body（膝盖×2 + 肩膀×2）在 body frame 的 3D 位置 |

AMPPlugin 从 rollout storage 取最近 `num_frames=2` 帧，拼接为 152 维判别器输入。

#### AmpConditionCfg（条件 ID，1 维）

```python
vel_cmd_condition_id(vx_threshold=1.1):
    |vx| ≤ 1.1 → 0 (walk)
    |vx| > 1.1 → 1 (run)
```

### 2.5 Rewards（奖励）

#### 核心 tracking reward

| Reward | 权重 | 说明 |
|--------|------|------|
| `track_lin_vel_xy_exp` | **+2.5** | 线速度追踪（yaw frame，std=√0.15） |
| `track_ang_vel_z_exp` | **+1.5** | 角速度追踪（world frame，std=√0.5） |

#### 核心 penalty

| Reward | 权重 | 说明 |
|--------|------|------|
| `action_rate_l2` | **-0.1** | 动作平滑性 |
| `dof_pos_limits` | **-1.0** | 关节限位惩罚 |
| `undesired_contacts` | **-2.0** | 非脚部碰撞惩罚 |

#### 正则化 reward（与 AMP style 协同）

| Reward | 权重 | 说明 |
|--------|------|------|
| `feet_air_time` | **+0.3** | 鼓励双足交替步态 |
| `feet_slide` | **-0.1** | 惩罚脚底滑动 |
| `joint_deviation_hip` | **-0.1** | hip/shoulder/elbow 偏离默认姿态 |
| `joint_deviation_arms` | **-0.1** | waist/shoulder_roll/yaw/wrist 偏离 |
| `joint_deviation_legs` | **-0.05** | hip_pitch/knee/ankle 偏离 |

#### 总奖励组成

```
task_reward = 上述所有 reward 加权求和
style_reward = AMP 判别器输出（见 §3）
total_reward = 0.5 × task_reward + 0.5 × style_reward
```

### 2.6 Events（域随机化）

| 事件 | 模式 | 内容 |
|------|------|------|
| physics_material | startup | 摩擦系数 ∈ [0.2, 1.3] |
| add_base_mass | startup | torso 质量 × [1.0, 1.2] |
| scale_link_mass | startup | 其他 link × [0.85, 1.15] |
| randomize_rigid_body_com | startup | torso 重心 ±2.5cm |
| scale_actuator_gains | startup | Kp/Kd × [0.8, 1.2] |
| scale_joint_armature | startup | armature × [0.75, 1.25] |
| reset_base | reset | 位置 ±0.5m，速度 ±0.5 |
| reset_robot_joints | reset | 关节 × [0.5, 1.5] |
| base_external_force_torque | interval 3.7~4.2s | 推力 ±200N (x,y), ±100N (z) |

### 2.7 Terminations（终止）

| 条件 | 说明 |
|------|------|
| time_out | 10 秒 |
| base_contact | torso_link 接触力 > 1N |

### 2.8 Curriculum（课程）

| 课程 | 说明 |
|------|------|
| terrain_levels | 基于速度追踪成功率调整地形难度 |
| command_levels | 逐步扩大速度范围到 ±2.0 / ±0.5 / ±2.0 |

---

## 3. 算法配置

### 3.1 使用的 Cfg

**`UnitreeG1RoughPluginRunnerCfg`** — 唯一推荐的训练配置。

| 参数 | 值 | 说明 |
|------|------|------|
| class_name | `"AMPPluginRunner"` | PPO + AMP 插件 |
| num_steps_per_env | 24 | rollout 长度 |
| max_iterations | 200000 | 最大训练步数 |
| save_interval | 500 | checkpoint 保存间隔 |
| empirical_normalization | True | 运行时观测归一化 |

### 3.2 PPO 算法（标准，无修改）

| 参数 | 值 |
|------|------|
| class_name | `"PPO"` |
| learning_rate | 1e-3 (adaptive) |
| gamma | 0.99 |
| lam | 0.95 |
| clip_param | 0.2 |
| entropy_coef | 0.005 |
| num_learning_epochs | 5 |
| num_mini_batches | 4 |
| desired_kl | 0.01 |

### 3.3 Actor-Critic 网络

| 参数 | 值 |
|------|------|
| Actor | MLP [1024, 512, 256], ELU |
| Critic | MLP [1024, 512, 256], ELU |
| init_noise_std | 1.0 |
| obs_normalization | True (both) |

### 3.4 AMPPlugin 配置

| 参数 | 值 | 说明 |
|------|------|------|
| obs_group | `"amp"` | 76 维 AMP 特征 |
| condition_obs_group | `"amp_condition"` | walk/run 条件 |
| num_frames | 2 | 判别器看 2 帧 |
| loss_type | `"LSGAN"` | Least-Squares GAN |
| style_reward_scale | 1.0 | style reward 缩放 |
| task_style_lerp | 0.5 | total = 0.5×task + 0.5×style |
| hidden_dims | [1024, 512] | 判别器 MLP |
| disc_learning_rate | 5e-4 | 判别器学习率 |
| grad_penalty_scale | 10.0 | 梯度惩罚 |
| num_conditions | 2 | walk=0, run=1 |
| condition_embedding_dim | 16 | 条件嵌入维度 |

### 3.5 Runner 变体

| Cfg | 环境 | 策略 | experiment_name |
|------|------|------|------|
| `UnitreeG1RoughPluginRunnerCfg` | rough | MLP | `unitree_g1_rough_plugin` |
| `UnitreeG1FlatPluginRunnerCfg` | flat | MLP [512,256,128] | `unitree_g1_flat_plugin` |
| `UnitreeG1RoughPluginGRURunnerCfg` | rough | GRU (512) | `unitree_g1_rough_plugin_gru` |

---

## 4. AMP 数据加载

### 4.1 数据来源

```
data/AMP/
├── lafan_walk_clips.pkl    ← LAFAN 步行动作 (retargeted to G1 29dof)
└── lafan_run_clips.pkl     ← LAFAN 跑步动作
```

### 4.2 加载流程

```
AMPPluginRunner.__init__()
    → _load_amp_offline_data()
        → env.cfg.load_amp_data()              ← UnitreeG1RoughEnvCfg 中定义
            → create_mirror_config(left, right, all)  ← 生成 mirror 映射
            → load_conditional_amp_data(
                  conditions = {"walk": [...], "run": [...]},
                  keys = ["dof_pos", "dof_vel", "root_angle_vel", "proj_grav"],
                  mirror = True,
              )
            → 返回 AMPMotionData (含 motion_data, condition_ids)
```

### 4.3 Mirror 增强

- 每个 clip 自动生成镜像版本（数据量 ×2）
- `dof_pos`/`dof_vel`: 左右关节索引交换 + roll/yaw 关节取反
- `root_angle_vel`: `[-ωx, ωy, -ωz]`
- `proj_grav`: `[gx, -gy, gz]`

### 4.4 训练时奖励计算

```
每步:
    AMP 观测 (76d) → 存入 rollout storage
    
每 24 步 PPO 更新:
    AMPPlugin.reward():
        取最近 2 帧 → (4096, 2, 76) → flatten → (4096, 152)
        拼接 condition embedding → (4096, 168)
        判别器 MLP → disc_score
        style_reward = clamp(1 - 0.25 × (disc_score - 1)²) × scale × dt
        
    total = 0.5 × task_reward + 0.5 × style_reward
    
    AMPPlugin.update():
        离线序列 vs 策略序列 → 判别器 loss + gradient penalty
```

---

## 5. Gym 注册表

| 任务 ID | 环境 Cfg | Runner Cfg |
|---------|----------|------------|
| `unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-v0` | `UnitreeG1RoughEnvCfg` | `UnitreeG1RoughPluginRunnerCfg` |
| `unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-Play-v0` | `UnitreeG1RoughEnvCfg_PLAY` | `UnitreeG1RoughPluginRunnerCfg` |
| `unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-GRU-v0` | `UnitreeG1RoughEnvCfg` | `UnitreeG1RoughPluginGRURunnerCfg` |
| `unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-GRU-Play-v0` | `UnitreeG1RoughEnvCfg_PLAY` | `UnitreeG1RoughPluginGRURunnerCfg` |
| `unitree_lab-Isaac-Velocity-Flat-Unitree-G1-AMP-v0` | `UnitreeG1FlatEnvCfg` | `UnitreeG1FlatPluginRunnerCfg` |
| `unitree_lab-Isaac-Velocity-Flat-Unitree-G1-AMP-Play-v0` | `UnitreeG1FlatEnvCfg_PLAY` | `UnitreeG1FlatPluginRunnerCfg` |

---

## 6. 脚本使用方式

### 6.1 训练

```bash
# 默认：rough terrain + AMP + wandb + sim2sim
bash train.sh

# Flat terrain
TASK=unitree_lab-Isaac-Velocity-Flat-Unitree-G1-AMP-v0 bash train.sh

# GRU policy
TASK=unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-GRU-v0 bash train.sh
```

### 6.2 回放

```bash
# 回放 checkpoint
CHECKPOINT=logs/rsl_rl/unitree_g1_rough_plugin/2026-03-20/model_50000.pt bash play.sh

# Flat terrain play
TASK=unitree_lab-Isaac-Velocity-Flat-Unitree-G1-AMP-Play-v0 \
CHECKPOINT=logs/.../model_50000.pt bash play.sh
```

### 6.3 MuJoCo Sim2Sim

```bash
# 交互式 sim2sim（键盘遥操作）
ONNX_PATH=logs/.../export/policy_iter_50000.onnx bash sim2sim.sh

# 站立测试
STAND=1 ONNX_PATH=logs/.../export/policy_iter_50000.onnx bash sim2sim.sh

# 地形 sim2sim
TASK=stairs_up ONNX_PATH=logs/.../export/policy_iter_50000.onnx bash sim2sim_terrain.sh

# 指定任务
TASK=mixed_terrain bash sim2sim.sh
TASK=flat_forward bash sim2sim.sh
```

---

## 7. 文件索引

```
unitree_lab/
├── train.sh                                         # 训练启动脚本
├── play.sh                                          # 回放启动脚本
├── sim2sim.sh                                       # MuJoCo sim2sim 脚本
├── sim2sim_terrain.sh                               # 地形 sim2sim 脚本
│
├── scripts/rsl_rl/
│   ├── train.py                                     # 训练入口（AMPPluginRunner）
│   └── play.py                                      # 回放入口
│
├── source/unitree_lab/unitree_lab/
│   ├── tasks/locomotion/
│   │   ├── robots/g1/
│   │   │   ├── __init__.py                          # Gym 注册表
│   │   │   ├── rough_env_cfg.py                     # G1 粗糙地形 + AMP + load_amp_data()
│   │   │   └── flat_env_cfg.py                      # G1 平地变体
│   │   ├── config/
│   │   │   ├── envs/base_env_cfg.py                 # 通用 locomotion 基类
│   │   │   └── agents/rsl_rl_ppo_cfg.py             # AMPPlugin Runner 配置
│   │   └── mdp/observations.py                      # amp_* 观测函数
│   │
│   ├── utils/amp_data_loader.py                     # LAFAN 数据加载 + mirror
│   └── data/AMP/                                    # LAFAN PKL 数据
│       ├── lafan_walk_clips.pkl
│       └── lafan_run_clips.pkl
│
└── rsl_rl/
    ├── plugins/amp.py                               # AMPPlugin 实现
    └── runners/amp_plugin_runner.py                  # AMPPluginRunner
```
