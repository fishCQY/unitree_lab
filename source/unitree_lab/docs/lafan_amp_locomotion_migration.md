# LAFAN AMP Locomotion 迁移记录

本文档记录将 unitree_lab 的 Locomotion + AMP 训练统一为 AMPPlugin 方式，
删除旧版 Legacy AMP（PPOAMP + AMPRunner），并对齐 bfm_training 数据流的所有变更。

---

## 1. 核心变更：数据流对齐

### 迁移前（旧版）

```
                     ┌─ AMPRunner (legacy) ─── PPOAMP ─── 内置判别器
训练 runner 选择  ─┤
                     └─ AMPPluginRunner ────── PPO + AMPPlugin（但无数据加载）
                     
数据加载：
  DiscriminatorDemoCfg → AMPDemoObsTerm → 从 pkl 直接构建连续帧
  数据路径: data/MotionData/g1_29dof/amp/lafan/ （不存在）
  mirror: 未启用
  条件 AMP: 观测组已定义，但数据无条件标签
```

### 迁移后（统一 AMPPlugin）

```
训练 runner: AMPPluginRunner → PPO (vanilla) + AMPPlugin (standalone)

数据加载:
  env.cfg.load_amp_data()
    → load_conditional_amp_data({"walk": [...], "run": [...]})
    → AMPMotionData (含 condition_ids)
    → AMPPlugin.set_offline_data()

数据路径: data/AMP/lafan_walk_clips.pkl, lafan_run_clips.pkl
mirror: 启用 (create_mirror_config 生成 G1 左右对称映射)
条件 AMP: walk (|vx| ≤ 1.1) / run (|vx| > 1.1)
```

---

## 2. 文件变更清单

### 修改的文件

| 文件 | 变更内容 |
|------|----------|
| `tasks/locomotion/robots/g1/rough_env_cfg.py` | 添加 `load_amp_data()` 方法；更新数据路径到 `data/AMP/`；删除 `DiscriminatorCfg` / `DiscriminatorDemoCfg` 观测组；恢复 reward 权重 |
| `tasks/locomotion/config/agents/rsl_rl_ppo_cfg.py` | 删除所有旧版配置（`AMPCfg`, `AMPDiscriminatorCfg`, `RslRlPpoAlgorithmCfg`, `UnitreeG1RoughPPORunnerCfg` 等）；新增 `UnitreeG1FlatPluginRunnerCfg`, `UnitreeG1RoughPluginGRURunnerCfg` |
| `tasks/locomotion/config/agents/__init__.py` | 更新导出为 Plugin 配置 |
| `tasks/locomotion/mdp/observations.py` | 删除 `AMPAgentObsTerm`, `AMPDemoObsTerm` 类 |
| `scripts/rsl_rl/train.py` | `AMPRunner` → `AMPPluginRunner` |
| `scripts/rsl_rl/play.py` | `AMPRunner` → `AMPPluginRunner` |
| `rsl_rl/runners/__init__.py` | 移除 `AMPRunner` 导出 |

### 保留不变的文件

| 文件 | 原因 |
|------|------|
| `rsl_rl/plugins/amp.py` | AMPPlugin 实现完整，无需修改 |
| `rsl_rl/runners/amp_plugin_runner.py` | Runner 逻辑正确，优先调用 `env.cfg.load_amp_data()` |
| `utils/amp_data_loader.py` | 数据加载器已完整 |
| `rsl_rl/runners/amp_runner.py` | 文件保留但不再导出，避免破坏 git 历史 |
| `flat_env_cfg.py` | 继承 rough，自动获得 `load_amp_data()` |
| terrain / scene 配置 | 地形与 AMP 无关 |

---

## 3. 数据加载详细流程

```
AMPPluginRunner.__init__()
    │
    └─ _load_amp_offline_data()
         │
         ├─ 优先级 1: env.cfg.load_amp_data()     ← 新增，主要路径
         │     │
         │     ├─ 构建 mirror config (create_mirror_config)
         │     │     左右关节名 → mirror_indices + mirror_signs
         │     │
         │     └─ load_conditional_amp_data(
         │           conditions = {"walk": ["lafan_walk_clips.pkl"],
         │                         "run":  ["lafan_run_clips.pkl"]},
         │           keys = ["dof_pos", "dof_vel", "root_angle_vel", "proj_grav"],
         │           mirror = True,
         │           joint_mirror_indices = [...],
         │           joint_mirror_signs = [...],
         │       )
         │     → 返回 AMPMotionData(motion_data, condition_ids, ...)
         │
         ├─ 优先级 2: amp_cfg.data_loader_func     ← 未使用
         └─ 优先级 3: amp_cfg.motion_files          ← 未使用

AMPPlugin.set_offline_data(dataset)
    │
    ├─ 归一化器初始化 (obs_normalizer.update)
    ├─ 构建离线序列 (_extract_sequences_from_flat)
    └─ 构建条件 ID 映射 (_offline_cond_ids)

每步训练:
    AMPPlugin.reward(obs, storage, step_dt)
        │
        ├─ 从 rollout storage 获取 amp 观测历史
        ├─ 取最后 num_frames=2 帧
        ├─ 归一化 + flatten
        ├─ 拼接条件 embedding (walk/run)
        ├─ 判别器推理 → disc_score
        └─ style_reward = step_dt * scale * clamp(1 - 0.25*(d-1)^2)

    AMPPluginRunner.learn()
        │
        ├─ total_reward = lerp * task_reward + (1-lerp) * style_reward
        │   (task_style_lerp = 0.5)
        └─ AMPPlugin.update() 训练判别器
```

---

## 4. Reward 权重变更

| Reward | 修改前 | 修改后 | 说明 |
|--------|--------|--------|------|
| `track_lin_vel_xy_exp` | 2.5 | 2.5 | 不变 |
| `track_ang_vel_z_exp` | 1.5 | 1.5 | 不变 |
| `action_rate_l2` | -0.1 | -0.1 | 不变 |
| `dof_pos_limits` | -1.0 | -1.0 | 不变 |
| `undesired_contacts` | -2.0 | -2.0 | 不变 |
| `feet_air_time` | **0.0** | **0.3** | 鼓励双足交替步态 |
| `feet_slide` | **0.0** | **-0.1** | 惩罚脚底滑动 |
| `joint_deviation_hip` | **0.0** | **-0.1** | 与 AMP style 协同 |
| `joint_deviation_arms` | **0.0** | **-0.1** | 与 AMP style 协同 |
| `joint_deviation_legs` | **0.0** | **-0.05** | 与 AMP style 协同 |

---

## 5. 删除的旧版代码

### 配置类（rsl_rl_ppo_cfg.py）

- `AMPDiscriminatorCfg` — PPOAMP 判别器配置
- `AMPCfg` — PPOAMP 的 AMP 配置
- `RslRlPpoAlgorithmCfg` — 扩展的算法配置（含 amp_cfg）
- `UnitreeG1RoughPPORunnerCfg` — Legacy AMPRunner 配置
- `UnitreeG1FlatPPORunnerCfg` — Legacy flat 变体
- `UnitreeG1RoughPPORunnerGRUCfg` — Legacy GRU 变体
- `UnitreeG1RoughDepthPPORunnerCfg` — Legacy depth 变体

### 观测类（observations.py）

- `AMPAgentObsTerm` — 3D 输出的 agent 观测（维护历史 buffer）
- `AMPDemoObsTerm` — 3D 输出的 demo 观测（直接从 pkl 加载帧对）

### 观测组（rough_env_cfg.py）

- `DiscriminatorCfg` — 使用 `AMPAgentObsTerm` 的观测组
- `DiscriminatorDemoCfg` — 使用 `AMPDemoObsTerm` 的观测组
- `disc_agent` / `disc_demo` 实例赋值

---

## 6. 新增的配置

| 配置 | 说明 |
|------|------|
| `UnitreeG1FlatPluginRunnerCfg` | Flat 地形 AMPPlugin runner（继承 rough） |
| `UnitreeG1RoughPluginGRURunnerCfg` | GRU 策略 AMPPlugin runner |
| `UnitreeG1RoughEnvCfg.load_amp_data()` | 条件 LAFAN 数据加载方法 |

---

## 7. Mirror 增强

使用 `amp_data_loader.create_mirror_config()` 从 G1 29dof 的左右关节名自动生成：

- `mirror_indices`: 左右关节索引交换（如 `left_hip_yaw → right_hip_yaw`）
- `mirror_signs`: roll/yaw 关节取反（物理对称性）
- `root_angle_vel` 镜像: `[-ωx, ωy, -ωz]` → 取反 y、z
- `proj_grav` 镜像: `[gx, -gy, gz]` → 取反 y

效果：每个 motion clip 自动生成一个镜像版本，训练数据量翻倍。
