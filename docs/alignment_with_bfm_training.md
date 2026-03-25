# unitree_lab 与 bfm_training 架构对齐报告

## 概述

本次对齐工作将 `bfm_training`（main 分支）中通用的架构设计和优秀实现移植到 `unitree_lab`。`unitree_lab` 主要使用 G1 机器人进行训练，`bfm_training` 使用其他本体。两者在实现细节上有差异，因此借鉴的是**架构模式**和**可通用的实现**，而非直接复制特定本体的配置。

---

## 一、变更文件清单

### 第一轮新增文件（9 个新模块）

| 文件路径 | 功能 |
|----------|------|
| `mujoco_utils/logging.py` | 统一的 mujoco_utils 日志系统 |
| `mujoco_utils/core/joint_mapping.py` | ONNX ↔ MuJoCo 关节映射工具 |
| `mujoco_utils/core/math_utils.py` | 四元数/SE3 变换数学工具 |
| `mujoco_utils/terrain/setup.py` | 地形环境一站式设置 |
| `mujoco_utils/evaluation/mujoco_eval_cfg.py` | MuJoCo 评测配置基类 |
| `mujoco_utils/visualization/{__init__,panels}.py` | 扭矩/外感受/合成可视化面板 |
| `envs/{__init__,unitree_rl_env*}.py` | UnitreeRLEnv + UnitreeRLEnvCfg |
| `utils/unitree_on_policy_runner.py` | UnitreeOnPolicyRunner |
| `rsl_rl/isaaclab_rl/{__init__,exporter}.py` | JIT/ONNX 策略导出 |

### 第二轮新增/重写（核心评测架构）

| 文件路径 | 功能 |
|----------|------|
| `mujoco_utils/evaluation/eval_task.py` | **重写**：地形预设 + callable 速度命令 + 多级任务集 |
| `mujoco_utils/evaluation/metrics.py` | **重写**：新增 MetricsCollector 流式采集 + EvalResult + 保留 legacy API |
| `mujoco_utils/evaluation/batch_evaluator.py` | **重写**：ProcessPoolExecutor 多进程并行评测 |
| `mujoco_utils/evaluation/mujoco_eval.py` | **重写**：使用 `run_batch_eval` 代替串行评测 |
| `mujoco_utils/simulation/base_simulator.py` | **新增**：`SimulatorWithRunLoop` 模板方法类 |
| `tasks/locomotion/mujoco_eval/__init__.py` | **新增**：G1 locomotion 评测子包 |
| `tasks/locomotion/mujoco_eval/g1_eval_cfg.py` | **新增**：G1LocomotionEvalCfg 配置 |
| `tasks/locomotion/mujoco_eval/simulator.py` | **新增**：G1LocomotionSimulator + run_locomotion_simulation |

### 修改文件

| 文件路径 | 修改内容 |
|----------|----------|
| `mujoco_utils/__init__.py` | 新增 SimulatorWithRunLoop/EvalResult/MetricsCollector/BatchEvalResult 等导出 |
| `mujoco_utils/evaluation/__init__.py` | 重写导出列表，使用新的 BatchEvalResult/EvalResult |
| `tasks/locomotion/robots/g1/__init__.py` | 新增 v1 版 UnitreeRLEnv gym 注册 |
| `scripts/mujoco_eval/run_sim2sim_locomotion.py` | 集成 create_combined_visualization 到视频录制 |

---

## 二、第二轮借鉴内容详细说明

### 1. eval_task.py — 完整评测任务系统

**借鉴点**：从 bfm_training 移植完整的评测任务定义系统。

**新增内容**：
- **Callable 速度命令**：`_const`、`_cyclic`、`_sequence`、`with_warmup` 等工具函数
- **30+ 原子速度命令**：`vel_cmd_forward_slow`、`vel_cmd_omnidirectional`、`vel_cmd_chaos` 等
- **地形预设字典**：`TERRAIN_FLAT`、`TERRAIN_ROUGH`、`TERRAIN_STAIRS_UP`、`TERRAIN_MIXED` 等 30+ 种配置
- **EvalTask dataclass**：统一 `terrain` + `vel_cmd_fn` + `duration` + `warmup`
- **三级任务集**：`EVAL_TASKS_DEFAULT`（默认精简）、`EVAL_TASKS_BABY`（简单）、`EVAL_TASKS_FULL`（全面）

**好处**：
- 旧版 `LocomotionEvalTask` 使用固定速度命令（tuple），无法测试动态指令跟踪
- 新版支持时变速度函数，能测试急停、之字形、全方位运动等复杂场景
- 地形预设字典格式与 `MujocoTerrainGenerator` 直接兼容
- warmup 自动包装，评测指标排除初始稳态阶段

### 2. metrics.py — 流式指标采集

**借鉴点**：`MetricsCollector` + `EvalResult` + `is_fallen` 函数。

**新增内容**：
- `MetricsCollector`：逐步记录 projected_gravity / cmd_velocity / actual_velocity / torque，最后一次性计算
- `EvalResult`：统一的评测结果结构（survival_rate, lin_vel_error, ang_vel_error, avg_torque_util）
- `is_fallen`：IsaacLab 兼容的跌倒检测

**好处**：
- 旧版 `compute_locomotion_metrics` 需要在 episode 结束后从 episode data 中后计算
- 新版 `MetricsCollector` 逐步采集，内存友好，支持 streaming 更新
- 跌倒检测与 IsaacLab 训练环境一致（orientation + contact force）
- 支持保存详细力矩数据到 `.npz`

### 3. batch_evaluator.py — 多进程并行评测

**借鉴点**：`ProcessPoolExecutor` 多进程并行 + 两阶段评测策略。

**核心设计**：
- **Phase 1**：所有任务无渲染并行（快速获取指标）
- **Phase 2**：选择性任务串行视频录制（mixed_terrain + worst N）
- **GL 后端自适应**：osmesa / egl 自动选择，GPU 分配
- **子进程隔离**：环境变量独立，random seed 重置
- **超时保护**：per-task timeout + 全局 timeout

**好处**：
- 旧版 `BatchEvaluator` 串行运行，8 个任务需要 8x 时间
- 新版 16 workers 并行，8 个任务约 1x 时间完成
- 两阶段策略避免不必要的渲染开销
- 环境变量隔离防止 OpenGL 上下文冲突

### 4. SimulatorWithRunLoop — 模板方法设计

**借鉴点**：继承 `BaseMujocoSimulator`，提供完整的 `run()` 循环和可覆盖的 hook 方法。

**Hook 方法**：
- `get_command()` — 返回当前速度命令
- `on_physics_step()` — 每个物理步后执行（IMU 延迟缓冲等）
- `on_control_step()` — 每个策略步后执行（指标采集）
- `render_frame()` — 自定义帧渲染（可视化叠加）
- `compute_results()` — 计算最终评测结果
- `cleanup_extras()` — 清理临时资源

**好处**：
- 旧版 `BaseMujocoSimulator` 的 `run_episode` 是硬编码的循环
- 新版模板方法使得新任务类型（操控、恢复等）只需覆盖 hook 即可
- 自动处理 renderer 创建/销毁、视频保存、窗口管理

### 5. G1LocomotionSimulator — G1 专用评测模拟器

**借鉴点**：继承 `SimulatorWithRunLoop`，专为 G1 locomotion 评测定制。

**功能**：
- 地形生成集成（`setup_terrain_env` → `setup_terrain_data_in_model`）
- 速度跟踪指标流式采集（`MetricsCollector`）
- 可视化叠加（`create_combined_visualization`）
- 临时文件自动清理

### 6. G1LocomotionEvalCfg — G1 评测配置

**借鉴点**：继承 `BaseMuJoCoEvalCfg`，提供 G1 特定的默认值。

**配置项**：
- `robot_model_path` — G1 MuJoCo XML 路径
- `eval_task_names` — 评测任务子集
- `num_worst_videos` / `save_mixed_terrain_video` — 视频策略
- `simulation_fn_path` — 指向 `run_locomotion_simulation`

### 7. UnitreeRLEnv gym.register — v1 版注册

**借鉴点**：为 G1 rough/flat 添加 `entry_point="unitree_lab.envs:UnitreeRLEnv"` 的 v1 版注册。

**好处**：
- v0 版保留不变（向后兼容）
- v1 版使用 `UnitreeRLEnv` 获得自动 ONNX 元数据和 sim2sim 集成
- 渐进迁移，不影响现有实验

### 8. 视频可视化集成

**借鉴点**：在 `run_sim2sim_locomotion.py` 的 `_record_video_headless` 和 `_render_one_frame_tracking` 中集成 `create_combined_visualization`。

**好处**：
- sim2sim 视频现在包含：扭矩利用率面板、基座速度、高度、时间、命令信息
- 交互模式录制的视频也有同样的可视化叠加
- import 失败时静默降级为原始帧

---

## 三、设计决策与保留项

### 第三轮重构：架构全面对齐 bfm_training

| 模块 | 变更 | 原因 |
|------|------|------|
| `BaseMujocoSimulator` | 改为 **ABC 抽象基类** + **显式 PD 控制** | 隐式 PD 在实际部署中几乎不会成功；ABC 强制子类实现 obs/reset |
| `onnx_utils.get_onnx_config` | 返回 **dict**（非 dataclass） | 与 bfm_training 对齐，更灵活 |
| `physics.pd_control` | 新签名含 `target_dq`，返回 `(torque, torque_util)` | 支持轮式关节，提供力矩利用率 |
| `SimulatorWithRunLoop` | 内循环改为 **物理频率**，PD 每步重算 | 与 bfm_training 一致 |

### 新增扩展模块

| 模块 | 路径 | 来源 |
|------|------|------|
| `PPO_TF` | `rsl_rl/algorithms/ppo_tf.py` | Transformer PPO 训练 |
| `RolloutStorageTF` | `rsl_rl/storage/rollout_storage_tf.py` | 滑动窗口 mini-batch |
| `ActorCriticTransformer` | `rsl_rl/modules/actor_critic_transformer*.py` | Transformer 策略 |
| `PPO_MoE` | `rsl_rl/algorithms/ppo_moe.py` | 多专家混合训练 |
| `ExpertGuidance` | `rsl_rl/isaaclab_rl/expert_guidance_cfg.py` | 专家引导配置 |
| `SymmetryClassifier` | `rsl_rl/algorithms/symmetry_classifier.py` | 对称性辅助奖励 |
| `DepthCameraRenderer` | `mujoco_utils/sensors/depth_camera.py` | MuJoCo 深度相机 |

### 保留 unitree_lab 独有优势

| 模块 | 保留内容 | 原因 |
|------|----------|------|
| `_configure_contact_params` | IsaacLab 接触参数对齐 | G1 sim2sim 稳定性 |
| `xml_parsing.build_joint_mapping` | 模糊匹配 + 排列唯一性检查 | 防止名称不一致的静默错误 |
| `AMPPluginRunner` | AMP 插件化 | PPO 保持纯净 |
| `TerrainConfig` + `TerrainType` | 结构化地形配置 | 枚举类型更安全 |
| `OnnxConfig` dataclass | 作为兼容层保留（deprecated） | 渐进迁移 |

---

## 四、架构对比图

### 对齐后

```
unitree_lab (训练)
├── Isaac Lab env → UnitreeRLEnv → RslRlVecEnvWrapper
│   ├── onnx_metadata 自动收集
│   └── mujoco_eval 集成 (G1LocomotionEvalCfg)
│       → UnitreeOnPolicyRunner
│           ├── 自动 ONNX 导出 + isaaclab_rl exporter
│           ├── wandb 代码快照
│           └── 自动 sim2sim → batch_evaluator (ProcessPoolExecutor)
│               └── G1LocomotionSimulator (SimulatorWithRunLoop)
│                   ├── 地形生成 + MetricsCollector
│                   └── create_combined_visualization

sim2sim 评测流程
├── run_batch_eval()
│   ├── Phase 1: 多进程并行无渲染 → EvalResult metrics
│   └── Phase 2: 串行视频录制 (worst + mixed_terrain)
│       └── G1LocomotionSimulator.run()
│           ├── EvalTask (callable vel_cmd + terrain presets)
│           ├── MetricsCollector (streaming)
│           └── create_combined_visualization (overlays)

独立 sim2sim 脚本
├── run_sim2sim_locomotion.py
│   ├── 交互模式 (viewer + teleop + visualization)
│   └── Headless 模式 (video + visualization overlays)
```

---

## 五、使用方式

### 启用 UnitreeRLEnv + 自动 sim2sim

在 G1 rough env cfg 中配置：
```python
from unitree_lab.tasks.locomotion.mujoco_eval import G1LocomotionEvalCfg

@configclass
class UnitreeG1RoughEnvCfg(UnitreeRLEnvCfg):
    robot_name = "unitree_g1"
    mujoco_eval = G1LocomotionEvalCfg(
        robot_model_path="/path/to/g1.xml",
    )
```

使用 v1 gym ID：
```
--task unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-v1
```

### 独立批量评测

```python
from unitree_lab.mujoco_utils.evaluation import run_batch_eval, BatchEvalConfig

config = BatchEvalConfig(num_workers=8)
result = run_batch_eval(
    onnx_path="policy.onnx",
    robot_model_path="g1.xml",
    config=config,
    video_dir="eval_videos",
)
print(result.summary())
```

### 自定义评测任务

```python
from unitree_lab.mujoco_utils.evaluation import (
    EvalTask, TERRAIN_STAIRS_UP_HARD, vel_cmd_forward_slow,
)

my_task = EvalTask(
    name="stairs_sprint",
    terrain=TERRAIN_STAIRS_UP_HARD,
    vel_cmd_fn=vel_cmd_forward_slow,
    duration=30.0,
    warmup=2.0,
)
```
