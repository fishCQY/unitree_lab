# MuJoCo Sim2Sim 与 Wandb 集成：unitree_lab vs bfm_training 差异与迁移

本文档对比 `unitree_lab` 和 `bfm_training` 两个项目中 MuJoCo Sim2Sim 评估流程与 Wandb 日志的实现差异，
解释 bfm_training 各设计决策的原因，并记录将 unitree_lab 迁移到 bfm_training 方案的变更。

---

## 1. 架构概览

### 1.1 bfm_training：Runner 内置 + 批量评估

```
训练循环 (checkpoint 保存)
    │
    ↓
LightOnPolicyRunner._save_checkpoint()
    ├─ 导出 ONNX + attach_onnx_metadata
    ├─ wandb.save(onnx)
    └─ env.unwrapped.mujoco_eval.batch_eval_and_log(onnx_path, iteration, num_workers=16)
           │
           ↓
       BaseMuJoCoEval.batch_eval_and_log()
           ├─ batch_eval()  →  run_batch_eval()
           │     ├─ Phase 1: _run_batch_headless()      ← 全部任务，无渲染，ProcessPoolExecutor 并行
           │     └─ Phase 2: _run_video_tasks_serial()   ← 选中任务（mixed_terrain + worst N），串行录视频
           └─ _log_to_wandb(result, iteration)
                 ├─ wandb.log(result.to_wandb_dict(), commit=False)    ← 结构化指标
                 └─ _upload_videos_to_wandb()                          ← 分类视频
                       ├─ sim2sim_video (mixed_terrain)
                       └─ sim2sim_video_worst_1, sim2sim_video_worst_2
```

### 1.2 unitree_lab（修改前）：子进程外挂 + 单视频上传

```
训练循环 (checkpoint 保存)
    │
    ↓
runner.save() → _save_with_sim2sim() [monkey-patch]
    ├─ _export_policy_to_onnx()
    ├─ build_onnx_metadata → dump deploy.yaml
    └─ subprocess.Popen("python run_sim2sim_locomotion.py --save-video ...")
           │ (后台线程等待)
           ↓
       _sim2sim_poller()
           └─ _log_sim2sim_video_to_wandb(mp4, iteration)
                 └─ wandb.log({"sim2sim_video": Video(...)}, commit=True)
                    wandb.save(mp4)
```

---

## 2. 关键设计差异与原因

### 2.1 触发方式：Runner 内置 vs 子进程外挂

| | bfm_training | unitree_lab（旧） |
|---|---|---|
| 方式 | Runner 内部直接调用 | monkey-patch `runner.save()`，启动子进程 |
| 优点 | 简洁可靠，无进程通信问题，错误可直接 catch | 不修改 runner 代码 |
| 缺点 | 需要 runner 有 mujoco_eval 钩子 | 子进程管理复杂，信号量竞争，日志分散 |

**为何 bfm_training 这样做**：Sim2Sim 评估是训练的核心反馈环，不应作为外部附加组件。
Runner 内部触发可以确保：
- ONNX 导出和评估的原子性（同一个 checkpoint 周期内完成）
- 错误处理清晰（`try/except` 不影响训练继续）
- 不需要线程、信号量等并发原语

### 2.2 评估流程：两阶段批量 vs 单任务单视频

| | bfm_training | unitree_lab（旧） |
|---|---|---|
| Phase 1 | 全部任务无渲染并行评估（ProcessPoolExecutor） | 无 |
| Phase 2 | 选中任务串行录视频 | 单任务录一个视频 |
| 任务选择 | mixed_terrain + 表现最差的 N 个任务 | CLI 固定指定一个任务 |
| 并行度 | 最多 16 worker，EGL 按 GPU 数量限制 | 1 个子进程 |

**为何两阶段**：
- Phase 1 无渲染是为了最大化吞吐量——MuJoCo 无渲染模式不需要 GPU，可以纯 CPU 并行。
  这样 10+ 个评估任务可以在 ~10 秒内全部完成。
- Phase 2 串行是因为 MuJoCo 渲染（特别是 EGL）不适合多进程并发——
  OpenGL context 竞争会导致段错误或空白帧。串行保证视频质量。
- 选择 mixed_terrain + worst 任务是为了在 Wandb 中既展示综合能力，又暴露薄弱环节。

### 2.3 Wandb 日志：结构化指标 vs 仅视频

| | bfm_training | unitree_lab（旧） |
|---|---|---|
| 指标上传 | `sim2sim_eval/{task}/survival_rate`<br>`sim2sim_eval/{task}/linear_velocity_error`<br>`sim2sim_eval/{task}/angular_velocity_error` | 无指标上传 |
| 视频上传 | `sim2sim_video` (mixed_terrain)<br>`sim2sim_video_worst_1`, `sim2sim_video_worst_2` | `sim2sim_video`（单个） |
| commit 策略 | `commit=False`（让训练循环统一 commit） | `commit=True`（强制推进 step） |
| caption | 含 iteration 标签 | 无 caption |

**为何 `commit=False`**：
Wandb 的 step 是全局递增的。如果 sim2sim 用 `commit=True` 上传视频，
会在训练 loss 曲线中插入一个不属于训练循环的 step，导致：
- 训练曲线出现"跳跃"（step 不连续）
- 多次 `wandb.log` 的指标在同一个 step 下不完整（部分 commit 了，部分没有）

正确做法是所有 sim2sim 的 `wandb.log` 都用 `commit=False`，让训练循环在 `runner.learn()`
内部的正常日志流程中统一 commit。

### 2.4 PD 控制：Position Servo（隐式）vs Explicit PD

| | bfm_training (`BaseMujocoSimulator`) | unitree_lab (`LocomotionMujocoSimulator`) |
|---|---|---|
| 方式 | 将 MuJoCo actuator 重配置为 position servo，`data.ctrl` 写目标位置 | 手动计算 τ = Kp(q_target - q) - Kd·dq，写入 `data.ctrl` |
| integrator | `mjINT_IMPLICITFAST` | `mjINT_IMPLICITFAST`（但 PD 是 explicit） |
| dof_damping | 清零（避免双重阻尼） | 不清零 |
| dof_frictionloss | 清零 | 不清零 |

**为何用 Position Servo**：
IsaacLab 训练时用 PhysX 的 **implicit actuator**——PD 力在约束求解器内部计算，
与接触力、关节限制等同时求解。这给予极高的关节刚度而不损失数值稳定性。

MuJoCo 的 explicit PD（通过 `qfrc_applied` 或 `data.ctrl` 写力矩）在数学上是等价的，
但因为力是在积分步**之前**计算的，同样的 Kp/Kd 值会产生明显不同的行为：
- 高 Kp 在 explicit 模式下容易导致数值振荡
- 机器人在站立时"软腿"——因为 explicit PD 的等效刚度比 implicit 低

MuJoCo 的 position servo（`gaintype=0, biastype=1`）将 PD 法则嵌入 MuJoCo 自己的
implicit 积分器中，效果与 PhysX implicit actuator 一致。配置方式：

```
gainprm[0] = Kp                  → force += Kp * ctrl
biasprm[1] = -Kp                 → force += -Kp * q_joint
biasprm[2] = -Kd                 → force += -Kd * dq_joint
```

此时 `data.ctrl[i] = q_target`，MuJoCo 计算的力为：
`force = Kp * q_target - Kp * q_current - Kd * dq = Kp * (q_target - q) - Kd * dq`

**清零 dof_damping 的原因**：
MuJoCo XML 中 joint 的 `damping` 属性会在 DoF 层面额外施加 `-damping * dq` 的力。
如果同时使用 position servo（biasprm[2] = -Kd），就会有两倍的速度阻尼，
导致关节速度远小于训练时 IsaacLab 中观测到的值，策略行为失配。

### 2.5 角速度来源：`data.cvel` vs `data.qvel[3:6]`

| | bfm_training | unitree_lab |
|---|---|---|
| 来源 | `data.cvel[body_id, 0:3]`（world frame） | `data.qvel[3:6]`（body frame） |
| 后处理 | ObservationBuilder 做 world→body 变换 | 直接使用 |
| decimation 后刷新 | 调 `mj_forward` 刷新 `cvel` | 不刷新 |

**为何用 `cvel`**：
`data.qvel[3:6]` 对于 free joint 是 **body-frame** 角速度，可以直接用于观测。
但问题是 `mj_step` 之后 `data.cvel` 和 `data.subtree_linvel` 等衍生量是 stale 的——
它们反映的是上一步的状态。如果不调 `mj_forward` 刷新，观测会滞后一个物理步。

bfm_training 的 `BaseMujocoSimulator.step()` 在 decimation 循环后调用 `mj_forward()`，
确保 `cvel` 是最新的。unitree_lab 的 `LocomotionMujocoSimulator` 不做这个刷新，
导致角速度观测滞后。

### 2.6 GL 后端与多 GPU 管理

| | bfm_training | unitree_lab |
|---|---|---|
| GL 选择 | 自动：ONNX 有 depth → EGL，否则 → osmesa | 无管理 |
| GPU 分配 | `MUJOCO_EGL_DEVICE_ID` 按 worker 轮询 | 无 |
| MP context | forkserver/spawn 自动选择 | 无（单进程） |

**为何这样做**：
- osmesa 是软件渲染，不需要 GPU，适合无深度观测的策略评估
- EGL 需要 GPU，多 worker 共享同一 GPU context 会死锁，因此需要 `MUJOCO_EGL_DEVICE_ID`
- `forkserver` 避免了 `fork` 在 CUDA 初始化后的不安全行为

---

## 3. unitree_lab 中发现的 Bug

### 3.1 `_find_sim2sim_resources` 路径硬编码（严重）

```python
# train.py:242
rl_lab_root = Path.home() / "unitree_lab"   # ← 硬编码 ~/unitree_lab
```

实际项目在 `~/bfm/unitree_lab`，导致 sim2sim 永远找不到脚本，被静默禁用。

**修复**：改为基于 `__file__` 解析项目根。

### 3.2 `run_sim2sim_from_xml.py` 导入不存在的模块（严重）

```python
# run_sim2sim_from_xml.py:68
from simulator import _deploy_yaml_to_config_override, _load_deploy_yaml, run_locomotion_simulation
```

项目中不存在 `simulator.py`，应导入 `run_sim2sim_locomotion`。

### 3.3 `LocomotionMujocoSimulator` 与 `BaseMujocoSimulator` 接口不兼容（严重）

`LocomotionMujocoSimulator` 引用了 `BaseMujocoSimulator` 中不存在的多个属性和方法：
- `joint_mapping_index`（Base 中是 `joint_mapping`）
- `is_wheel_mask`, `reset_hidden_states()`, `step_inference()`, `expected_obs_dim`, `step_physics()`
- `tau_limits` 形状 `(N,2)` vs 期望 `(N,)`

当前 `LocomotionMujocoSimulator` 无法实例化。

### 3.4 Wandb `commit=True` 打乱 step 对齐（中等）

```python
# train.py:269
wandb.log({"sim2sim_video": wandb.Video(...)}, step=cur_step, commit=True)
```

应改为 `commit=False`。

### 3.5 `BatchEvaluator` 视频 FPS 硬编码为 60（低）

录制的每帧对应一个 policy step（~50Hz），60fps 回放会偏快。
应根据 `policy_dt` 计算正确 FPS。

### 3.6 `spawn_root_z_offset` 仅在 teleop 模式设置（低）

非遥操作 + heightfield 场景下机器人可能嵌入地形。

---

## 4. 迁移方案

将 unitree_lab 的 sim2sim + wandb 集成改为 bfm_training 的方式：

1. **修复 `_find_sim2sim_resources`**：基于 `__file__` 解析路径
2. **修复 `run_sim2sim_from_xml.py`**：修正导入路径
3. **修复 `LocomotionMujocoSimulator`**：对齐 `BaseMujocoSimulator` 接口
4. **修复 `_log_sim2sim_video_to_wandb`**：`commit=True` → `commit=False`
5. **修复 `BatchEvaluator` 视频 FPS**：基于 `policy_dt` 计算
6. **修复 `spawn_root_z_offset`**：统一在 reset 前设置

---

## 5. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `scripts/rsl_rl/train.py` | 修改 | 修复路径硬编码，wandb commit 策略 |
| `scripts/mujoco_eval/run_sim2sim_from_xml.py` | 修改 | 修正 import 路径 |
| `mujoco_utils/simulation/locomotion_simulator.py` | 修改 | 对齐 BaseMujocoSimulator API |
| `mujoco_utils/evaluation/batch_evaluator.py` | 修改 | 视频 FPS 基于 policy_dt |
