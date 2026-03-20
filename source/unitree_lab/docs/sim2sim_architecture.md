# unitree_lab MuJoCo Sim2Sim 架构说明

本文档描述 unitree_lab 中 MuJoCo Sim2Sim 的完整实现方式，涵盖四层架构、
关键设计决策及其原因、使用方式。

---

## 1. 四层架构总览

```
┌─────────────────────────────────────────────────────────┐
│  第 4 层：脚本入口  scripts/mujoco_eval/                │
│    run_sim2sim_locomotion.py   交互/无头/视频           │
│    run_sim2sim_from_xml.py     直接加载 XML+ONNX        │
├─────────────────────────────────────────────────────────┤
│  第 3 层：评估框架  mujoco_utils/evaluation/            │
│    MuJoCoEval          训练集成（batch_eval_and_log）   │
│    BatchEvaluator      多任务评估 + 视频录制            │
│    eval_task.py         10 种预定义评估任务              │
│    metrics.py           存活率/速度误差/距离 等指标      │
├─────────────────────────────────────────────────────────┤
│  第 2 层：仿真器  mujoco_utils/simulation/              │
│    BaseMujocoSimulator        核心仿真器 (~1014 行)     │
│    LocomotionMujocoSimulator  locomotion 薄封装         │
│    ObservationBuilder         观测构建器                │
├─────────────────────────────────────────────────────────┤
│  第 1 层：核心工具  mujoco_utils/core/                  │
│    onnx_utils.py    ONNX 加载 + IsaacLab 元数据解析     │
│    physics.py       PD 控制 / 四元数 / 重力投影         │
│    xml_parsing.py   关节映射 / actuator 解析            │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 第 1 层：核心工具 (`mujoco_utils/core/`)

### 2.1 onnx_utils.py

- **`OnnxConfig`**：从 ONNX 文件的 metadata_props 中解析 IsaacLab 导出的配置
  - 关节名（`joint_names`）、PD 增益（`joint_stiffness` / `joint_damping`）
  - `action_scale` / `action_offset` / `default_joint_pos`
  - 观测结构（`observation_names` / `observation_dims`）、历史长度
  - 时间步（`sim_dt` / `decimation`）
  - 隐藏状态维度（GRU/LSTM 策略）
- **`OnnxInference`**：推理封装
  - 前馈策略：`policy(obs)` → action
  - 循环策略：维护 hidden state，支持 `reset_hidden_state()`

### 2.2 physics.py

```python
pd_control(target_q, current_q, current_dq, kp, kd, tau_limits)
    → τ = Kp * (q_target - q) - Kd * dq，clamp 到力矩限制

pd_control_velocity(target_dq, current_dq, kd, tau_limits)
    → τ = Kd * (dq_target - dq)，用于轮式关节

quat_rotate_inverse(quat, vec)
    → 将 world frame 向量旋转到 body frame

apply_onnx_physics_params(model, armature, damping, friction)
    → 将 ONNX 元数据中的物理参数写入 MuJoCo model
```

### 2.3 xml_parsing.py

```python
build_joint_mapping(onnx_joint_names, mujoco_actuator_names)
    → 返回 mapping[i] = j，即 MuJoCo actuator i 对应 ONNX joint j

get_actuator_names(xml_path)
    → 从 XML 解析 actuator 名称列表
```

---

## 3. 第 2 层：仿真器 (`mujoco_utils/simulation/`)

### 3.1 BaseMujocoSimulator 初始化流程

```
__init__(xml_path, onnx_path, config_override)
    │
    ├─ 1. 加载 ONNX config（get_onnx_config）
    ├─ 2. 加载 MuJoCo model + data
    ├─ 3. 查找 free joint → base_body_id（用于 data.cvel 读取速度）
    │
    ├─ 4. 配置时间步
    │       sim_dt     = onnx_config.sim_dt        (默认 0.005s)
    │       decimation = onnx_config.decimation     (默认 4)
    │       policy_dt  = sim_dt × decimation        (默认 0.02s = 50Hz)
    │
    ├─ 5. 设置积分器 → mjINT_IMPLICITFAST
    ├─ 6. 求解器迭代 → iterations=50, ls_iterations=50
    │
    ├─ 7. _setup_actuator_state_mapping()
    │       从 model.actuator_trnid 构建每个 actuator 的：
    │         - joint id
    │         - qpos 地址
    │         - dof 地址
    │         - 力矩限制（jnt_actfrcrange）
    │
    ├─ 8. _setup_joint_mapping()
    │       ONNX joint 名 ←→ MuJoCo actuator 名 的映射
    │       joint_mapping[i] = j  →  actuator i 对应 ONNX joint j
    │       inv_joint_mapping[j] = i  →  反向映射
    │
    ├─ 9. _apply_physics_params()
    │       armature / damping / friction → 写入 MuJoCo model
    │
    ├─ 10. _setup_pd_gains()
    │       Kp = joint_stiffness[joint_mapping]
    │       Kd = joint_damping[joint_mapping]
    │       action_scale = action_scale[joint_mapping]
    │       default_joint_pos = default_joint_pos[joint_mapping]
    │
    ├─ 11. _configure_position_servos()   ← ★ 关键步骤
    │       详见 §3.2
    │
    ├─ 12. _configure_contact_params()
    │       详见 §3.3
    │
    ├─ 13. 加载 OnnxInference 策略
    └─ 14. 创建 ObservationBuilder
```

### 3.2 Position Servo 配置（_configure_position_servos）

**为什么不用 explicit PD？**

IsaacLab 训练时用 PhysX 的 **implicit actuator**——PD 力在约束求解器内部与接触力同时求解。
MuJoCo 的 explicit PD（手动算 τ = Kp(e) - Kd(dq) 再写入 `data.ctrl`）在数学上等价，
但因为力是在积分步**之前**计算的，同样的 Kp/Kd 值会导致：

- 高 Kp 产生数值振荡
- 机器人站立时"软腿"——等效刚度比 implicit 模式低

**解决方案：Position Servo**

将每个 motor actuator 重配为 position servo：

```
gainprm[0] = Kp          → force += Kp × ctrl
biasprm[1] = -Kp         → force += -Kp × q_joint
biasprm[2] = -Kd         → force += -Kd × dq_joint
```

此后 `data.ctrl[i]` 接受**目标关节位置**（不是力矩）。MuJoCo 计算：

```
force = Kp × q_target - Kp × q_current - Kd × dq
      = Kp × (q_target - q) - Kd × dq
```

PD 法则完全嵌入 MuJoCo 的隐式积分器中，与 PhysX 行为一致。

**附加处理：清零 dof_damping 和 dof_frictionloss**

MuJoCo XML 中 joint 的 `damping` 属性会额外施加 `-damping × dq` 力。
如果同时使用 position servo（biasprm[2] = -Kd），就会有**双重速度阻尼**，
导致关节速度远小于训练时的值。因此必须清零。

### 3.3 接触参数配置（_configure_contact_params）

| 参数 | 设置 | 原因 |
|------|------|------|
| 摩擦锥 | `mjCONE_PYRAMIDAL` | PhysX 使用 pyramid 模型 |
| noslip_iterations | 10 | 防止脚底滑动 |
| solref | `[0.005, 1.0]` | 比 MuJoCo 默认 `[0.02, 1.0]` 更硬，接近 PhysX |
| solimp | `[0.95, 0.99, 0.001, 0.5, 2.0]` | 更高阻抗 |
| floor 摩擦 | `[1.0, 0.005, 0.0001]` | 匹配 IsaacLab terrain physics_material |
| robot geom 摩擦 | `[1.0, 0.005, 0.0001]` | 匹配 rigid_body_material 默认值 |

### 3.4 单步执行流程（step 方法）

```
step(action=None)
    │
    ├─ 若 action 为空：
    │     obs = build_observation()
    │     action = policy(obs)           ← ONNX 推理，ONNX 关节顺序
    │
    ├─ _last_action = action.copy()      ← 保存用于下次观测反馈
    │
    ├─ 关节顺序转换：ONNX → MuJoCo actuator
    │     action_act = action[joint_mapping]
    │
    ├─ 计算目标位置：
    │     target_q = offset + action_act × scale
    │     （若有 action_offset 用 offset，否则用 default_joint_pos）
    │
    ├─ data.ctrl[:] = target_q           ← 写目标位置到 position servo
    │
    ├─ for _ in range(decimation):
    │       mujoco.mj_step()             ← MuJoCo 隐式 PD 求解
    │
    ├─ mujoco.mj_forward()               ← ★ 刷新 cvel 等衍生量
    │     确保 base_ang_vel / base_lin_vel 读到最新值
    │
    ├─ episode_length += 1
    └─ 返回 (build_observation(), info)
```

### 3.5 Reset 流程

```
reset(initial_state=None)
    │
    ├─ mj_resetData()
    ├─ 设置关节到 default_joint_pos
    ├─ 若有 spawn_root_z_offset → 抬升 base z（heightfield 场景）
    ├─ mj_forward()
    │
    ├─ ★ 反穿透保护：
    │     估算所有碰撞 geom 的最低点
    │     若低于地面 → 自动抬升 base
    │
    ├─ 重置 episode_length / phase / last_action
    ├─ policy.reset_hidden_state()
    ├─ obs_builder.reset()
    └─ 返回 build_observation()
```

### 3.6 角速度与线速度来源

| 量 | 来源 | 说明 |
|----|------|------|
| `base_ang_vel` | `data.cvel[body_id, 0:3]` | world frame，`mj_forward` 后刷新 |
| `base_lin_vel` | `data.cvel[body_id, 3:6]` | world frame |
| 备用 | `qvel[3:6]` 旋转到 world | 当 `_base_body_id` 不可用时 |

为什么用 `cvel` 而非 `qvel[3:6]`：`mj_step` 后 `cvel` 是 stale 的，必须调 `mj_forward` 刷新。
Base 中 `step()` 在 decimation 循环后始终调用 `mj_forward()`，保证观测不滞后。

### 3.7 ObservationBuilder

支持的观测项（从 ONNX 元数据自动推断）：

| 观测项 | 维度 | 说明 |
|--------|------|------|
| `base_ang_vel` | 3 | body frame 角速度 × scale（默认 0.25） |
| `projected_gravity` | 3 | body frame 重力投影 |
| `velocity_commands` | 3 | [vx, vy, wz] |
| `joint_pos` / `joint_pos_rel` | N | 关节位置（绝对/相对默认姿态） |
| `joint_vel` / `joint_vel_rel` | N | 关节速度 × scale（默认 0.05） |
| `last_action` / `actions` | N | 上一步动作反馈 |
| `gait_phase` | 2 | sin/cos 步态相位 |
| `height_scan` | M | 高度扫描点 |
| `base_lin_vel` | 3 | body frame 线速度 |

支持历史堆叠（history stacking）：当 `history_length > 1` 时自动拼接多帧观测。

### 3.8 LocomotionMujocoSimulator

`BaseMujocoSimulator` 的薄封装，仅添加：

- **跌倒检测**：`base_pos[2] < 0.12m` 或倾斜角 > 57°（arccos(-proj_g_z) > 1.0 rad）
- **`run_episode()`**：便捷方法，跑一个 episode 返回 `{steps, terminated, mean_velocity_error, forward_distance}`

---

## 4. 第 3 层：评估框架 (`mujoco_utils/evaluation/`)

### 4.1 预定义评估任务（eval_task.py）

| 任务名 | 地形 | 速度指令 | 最大步数 |
|--------|------|----------|----------|
| `flat_forward` | flat | (0.5, 0, 0) | 500 |
| `flat_backward` | flat | (-0.3, 0, 0) | 500 |
| `flat_lateral` | flat | (0, 0.3, 0) | 500 |
| `flat_turn` | flat | (0, 0, 0.5) | 500 |
| `flat_fast` | flat | (1.0, 0, 0) | 500 |
| `rough_forward` | course | (0.5, 0, 0) | 3000 |
| `stairs_up` | pyramid_stairs | (0.5, 0, 0) | 3000 |
| `stairs_down` | pyramid_stairs_inv | (0.5, 0, 0) | 3000 |
| `slope_up` | pyramid_sloped | (0.5, 0, 0) | 3000 |
| `mixed_terrain` | mixed | 随机 | 1000 |

### 4.2 MuJoCoEval — 训练集成入口

```
MuJoCoEval(robot_xml_path, eval_task_names, num_worst_videos, save_mixed_terrain_video)
```

**`batch_eval_and_log(onnx_path, iteration)` 流程**：

```
batch_eval_and_log()
    │
    ├─ Phase 1: batch_eval()
    │     遍历所有任务：
    │       创建 BaseMujocoSimulator → run_episode() × num_episodes → 收集指标
    │     返回 BatchEvalResult（含每个任务的 LocomotionMetrics）
    │
    ├─ Phase 2: 选择视频任务
    │     ├─ mixed_terrain（总是录）
    │     └─ survival_rate 最低的 N 个任务
    │     └─ _record_task_video()
    │           ├─ 创建 simulator + tracking camera
    │           ├─ 逐帧渲染到内存
    │           └─ ffmpeg pipe → H.264 MP4（cv2 兜底）
    │
    └─ _log_to_wandb(result, iteration)
          所有 wandb.log 使用 commit=False
          ├─ 结构化指标：
          │     sim2sim_eval/{task}/survival_rate
          │     sim2sim_eval/{task}/mean_velocity_error
          │     sim2sim_eval/{task}/mean_forward_distance
          │     sim2sim_eval/{task}/velocity_error_x
          │     sim2sim_eval/{task}/velocity_error_y
          │
          ├─ sim2sim_video         ← mixed_terrain 视频
          ├─ sim2sim_video_worst_1 ← 最差任务 #1 视频
          └─ sim2sim_video_worst_2 ← 最差任务 #2 视频
```

**为什么 `commit=False`**：让训练循环统一 commit，避免在 loss 曲线中插入不属于训练步的 step。

### 4.3 视频录制

- 分辨率：1280×720
- FPS：基于 `policy_dt` 计算（`1 / policy_dt`，通常 50fps）
- 编码：优先 ffmpeg pipe（H.264），兜底 cv2 mp4v
- 相机：tracking camera 跟随机器人

---

## 5. 第 4 层：脚本入口 (`scripts/mujoco_eval/`)

### 5.1 run_sim2sim_locomotion.py — 完整独立工具

**使用方式**：

```bash
# 交互式（MuJoCo viewer + 键盘遥操作）
python scripts/mujoco_eval/run_sim2sim_locomotion.py \
    --robot g1 --onnx policy.onnx --render --teleop keyboard --follow

# 无头批量评估
python scripts/mujoco_eval/run_sim2sim_locomotion.py \
    --robot g1 --onnx policy.onnx --task mixed_terrain --num-episodes 10

# 录制视频
python scripts/mujoco_eval/run_sim2sim_locomotion.py \
    --robot g1 --onnx policy.onnx --task flat_forward --save-video --output-dir videos/

# 指定地形 + deploy 配置
python scripts/mujoco_eval/run_sim2sim_locomotion.py \
    --robot g1 --onnx policy.onnx --xml scene_terrain.xml \
    --deploy-yaml deploy_latest.yaml --task rough_forward
```

**功能**：
- 自动查找 robot XML（支持 flat / terrain 两种场景）
- 地形注入：course 类型生成带 box geom 的临时 XML，heightfield 注入高度数据
- 键盘遥操作：UP/DOWN = vx，PgUp/PgDn = vy，LEFT/RIGHT = wz
- deploy.yaml 加载 PD 增益 / 时间步 / 关节配置覆盖

### 5.2 run_sim2sim_from_xml.py — 简化版

直接加载已有 XML + ONNX，不注入地形（使用 XML 中已有的地形）：

```bash
python scripts/mujoco_eval/run_sim2sim_from_xml.py \
    --onnx policy.onnx \
    --xml scene_29dof_terrain.xml \
    --deploy-yaml deploy_latest.yaml \
    --render --follow --teleop keyboard
```

### 5.3 训练中自动触发（train.py）

```
训练循环
    │
    ├─ runner.save() → _save_with_sim2sim()  [monkey-patch]
    │     ├─ 导出 ONNX → export/policy_iter_{N}.onnx
    │     ├─ 构建 deploy 元数据 → export/deploy_latest.yaml
    │     └─ subprocess.Popen 启动 run_sim2sim_locomotion.py
    │           --save-video --velocity 1.0 0.0 0.0
    │
    └─ 后台线程 _sim2sim_poller() 等待子进程完成
          └─ _log_sim2sim_video_to_wandb(mp4, iteration)
                wandb.log(sim2sim_video, commit=False)
```

路径解析基于 `__file__` 计算项目根，信号量限制同时只有 1 个 sim2sim 子进程。

---

## 6. 关键设计决策汇总

| 设计点 | 选择 | 原因 |
|--------|------|------|
| PD 控制 | Position servo（隐式） | 匹配 IsaacLab/PhysX implicit actuator，避免 explicit PD "软腿" |
| `data.ctrl` 语义 | 目标关节位置 | position servo 模式下 ctrl 是 q_target |
| 积分器 | `mjINT_IMPLICITFAST` | 隐式积分处理高刚度 PD 不振荡 |
| dof_damping | 清零 | 避免与 PD servo 双重阻尼 |
| 摩擦锥 | Pyramidal | PhysX 使用 pyramid 模型 |
| 接触刚度 | solref=[0.005, 1.0] | 比 MuJoCo 默认更硬，接近 PhysX |
| 角速度来源 | `data.cvel` + `mj_forward` | 保证 decimation 后读到最新值 |
| 反穿透 | reset 时自动检测抬升 | 防止默认姿态导致脚嵌入地面 |
| Wandb commit | `commit=False` | 不打乱训练 loss 曲线的 step 对齐 |
| 视频 FPS | `1 / policy_dt` | 确保回放速度与真实时间一致 |
| 视频选择 | mixed_terrain + worst N | 综合能力 + 暴露薄弱环节 |

---

## 7. 文件索引

```
unitree_lab/
├── scripts/
│   ├── mujoco_eval/
│   │   ├── run_sim2sim_locomotion.py    # 独立 sim2sim 工具（交互/无头/视频）
│   │   └── run_sim2sim_from_xml.py      # 简化版（直接加载 XML）
│   └── rsl_rl/
│       └── train.py                     # 训练脚本（含 sim2sim 子进程集成）
│
└── source/unitree_lab/unitree_lab/mujoco_utils/
    ├── core/
    │   ├── onnx_utils.py                # OnnxConfig + OnnxInference
    │   ├── physics.py                   # PD 控制 / 四元数 / 重力投影
    │   └── xml_parsing.py               # 关节映射 / actuator 解析
    ├── simulation/
    │   ├── base_simulator.py            # BaseMujocoSimulator（核心，~1014 行）
    │   ├── locomotion_simulator.py      # LocomotionMujocoSimulator（薄封装）
    │   └── observation_builder.py       # ObservationBuilder（自动推断观测结构）
    ├── evaluation/
    │   ├── mujoco_eval.py               # MuJoCoEval（训练集成 + Wandb 上传）
    │   ├── batch_evaluator.py           # BatchEvaluator（多任务评估）
    │   ├── eval_task.py                 # 10 种预定义评估任务
    │   └── metrics.py                   # LocomotionMetrics 指标计算
    ├── sensors/
    │   ├── height_scanner.py            # 高度扫描
    │   └── contact_detector.py          # 接触检测
    └── terrain/
        ├── generator.py                 # 地形生成
        └── xml_generation.py            # 地形 XML 生成
```
