# LAFAN1 数据处理文档

## 1. 概述

本文档记录了将 [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) 中 G1 机器人的重定向数据，分别处理为 **AMP (Adversarial Motion Priors)** 和 **DeepMimic (Motion Tracking)** 两种训练格式的完整过程。

处理脚本：`scripts/process_lafan1_for_training.py`

### 1.1 数据源

LAFAN1 原始数据以 CSV 格式存储，每行一帧，30 FPS，36 列：

| 列范围 | 内容 | 形状 |
|--------|------|------|
| `[0:3]` | root position `(x, y, z)` | 3 |
| `[3:7]` | root rotation quaternion `(qx, qy, qz, qw)` | 4 |
| `[7:36]` | 29 个关节角度（URDF body-part 顺序） | 29 |

共 40 个 CSV 文件，涵盖 walk、run、sprint、dance、jump、fight、fallAndGetUp 等动作，总计 264,705 帧（约 8,824 秒）。

### 1.2 输出格式总览

| 维度 | AMP 格式 | DeepMimic 格式 |
|------|----------|----------------|
| **帧率** | 50 Hz（从 30fps 重采样） | 30 Hz（保持原始） |
| **文件组织** | 按动作类别分为多个 PKL | 全部合并为单个 PKL |
| **核心字段** | `dof_pos, dof_vel, root_angle_vel, proj_grav` | `dof_pos, dof_vel, root_pos, root_rot, root_vel, root_angle_vel, robot_points, feet_contact` |
| **用途** | AMP 判别器的离线参考数据 | MotionLib 的轨迹跟踪参考 |
| **输出路径** | `data/AMP/` | `data/Mimic/` |

---

## 2. 为什么 AMP 和 DeepMimic 的数据处理方式不同

### 2.1 算法原理的根本区别

**AMP（Adversarial Motion Priors）** 的核心思想是训练一个判别器来区分「策略产生的运动」和「参考运动数据中的运动」。策略不需要跟踪某一段特定轨迹，而是让自己的运动"看起来像"参考数据中的某种运动风格。因此：

- 判别器只需要**短时间窗口**（通常 2 帧）的特征来判断运动是否自然
- 不需要知道运动在空间中的绝对位置，只需要关节状态、角速度等**本体感知特征**
- 数据可以被切成独立的 clips，因为每次采样只取连续 2 帧

**DeepMimic（Motion Tracking）** 的核心思想是让策略**精确跟踪**参考轨迹中的每一帧。在训练时，每个 episode 会从某个参考动作的某个时间点开始，策略需要持续跟踪后续的目标状态。因此：

- 需要完整的运动轨迹，包括**全局位置和姿态**
- 需要 `robot_points`（关键身体点的世界坐标）来计算跟踪奖励
- 需要 `feet_contact` 来实现接触一致性奖励
- 需要通过**时间插值**获取任意时刻的精确状态

### 2.2 帧率处理策略不同的原因

**AMP 需要 50 Hz 的原因：**

仿真环境中，物理引擎以 200 Hz 运行（`sim.dt = 0.005s`），策略每 4 个物理步执行一次（`decimation = 4`），因此策略频率为 50 Hz。AMP 判别器直接比较策略输出的观测和参考数据的帧，两者**必须频率对齐**，否则判别器会因为时间分辨率不同而产生 artifact。

```
物理仿真: 200 Hz (sim.dt = 0.005)
        ↓ decimation = 4
策略执行: 50 Hz (step_dt = 0.02)
        ↕ 必须对齐
AMP 参考: 50 Hz (从 30fps 重采样)
```

**DeepMimic 保持 30 Hz 的原因：**

MotionLib 内部实现了**帧间插值**机制（线性插值 + SLERP），可以根据任意查询时间 $t$ 计算两个相邻帧之间的插值状态。因此原始帧率不影响查询精度，保持 30 Hz 可以减少数据量和处理开销。

### 2.3 字段需求不同的原因

| 字段 | AMP 是否需要 | DeepMimic 是否需要 | 原因 |
|------|:-----------:|:------------------:|------|
| `dof_pos` | 是 | 是 | 基本关节状态 |
| `dof_vel` | 是 | 是 | 运动学特征 |
| `root_angle_vel` | 是 | 是 | 身体旋转特征 |
| `proj_grav` | 是 | 否 | AMP 判别器需要身体姿态信息 |
| `dof_pos_rel` | 是 | 否 | 相对于默认站姿的偏移 |
| `root_pos` | 仅存储 | 是 | DeepMimic 需要全局位置做跟踪 |
| `root_vel` | 否 | 是 | DeepMimic 需要线速度做跟踪奖励 |
| `robot_points` | 可选 | 是 | DeepMimic 需要身体关键点做跟踪 |
| `feet_contact` | 否 | 是 | DeepMimic 需要接触一致性 |

AMP 判别器不需要全局位置信息，因为它只判断运动的"风格"是否自然，而不关心机器人在世界中的具体位置。这也是 AMP 的优势——策略可以自由移动，同时保持自然的运动风格。

### 2.4 文件组织不同的原因

**AMP 按动作类别分文件**是为了支持 **Conditional AMP**——不同的速度指令（walk / run）触发不同的风格判别器。训练时根据指令切换对应的参考数据集：

```python
amp_conditions = {
    "walk": ["lafan_walk_clips.pkl"],
    "run":  ["lafan_run_clips.pkl"],
}
```

**DeepMimic 合并为单文件**是因为 MotionLib 需要在所有动作中随机采样，让策略学会跟踪各种动作。

---

## 3. 数据处理流程

### 3.1 总体流程图

```
LAFAN1 CSV (30fps, 36 columns)
    │
    ├─── AMP 分支 ──────────────────────────────────────────┐
    │     │                                                  │
    │     ├── 1. 加载 CSV → {root_pos, root_rot(wxyz), dof_pos}
    │     ├── 2. 三次样条重采样 30fps → 50fps               │
    │     ├── 3. 计算派生特征:                              │
    │     │      dof_vel (中心差分)                          │
    │     │      root_angle_vel (四元数微分)                 │
    │     │      proj_grav (重力投影)                        │
    │     │      dof_pos_rel (相对默认姿态)                  │
    │     ├── 4. 转换 root_rot: wxyz → xyzw                 │
    │     └── 5. 按动作类别分组存储为 PKL                   │
    │          → data/AMP/lafan_{category}_clips.pkl         │
    │                                                        │
    └─── DeepMimic 分支 ────────────────────────────────────┐
          │                                                  │
          ├── 1. 加载 CSV → {root_pos, root_rot(wxyz), dof_pos}
          ├── 2. 保持 30fps 不重采样                        │
          ├── 3. 计算派生特征:                              │
          │      dof_vel (中心差分)                          │
          │      root_vel (位置差分 + 旋转到 base 坐标系)   │
          │      root_angle_vel (四元数微分)                 │
          ├── 4. Pinocchio FK 计算 robot_points (14 bodies) │
          ├── 5. 估计 feet_contact (高度 + 速度阈值)        │
          ├── 6. 转换 root_rot: wxyz → xyzw                 │
          └── 7. 存储为单一 PKL                             │
               → data/Mimic/lafan.pkl                       │
```

### 3.2 CSV 加载

CSV 文件没有表头，直接读取数值：

```python
data = np.genfromtxt(csv_path, delimiter=",")  # (T, 36)
root_pos = data[:, 0:3]                        # (T, 3) — 世界坐标
qx, qy, qz, qw = data[:, 3:7].T               # CSV 中四元数为 (qx, qy, qz, qw)
root_rot_wxyz = [qw, qx, qy, qz]              # 转为 (w, x, y, z) 用于内部计算
dof_pos = data[:, 7:36]                        # (T, 29) — 29 个关节角度
```

> **四元数约定说明**：CSV 中存储为 `(qx, qy, qz, qw)`，内部计算统一使用 `(w, x, y, z)` 以简化 Hamilton 乘法，最终输出 PKL 时转为 `(x, y, z, w)` 以匹配 `LightAMPDataLoader` 和 `MotionLib` 的约定。

---

## 4. 公共派生特征的数学公式

### 4.1 关节速度 `dof_vel` — 中心差分

对关节角度序列 $\theta_t$ 使用**中心差分**估计速度：

$$
\dot{\theta}_t = \frac{\theta_{t+1} - \theta_{t-1}}{2 \Delta t}, \quad t \in [1, T-2]
$$

边界处使用前向/后向差分：

$$
\dot{\theta}_0 = \frac{\theta_1 - \theta_0}{\Delta t}, \qquad
\dot{\theta}_{T-1} = \frac{\theta_{T-1} - \theta_{T-2}}{\Delta t}
$$

其中 $\Delta t = 1 / \text{fps}$。

**选择中心差分的原因**：相比前向差分 $(\theta_{t+1} - \theta_t) / \Delta t$，中心差分是二阶精度（误差 $O(\Delta t^2)$ vs $O(\Delta t)$），且不会引入半帧的时间偏移。

### 4.2 基座角速度 `root_angle_vel` — 四元数微分

给定时间序列的单位四元数 $q_t = (w, x, y, z)$，body 坐标系下的角速度通过以下公式计算：

$$
\dot{q}_t = \text{central\_diff}(q_t, \Delta t)
$$

$$
\omega_t^{\text{quat}} = 2 \cdot q_t^{*} \otimes \dot{q}_t
$$

$$
\boldsymbol{\omega}_t = (\omega_x, \omega_y, \omega_z) = \text{Im}(\omega_t^{\text{quat}})
$$

其中 $q^*$ 是四元数共轭，$\otimes$ 是 Hamilton 乘法。

**推导**：四元数的时间导数满足 $\dot{q} = \frac{1}{2} q \otimes \boldsymbol{\omega}^{\text{quat}}$（其中 $\boldsymbol{\omega}^{\text{quat}} = (0, \omega_x, \omega_y, \omega_z)$）。左乘 $q^*$ 可解出 body 坐标系下的角速度：

$$
q^* \otimes \dot{q} = q^* \otimes \frac{1}{2} q \otimes \boldsymbol{\omega}^{\text{quat}} = \frac{1}{2} \boldsymbol{\omega}^{\text{quat}}
$$

因此 $\boldsymbol{\omega}^{\text{quat}} = 2 (q^* \otimes \dot{q})$，取虚部即为角速度 $(\omega_x, \omega_y, \omega_z)$。

### 4.3 基座线速度 `root_vel`（仅 DeepMimic）

世界坐标系下的位移差分：

$$
\mathbf{v}_t^{W} = \text{central\_diff}(\mathbf{p}_t, \Delta t)
$$

转换到 body 坐标系：

$$
\mathbf{v}_t^{B} = R(q_t)^{-1} \cdot \mathbf{v}_t^{W}
$$

其中 $R(q_t)$ 是从四元数 $q_t$ 提取的旋转矩阵。

**为什么转到 body 坐标系**：body frame 的线速度与机器人的朝向无关，是 **intrinsic** 的量。如果用世界坐标系的速度，相同的行走动作在不同朝向下会有完全不同的速度向量，这对跟踪和学习不利。

---

## 5. AMP 特有的特征

### 5.1 投影重力 `proj_grav`

将世界坐标系中的重力向量投影到 body 坐标系：

$$
\mathbf{g}^{B}_t = R(q_t)^{-1} \cdot \mathbf{g}^{W}
$$

其中 $\mathbf{g}^{W} = (0, 0, -1)$ 是归一化的重力方向。

**用途**：投影重力反映了机器人的身体倾斜程度。对 AMP 判别器来说，这是区分自然运动（身体大致直立）和不自然运动（身体严重倾斜）的重要信号。如果机器人摔倒或严重倾斜，$\mathbf{g}^B$ 会显著偏离 $(0, 0, -1)$。

### 5.2 相对关节位置 `dof_pos_rel`

$$
\theta_t^{\text{rel}} = \theta_t - \theta^{\text{default}}
$$

其中 $\theta^{\text{default}}$ 是 G1 的默认站立姿态关节角。

**用途**：相对于默认姿态的偏移更容易被判别器区分——自然运动通常在默认姿态附近波动，而不自然运动（如关节超限）会产生极大的偏移。

### 5.3 三次样条重采样 30fps → 50fps

给定源时间戳 $t_0, t_1, \ldots, t_{N-1}$（间隔 $1/30$ 秒）和对应采样值，构造**自然三次样条**（natural cubic spline），然后在目标时间戳（间隔 $1/50$ 秒）处求值。

自然三次样条在每个区间 $[t_i, t_{i+1}]$ 上是一个三次多项式，满足：
- 过所有数据点
- 一阶导数连续
- 二阶导数连续
- 边界条件：两端二阶导数为 0

**四元数的特殊处理**：

1. **半球连续性**：相邻帧的四元数可能在 $q$ 和 $-q$ 之间跳转（它们表示相同的旋转），这会导致样条插值出错。处理方法：

$$
\text{if } q_{t-1} \cdot q_t < 0, \quad \text{then } q_t \leftarrow -q_t
$$

2. **归一化**：样条插值后四元数可能不再是单位四元数，需要重新归一化：

$$
q \leftarrow \frac{q}{\|q\|}
$$

**为什么选择三次样条而不是线性插值**：关节运动通常是光滑的，三次样条可以保持二阶导数连续（加速度连续），而线性插值会在采样点处产生速度跳变，导致差分计算出的 `dof_vel` 不够平滑。

---

## 6. Pinocchio FK 计算 `robot_points`

### 6.1 什么是 `robot_points`

`robot_points` 是机器人关键身体部位在世界坐标系中的 3D 位置序列，形状为 $(T, R, 3)$，其中 $R$ 是选取的身体数量（本数据集中 $R = 14$）。选取的 14 个关键 body 为：

```
pelvis
left_hip_yaw_link, left_knee_link, left_ankle_roll_link
right_hip_yaw_link, right_knee_link, right_ankle_roll_link
torso_link
left_shoulder_yaw_link, left_elbow_link, left_wrist_yaw_link
right_shoulder_yaw_link, right_elbow_link, right_wrist_yaw_link
```

这些点覆盖了机器人的骨盆、躯干、四肢末端，可以完整描述机器人的空间姿态。

### 6.2 Forward Kinematics 原理

**Forward Kinematics (FK)** 的任务是：给定关节角度，计算每个关节/body 在空间中的位置和姿态。

对于一个有 $n$ 个关节的串联机械臂/人形机器人，从基座到第 $k$ 个 body 的变换是逐级传递的：

$$
T_k^{\text{world}} = T_{\text{root}}^{\text{world}} \cdot T_1(\theta_1) \cdot T_2(\theta_2) \cdots T_k(\theta_k)
$$

其中每个 $T_i(\theta_i)$ 是由第 $i$ 个关节角度决定的齐次变换矩阵。

### 6.3 Pinocchio 实现步骤

Pinocchio 是一个高效的刚体动力学库，内部使用 Featherstone 的 $O(n)$ 递归算法完成 FK。

**Step 1: 加载 URDF 模型**

```python
model = pin.buildModelFromUrdf("g1_29dof_rev_1_0.urdf")
data = model.createData()
```

URDF 文件定义了机器人的运动学树结构（关节层级、关节类型、link 尺寸等）。Pinocchio 解析后构建运动学模型。

**Step 2: 对每帧执行 FK**

```python
q = dof_pos[i]                                    # (29,) 关节角
pin.forwardKinematics(model, data, q)             # 正运动学：计算所有关节位姿
pin.updateFramePlacements(model, data)            # 更新所有 frame 的位姿
```

此时 `data.oMf[frame_id]` 包含了每个 body frame 相对于**模型基座**的 SE(3) 变换。

**Step 3: 变换到世界坐标系**

Pinocchio 的 FK 结果是相对于模型基座的局部坐标，需要用 root 的姿态将其变换到世界坐标系：

$$
\mathbf{p}_k^{\text{world}} = R_{\text{root}} \cdot \mathbf{p}_k^{\text{local}} + \mathbf{t}_{\text{root}}
$$

```python
w, x, y, z = root_rot_wxyz[i]
R_root = pin.Quaternion(w, x, y, z).toRotationMatrix()  # (3, 3)
t_root = root_pos[i]                                     # (3,)

for j, frame_id in enumerate(body_frame_ids):
    p_local = data.oMf[frame_id].translation             # (3,)
    p_world = R_root @ p_local + t_root                   # (3,)
    robot_points[i, j] = p_world
```

### 6.4 为什么需要 `robot_points`

DeepMimic 的跟踪奖励直接在笛卡尔空间中比较参考运动和策略执行的身体位置：

$$
r_{\text{track}} \propto \exp\left(-\alpha \sum_{k=1}^{R} \left\| \mathbf{p}_k^{\text{ref}} - \mathbf{p}_k^{\text{sim}} \right\|^2 \right)
$$

如果只用关节角度做跟踪，一个小的基座误差会被所有末端放大（杠杆效应），而在笛卡尔空间中比较可以更准确地反映姿态差异。

---

## 7. 脚部接触估计 `feet_contact`

### 7.1 问题背景

在仿真环境中，`feet_contact` 可以通过接触力传感器直接获取（力 > 阈值即为接触）。但我们处理的是**离线运动数据**，没有仿真器，需要从运动学信息中估计接触状态。

### 7.2 估计方法

使用**高度判据**和**速度判据**的组合：

$$
\text{contact}_t = \underbrace{(z_t^{\text{foot}} - z_{\min} < h_{\text{thresh}})}_{\text{高度判据}} \;\wedge\; \underbrace{(\|\dot{\mathbf{p}}_t^{\text{foot}}\| < v_{\text{thresh}})}_{\text{速度判据}}
$$

其中：
- $z_t^{\text{foot}}$ 是脚部 body（`left_ankle_roll_link` / `right_ankle_roll_link`）在第 $t$ 帧的世界坐标系 z 分量
- $z_{\min} = \min_t z_t^{\text{foot}}$ 是整个运动中脚部的最低高度（近似地面高度）
- $h_{\text{thresh}} = 0.05$ m 是高度阈值
- $\dot{\mathbf{p}}_t^{\text{foot}}$ 是脚部位置的 3D 速度（通过中心差分计算）
- $v_{\text{thresh}} = 0.3$ m/s 是速度阈值

### 7.3 为什么需要两个判据

**单独高度判据的问题**：脚在地面上滑行时高度低于阈值，但实际上并非稳定接触，此时判定为接触会误导训练。

**单独速度判据的问题**：脚在空中最高点附近速度接近零（抛物线顶点），此时判定为接触明显错误。

**组合判据**：只有当脚部同时满足「距地面很近」且「移动速度很慢」两个条件时，才判定为接触。这种组合可以有效排除上述两种误判情况。

### 7.4 阈值选择

| 参数 | 值 | 选择依据 |
|------|-----|---------|
| $h_{\text{thresh}}$ | 0.05 m | G1 脚部 link 厚度约 3-4 cm，留少许余量 |
| $v_{\text{thresh}}$ | 0.3 m/s | 正常站立时脚部速度 < 0.1 m/s，行走摆动相脚部速度 > 0.5 m/s |

---

## 8. 处理结果

### 8.1 AMP 输出 (`data/AMP/`)

| 文件名 | clips 数 | 帧数 | 时长 (s) |
|--------|---------|------|---------|
| `lafan_walk_clips.pkl` | 12 | 144,776 | 2,896 |
| `lafan_run_clips.pkl` | 6 | 75,576 | 1,512 |
| `lafan_dance_clips.pkl` | 8 | 76,142 | 1,523 |
| `lafan_fight_clips.pkl` | 5 | 61,248 | 1,225 |
| `lafan_getup_clips.pkl` | 6 | 46,734 | 935 |
| `lafan_jump_clips.pkl` | 3 | 36,669 | 733 |
| `lafan_all_clips.pkl` | 40 | 441,145 | 8,823 |

每个 PKL 的结构：

```python
{
    "clip_name_1": {
        "fps": 50,
        "dof_pos": np.ndarray (T, 29),        # float64
        "dof_vel": np.ndarray (T, 29),        # float32
        "dof_names": list[29],
        "root_pos": np.ndarray (T, 3),        # float64
        "root_rot": np.ndarray (T, 4),        # float64, (x, y, z, w)
        "root_angle_vel": np.ndarray (T, 3),  # float32
        "proj_grav": np.ndarray (T, 3),       # float32
        "dof_pos_rel": np.ndarray (T, 29),    # float32
    },
    "clip_name_2": { ... },
    ...
}
```

### 8.2 DeepMimic 输出 (`data/Mimic/`)

| 文件名 | motions 数 | 帧数 | 时长 (s) |
|--------|-----------|------|---------|
| `lafan.pkl` | 40 | 264,705 | 8,824 |

每个 motion 的结构：

```python
{
    "clip_name": {
        "fps": 30,
        "dof_pos": np.ndarray (T, 29),           # float64
        "dof_vel": np.ndarray (T, 29),           # float64
        "dof_names": list[29],
        "root_pos": np.ndarray (T, 3),           # float64
        "root_rot": np.ndarray (T, 4),           # float64, (x, y, z, w)
        "root_vel": np.ndarray (T, 3),           # float64, body frame
        "root_angle_vel": np.ndarray (T, 3),     # float64, body frame
        "robot_points": np.ndarray (T, 14, 3),   # float64, world frame
        "smpl_points": np.ndarray (T, 14, 3),    # float32, = robot_points
        "feet_contact": np.ndarray (T, 2),       # bool, [left, right]
        "hands_contact": np.ndarray (T, 2),      # bool, [left, right] = zeros
    },
    ...
}
```

> **注意**：`smpl_points` 在本数据集中等同于 `robot_points`，因为我们没有原始的 SMPL 人体模型数据。在由完整 retarget 管线生成的数据中，`smpl_points` 是人体模型的关键点坐标，`robot_points` 是对应的机器人关键点坐标。

---

## 9. 使用方法

### 9.1 重新生成数据

```bash
cd /path/to/unitree_lab

python scripts/process_lafan1_for_training.py \
    --input-dir  source/unitree_lab/unitree_lab/data/LAFAN1_Retargeting_Dataset/g1 \
    --urdf       source/unitree_lab/unitree_lab/data/LAFAN1_Retargeting_Dataset/robot_description/g1/g1_29dof_rev_1_0.urdf \
    --amp-output-dir   source/unitree_lab/unitree_lab/data/AMP \
    --mimic-output-dir source/unitree_lab/unitree_lab/data/Mimic \
    --amp-fps 50
```

### 9.2 在训练中使用

**AMP（Locomotion 任务）**：修改 `legged_env_cfg.py` 中的 `load_amp_data` 方法，将路径指向新的 PKL 文件：

```python
amp_conditions = {
    "walk": ["data/AMP/lafan_walk_clips.pkl"],
    "run":  ["data/AMP/lafan_run_clips.pkl"],
}
```

**DeepMimic（Motion Tracking 任务）**：修改 `deepmimic/config/legged_env_cfg.py` 中的 motion_file 路径：

```python
motion_files = ["data/Mimic/lafan.pkl"]
```

---

*文档版本: 1.0*
*生成时间: 2026-03-19*
*处理脚本: scripts/process_lafan1_for_training.py*
