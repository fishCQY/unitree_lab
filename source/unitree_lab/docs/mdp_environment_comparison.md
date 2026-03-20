# MDP 环境对比分析：unitree_lab vs bfm_training

本文档详细对比两个代码库中 **AMP Locomotion** 和 **Mimic (DeepMimic)** 两类任务的 MDP 环境设计。

---

## 第一部分：AMP Locomotion 环境

### 1. 奖励函数

#### 1.1 bfm_training 的奖励（极简 7 项）

| 奖励项 | 权重 | 公式/说明 |
|--------|------|-----------|
| `tracking_lin_vel` | **2.5** | `exp(-‖v_cmd - v‖² / 0.15)`，yaw 系，σ²=0.15 |
| `tracking_ang_vel` | **1.5** | `exp(-(ω_cmd - ω)² / 0.5)`，σ²=0.5 |
| `orientation` | **1.0** | `exp(-2‖grav_xy‖²) + 0.1·exp(-‖grav_xy‖)`，正奖励 |
| `action_rate` | **-0.1** | `action_rate_l2_clipped`，max_val=5.0 |
| `dof_pos_limits` | **-1.0** | 关节接近限位惩罚，threshold=0.1 |
| `undesired_contacts` | **-2.0** | 非足端接触，threshold=1.0 |
| `illegal_footstep` | **0.0** | 默认关闭 |

#### 1.2 unitree_lab 的奖励（20 项，AMP 下部分关闭）

| 奖励项 | 原始权重 | AMP 下权重 | bfm 对标 |
|--------|----------|-----------|----------|
| `track_lin_vel_xy_exp` | 3.0 | 1.0 | tracking_lin_vel (2.5) |
| `track_ang_vel_z_exp` | 1.5 | 1.0 | tracking_ang_vel (1.5) |
| `body_orientation_l2` | -2.0 | -1.0 | orientation (1.0) |
| `action_rate_l2` | -0.01 | -0.01 | action_rate (-0.1) |
| `dof_pos_limits` | -2.0 | -2.0 | dof_pos_limits (-1.0) |
| `undesired_contacts` | -1.0 | -1.0 | undesired_contacts (-2.0) |
| `lin_vel_z_l2` | -0.25 | -1.0 | 无（bfm 不用） |
| `ang_vel_xy_l2` | -0.05 | -0.05 | 无 |
| `energy` | -1e-3 | -1e-3 | 无 |
| `dof_acc_l2` | -0.5e-7 | -0.5e-7 | 无 |
| `termination_penalty` | -200.0 | -200.0 | 无 |
| `feet_air_time` | 0.15 | 0.4 | 无 |
| `feet_slide` | -0.25 | -0.25 | 无 |
| `feet_too_near` | -2.0 | -2.0 | 无 |
| `feet_stumble` | -2.0 | -2.0 | 无 |
| `feet_force` | -3e-3 | 0.0 | 无 |
| `fly` | -1.0 | 0.0 | 无 |
| `flat_orientation_l2` | -1.0 | 0.0 | 无 |
| `joint_deviation_hip` | -0.15 | 0.0 | 无 |
| `joint_deviation_arms` | -0.2 | 0.0 | 无 |
| `joint_deviation_legs` | -0.02 | 0.0 | 无 |

### 2. 终止条件

| 条件 | unitree_lab | bfm_training |
|------|-------------|-------------|
| 超时 | 20s | **10s** |
| 基座接触 | torso 接触 > 1N | 有，带 **10% 概率免疫** |
| 地形出界 | 无 | 有 |
| 姿态异常 | 无 | 有 (>45°, prob=1%) |

### 3. 域随机化

| 项目 | unitree_lab | bfm_training |
|------|-------------|-------------|
| 摩擦 | static [0.6,1.0], dynamic [0.4,0.8] | static **[0.2,1.3]**, dynamic **[0.2,1.3]** |
| 基座质量 | 加性 (-5,5) kg | 乘性 **(1.0,1.2)** |
| 全身质量 | 乘性 [0.8,1.2] | 乘性 **(0.85,1.15)** + 重算惯量 |
| 质心偏移 | [-0.05, 0.05] m | **[-0.025, 0.025] m** |
| 外力推动 | 每步 0.2% 概率, ±1000N | 每 **3.7-4.2s**, **速度扰动 ±0.5 m/s** |
| 库仑/粘滞摩擦 | 无 | 有 |
| 默认关节偏移 | 无 | 有 (±0.05 rad) |

### 4. 速度指令

| 参数 | unitree_lab | bfm_training |
|------|-------------|-------------|
| vx | (-1.0, 1.0) | (-1.0, 1.0) |
| vy | (-0.5, 0.5) | (-0.5, 0.5) |
| ωz | **(-1.0, 1.0)** | **(-2.0, 2.0)** |
| 重采样 | **固定 10s** | **随机 4-6s** |
| 静止比例 | 5% | **10%** |

### 5. 设计哲学

- **unitree_lab**：手动精细雕刻 20 项奖励做兜底，AMP 只接管风格约束
- **bfm_training**：极简 7 项核心目标，大量行为约束交给 AMP 风格奖励

---

## 第二部分：Mimic (DeepMimic) 环境

### 1. 奖励函数

#### 1.1 bfm_training DeepMimic 奖励

| 奖励项 | 权重 | 参数 |
|--------|------|------|
| `tracking_root_height` | **3.0** | σ²=1/12 |
| `tracking_root_quat` | **3.0** | σ²=0.16 |
| `tracking_root_orientation` | **3.0** | σ²=1/12 |
| `tracking_root_velocity` | **4.0** | σ²=1/4 |
| `tracking_root_angle_velocity` | **1.0** | σ²=1/12 |
| `tracking_upper_joint_pos` | **1.0** | σ²=1/4 |
| `tracking_lower_joint_pos` | **1.0** | σ²=1/4 |
| `tracking_keypoints` | **4.0** | σ²=1/8 |
| `tracking_feet_keypoints` | **3.0** | σ²=1/8 |
| `tracking_feet_contact_number` | **1.0** | - |
| `action_rate_l2` | **-0.3** | - |
| `joint_limit` | **-1.0** | - |
| `termination_penalty` | **-200.0** | - |

#### 1.2 unitree_lab MotionTracking 奖励

| 奖励项 | 权重 | bfm 对标 |
|--------|------|----------|
| `tracking_joint_pos` | 1.0 | tracking_upper/lower_joint_pos (1.0 each) |
| `tracking_joint_vel` | 0.5 | 无（bfm 不用） |
| `tracking_body_lin_vel` | 0.5 | tracking_root_velocity (4.0) |
| `tracking_body_ang_vel` | 0.5 | tracking_root_angle_velocity (1.0) |
| `tracking_key_points_w_exp` | 1.0 | tracking_keypoints (4.0) |
| `tracking_key_points_exp` | 1.0 | tracking_feet_keypoints (3.0) |
| `feet_height_l2` | -10.0 | 无（bfm 不用） |
| `torso_contacts` | -1.0 | 无 |
| `energy` | -1e-3 | 无 |
| `dof_acc_l2` | -0.5e-7 | 无 |
| `action_rate_l2` | -0.01 | action_rate_l2 (-0.3) |
| `termination_penalty` | -200.0 | termination_penalty (-200.0) |
| `dof_pos_limits` | -2.0 | joint_limit (-1.0) |
| `feet_stumble` | -2.0 | 无 |
| `feet_slide` | -0.25 | 无 |

### 2. 关键差异

| 维度 | unitree_lab | bfm_training |
|------|-------------|-------------|
| **根状态跟踪** | 无独立项（通过 body_vel 间接） | **拆成 5 项**（高度/姿态/方向/速度/角速度） |
| **接触一致性** | 无 | **有**（feet_contact_number） |
| **上下肢分开** | 无 | **有**（upper/lower 独立权重） |
| **动作平滑** | w=-0.01 | **w=-0.3**（30 倍） |
| **Episode 模式** | 循环播放 (20s) | **到末尾终止 (10s)** |
| **yaw 增强** | 无 | **有** |
| **Critic 跟踪误差** | 无 | **有**（直接输入各项误差） |

---

## 第三部分：修改记录

为使 unitree_lab 的 AMP 和 Mimic 环境与 bfm_training 一致，做了以下修改：

### AMP Locomotion 环境 (`rough_env_cfg.py`)

**奖励权重修改：**
- `track_lin_vel_xy_exp`: 1.0 → **2.5**，σ: √0.25 → **√0.15**
- `track_ang_vel_z_exp`: 1.0 → **1.5**，σ: √0.25 → **√0.5**
- `body_orientation_l2`: -1.0 → **0.0**（关闭，由 orientation 正奖励替代）
- `action_rate_l2`: -0.01 → **-0.1**
- `dof_pos_limits`: -2.0 → **-1.0**
- `undesired_contacts`: -1.0 → **-2.0**
- 以下项设为 0（保留但不使用）：`lin_vel_z_l2`, `ang_vel_xy_l2`, `energy`, `dof_acc_l2`, `feet_air_time`, `feet_slide`, `feet_too_near`, `feet_stumble`, `termination_penalty`

**终止条件修改：**
- `episode_length_s`: 20.0 → **10.0**

**速度指令修改：**
- `ang_vel_z`: (-1.0, 1.0) → **(-2.0, 2.0)**
- `resampling_time_range`: (10.0, 10.0) → **(4.0, 6.0)**
- `rel_standing_envs`: 0.05 → **0.1**

**域随机化修改：**
- 摩擦范围：对齐到 bfm_training 的 (0.2, 1.3)
- 质心偏移：缩小到 (-0.025, 0.025)
- 外力推动：改为速度扰动方式

### Mimic 环境 (`tracking_env_cfg.py`)

**奖励权重修改：**
- `tracking_key_points_w_exp`: 1.0 → **4.0**
- `tracking_key_points_exp`: 1.0 → **3.0** (对标 feet_keypoints)
- `tracking_body_lin_vel`: 0.5 → **4.0**
- `tracking_body_ang_vel`: 0.5 → **1.0**
- `action_rate_l2`: -0.01 → **-0.3**
- `dof_pos_limits`: -2.0 → **-1.0**
- 以下项设为 0（保留但不使用）：`tracking_joint_vel`, `energy`, `dof_acc_l2`, `feet_stumble`, `feet_slide`, `feet_height_l2`, `torso_contacts`

**Episode 长度修改：**
- `episode_length_s`: 20.0 → **10.0**
