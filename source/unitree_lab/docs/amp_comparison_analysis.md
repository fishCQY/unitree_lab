# AMP 算法实现对比分析：unitree_lab vs bfm_training

## 1. 概述

本文档对比分析 `unitree_lab` 和 `bfm_training` 两个代码库中 **AMP（Adversarial Motion Priors）** 算法的实现差异，并阐述各自设计选择的影响。

### 核心文件对照

| 功能 | unitree_lab | bfm_training |
|------|-------------|-------------|
| 判别器网络 | `rsl_rl/modules/amp.py` | `source/rsl_rl/rsl_rl/algorithms/amp.py` |
| 训练算法 | `rsl_rl/algorithms/ppo_amp.py` | `source/rsl_rl/rsl_rl/algorithms/amp.py` |
| 训练循环 | `rsl_rl/runners/amp_runner.py` | `source/rsl_rl/rsl_rl/runners/on_policy_runner.py` |
| 数据加载 | `unitree_lab/utils/amp_data_loader.py` | `light_gym/utils/light_amp_data_loader.py` |
| 观测定义 | `locomotion/mdp/observations.py` | `locomotion/config/base_env_cfg.py` |
| 配置 | `config/agents/rsl_rl_ppo_cfg.py` | `config/agents/rsl_rl_ppo_cfg.py` |

---

## 2. 架构设计差异

### 2.1 集成模式

**unitree_lab — PPO 子类模式：**
- `PPOAMP` 继承 `PPO`，将 AMP 判别器训练嵌入到 PPO 的 `update()` 循环内
- 判别器观测存储在独立的 `CircularBuffer` 中
- 与 PPO 共享 mini-batch 迭代

**bfm_training — Plugin 模式：**
- `AMP` 是独立类，不继承任何 RL 算法
- Runner 在 rollout 时调用 `amp.reward()` 计算风格奖励
- PPO 更新后调用 `amp.update()` 独立训练判别器
- AMP 有自己的 epoch/mini-batch 配置

**影响：**
- Plugin 模式更解耦，AMP 可以附加到任何 RL 算法上
- Plugin 模式的判别器训练频率可以独立于策略训练调节

### 2.2 数据流

**unitree_lab：**
```
环境 → AMPAgentObsTerm (3D tensor, 2帧序列) → CircularBuffer → 判别器
环境 → AMPDemoObsTerm (每步随机采样demo) → CircularBuffer → 判别器
```

**bfm_training：**
```
环境 → amp obs group (单步2D tensor) → RolloutStorage → AMP Plugin 构建序列 → 判别器
离线数据集 → AMP Plugin 构建序列 → 判别器
```

---

## 3. 奖励融合方式

### 3.1 unitree_lab — 线性插值 (lerp)

```python
r_total = task_style_lerp * r_task + (1 - task_style_lerp) * r_style
```

默认 `task_style_lerp=0.5`，任务与风格各占一半。

### 3.2 bfm_training — 直接相加

```python
r_total = r_task + reward_weight * r_style * step_dt
```

默认 `reward_weight=2.0`。

### 3.3 影响

| 方面 | lerp 插值 | 直接相加 |
|------|----------|---------|
| 奖励尺度 | 固定预算内竞争 | 独立叠加 |
| 稳定性 | 更稳定 | 需调参 |
| 灵活性 | 较低 | 更灵活 |

---

## 4. 条件 AMP (Conditional AMP)

**unitree_lab：** 不支持。

**bfm_training：** 完整支持。使用 `nn.Embedding` 将条件 ID 映射为向量，与 AMP 观测拼接后送入判别器。训练时按条件 ID 匹配 policy 与 offline 样本。

**影响：** 条件 AMP 允许单个判别器处理多种步态（walk/run/sprint），策略可以根据速度命令切换风格。

---

## 5. 观测空间

### 5.1 unitree_lab（64 维/帧）
```
[root_angle_vel(3), proj_grav(3), dof_pos_rel(29), dof_vel(29)]
```

### 5.2 bfm_training（64+12 维/帧）
```
[joint_pos(N), joint_vel(N), base_ang_vel(3), projected_gravity(3), body_pos_b(K×3)]
```

### 5.3 关键点 (body_pos_b) 的作用

bfm_training 选择的关键点：左右膝盖 + 左右肘部（4 个点 × 3D = 12 维）。

**关键点的计算方式：** 在 body frame 下的相对位置
```python
relative_body_pos = body_pos_w - root_pos_w
body_pos_b = quat_apply_inverse(root_quat_w, relative_body_pos)
```

**选择原因：**
1. 处于运动链中段，能有效反映整条肢体链的姿态
2. 对步态视觉观感影响最大（膝盖弯曲、手臂摆动）
3. 提供任务空间（笛卡尔空间）信息，弥补关节空间的非线性盲区

---

## 6. 损失计算差异

### 6.1 判别器损失

**unitree_lab（支持 GAN / LSGAN / WGAN）：**
```python
# LSGAN
policy_loss = MSELoss(disc_score, -1)
demo_loss = MSELoss(disc_demo_score, +1)
disc_loss = 0.5 * (policy_loss + demo_loss)
disc_total_loss = disc_loss + grad_penalty_scale * GP
```

**bfm_training（仅 LSGAN）：**
```python
offline_loss = (offline_d - 1).pow(2).mean()
policy_loss = (policy_d + 1).pow(2).mean()
amp_loss = offline_loss + policy_loss + 0.5 * amp_lambda * GP
```

### 6.2 梯度惩罚相对权重

| 组成 | unitree_lab | bfm_training |
|------|-------------|-------------|
| 判别损失 | `0.5 × (L_policy + L_demo)` | `L_policy + L_demo` |
| 梯度惩罚 | `10.0 × GP` | `5.0 × GP` |
| **GP / 判别损失比值** | **20** | **5** |

unitree_lab 的 GP 相对权重是 bfm_training 的 4 倍，判别器更平滑但区分力更弱。

---

## 7. 数据处理差异

### 7.1 镜像增强

**unitree_lab：** 数据加载器支持但未在默认配置中启用。

**bfm_training：** 默认启用，数据加载时自动做左右镜像。训练数据翻倍，减少不对称步态。

### 7.2 数据采样

**unitree_lab：** CircularBuffer 存储历史观测，可能包含 off-policy 数据。

**bfm_training：** 从 RolloutStorage 在线提取有效序列（`_extract_valid_sequences`），排除跨 episode 边界的序列。

### 7.3 训练噪声

**unitree_lab：** 无。

**bfm_training：** 可选 `noise_scale`，训练时给观测加均匀噪声，起正则化作用。

---

## 8. 分层权重衰减 (Per-Layer Weight Decay)

### 8.1 什么是权重衰减

权重衰减（Weight Decay）是 L2 正则化的等价形式。每次参数更新时，额外把权重往零方向"拉"：

```
θ_new = θ_old - lr × (gradient + weight_decay × θ_old)
```

权重越大，被拉回零的力越大。这防止了权重"爆炸"，起到正则化作用。

### 8.2 unitree_lab — 分层权重衰减

将判别器拆成两个参数组，分别设置不同的衰减强度：

```python
params = [
    {"name": "disc_trunk",  "params": disc_trunk.parameters(),  "weight_decay": 1e-4},
    {"name": "disc_linear", "params": disc_linear.parameters(), "weight_decay": 1e-2},
]
optimizer = Adam(params, lr=5e-4)
```

| 参数组 | 包含层 | weight_decay |
|--------|--------|-------------|
| `disc_trunk` | 所有隐藏层 (Linear+ReLU × N) | **1e-4**（弱约束） |
| `disc_linear` | 最终输出层 (Linear → 1) | **1e-2**（强约束，是 trunk 的 100 倍） |

### 8.3 bfm_training — 全局统一衰减

```python
params = list(discriminator.parameters())
if condition_embedding is not None:
    params += list(condition_embedding.parameters())
optimizer = Adam(params, weight_decay=1e-5, lr=1e-4)
```

所有参数（判别器全部层 + 条件嵌入）共享 `weight_decay=1e-5`。

### 8.4 为什么 unitree_lab 对输出层施加 100 倍的强衰减

判别器的结构是：`输入 → [Linear→ReLU]×N (trunk) → Linear(hidden→1) (output)`

**（1）输出层权重控制判别器的"绝对信心"**

隐藏层学习特征表达（把运动映射到高维空间），输出层学习决策边界（在特征空间中区分"真/假"）。如果输出层权重过大，判别器输出值会非常极端（如 +50 或 -50），表现为"过度自信"。

**（2）防止奖励信号饱和**

以 LSGAN 的奖励公式为例：`reward = clamp(1 - 0.25 × (D(x) - 1)², min=0)`

- D(x) = 1 时 reward = 1（最大值）
- D(x) = 3 或 D(x) = -1 时 reward = 0（已饱和）

如果输出层权重太大导致 D(x) 远超有效范围，reward 几乎总是 0，策略得不到有效梯度。强衰减把输出层权重压小，让 D(x) 留在奖励函数的有效区间内。

**（3）隐藏层为何用弱衰减**

隐藏层需要足够大的权重来编码复杂的非线性特征（区分自然与不自然步态的微妙差异）。1e-4 的弱衰减只做轻微约束，保持特征表达能力。

**类比：** 判别器是考官——隐藏层是"知识水平"（需要丰富），输出层是"评分尺度"（需要克制）。

### 8.5 bfm_training 不用分层的原因

bfm_training 通过其他机制达到类似效果：
- 更低的学习率（1e-4 vs 5e-4），输出层权重增长更慢
- 可选训练噪声（noise_scale），模糊判别边界
- 更小的实际梯度惩罚权重（5.0 vs 10.0）

这些叠加在一起也能控制判别器输出范围，只是调参策略不同。

### 8.6 影响对比

| 方面 | 分层衰减 (unitree_lab) | 全局衰减 (bfm_training) |
|------|------------------------|--------------------------|
| 输出值范围 | 较小、温和 | 较大、可能极端 |
| 奖励信号 | 更平滑、不易饱和 | 可能更尖锐 |
| 特征学习 | trunk 自由度高、表达力强 | 整体差异小 |
| 训练稳定性 | 更稳定 | 需其他手段补偿 |

---

## 9. 超参数对比

| 参数 | unitree_lab | bfm_training |
|------|-------------|-------------|
| 判别器隐藏层 | [1024, 512] | [1024, 512] |
| 判别器学习率 | 5e-4（固定） | 1e-4（可随策略 lr 缩放） |
| 权重衰减 | trunk 1e-4, linear 1e-2（分层） | 全局 1e-5 |
| 梯度裁剪 | 0.5 | 1.0 |
| 判别器训练次数 | 与 PPO 共享 | 独立 2 epoch × 2 mini-batch |
| 梯度惩罚系数 | 10.0 | 10.0 (实际 ×0.5 = 5.0) |
| 奖励缩放 | 1.0 × dt | 2.0 × dt |
| 奖励融合 | lerp (0.5) | 相加 |
| 条件嵌入维度 | N/A | 16 |
| 训练噪声 | 无 | 可选 |
| 多 GPU | 基础支持 | 完整支持 |

---

## 10. 总结与重构建议

### 各自优势

**unitree_lab 优势：**
- 多损失类型支持（GAN / LSGAN / WGAN）
- Lerp 奖励融合更稳定
- 分层权重衰减，训练更稳定

**bfm_training 优势：**
- Plugin 架构，解耦清晰
- 条件 AMP 支持
- 关键点（body_pos_b）观测
- 镜像数据增强
- 从 RolloutStorage 直接构建序列（向量化）
- 训练噪声正则化
- 完整多 GPU 支持
- LR 缩放同步策略

### 重构方向

将两者优势结合：
1. **采用 Plugin 架构**（来自 bfm_training）
2. **保留多损失类型和 lerp 奖励融合**（来自 unitree_lab）
3. **加入条件 AMP、关键点、镜像增强、训练噪声**（来自 bfm_training）
4. **改用 RolloutStorage 序列构建**（来自 bfm_training）
5. **保留分层权重衰减**（来自 unitree_lab）

---

## 11. 重构后的 AMPPlugin 功能清单

重构后的 `rsl_rl/plugins/amp.py` (AMPPlugin) 整合了两者优势，以下是功能对照：

### 11.1 功能矩阵

| 功能 | 来源 | AMPPlugin 中的实现 | 对应代码位置 |
|------|------|-------------------|-------------|
| **多损失类型** | unitree_lab | `LossType` 枚举，支持 GAN/LSGAN/WGAN | `_compute_disc_loss()`, `_score_to_reward()` |
| **Lerp 奖励融合** | unitree_lab | `lerp_reward()` | `task_style_lerp * task + (1-lerp) * style` |
| **分层权重衰减** | unitree_lab | trunk 和 linear 分开配 weight_decay | optimizer 的 `param_groups` |
| **Plugin 架构** | bfm_training | 独立类，不继承 PPO | `AMPPlugin` + `AMPPluginRunner` |
| **条件 AMP** | bfm_training | `nn.Embedding` + 条件匹配采样 | `condition_embedding`, `_match_conditions()` |
| **可选训练噪声** | bfm_training | `noise_scale` 参数控制 | `update()` 中的噪声注入 |
| **RolloutStorage 序列构建** | bfm_training | 向量化提取有效序列 | `_extract_sequences_from_rollout()` |
| **多 GPU 支持** | bfm_training | broadcast + all-reduce | `broadcast_parameters()`, `_reduce_gradients()` |
| **LR 缩放** | bfm_training | `lr_scale` 参数 | `update()` 中的学习率调整 |
| **关键点观测** | bfm_training | `amp_body_pos_b()` 观测函数 | `observations.py` |
| **条件观测** | bfm_training | `vel_cmd_condition_id()` | `observations.py` |
| **镜像增强** | bfm_training | 数据加载器支持 | `amp_data_loader.py` |
| **WGAN 输出归一化** | unitree_lab | `EmpiricalNormalization` | `output_normalizer` |
| **梯度惩罚** | 共有 | `(||∇D|| - 0)^2` | `_grad_penalty()` |
| **完整 state_dict** | 共有 | 含判别器+归一化+优化器+嵌入 | `state_dict()`, `load_state_dict()` |

### 11.2 可选训练噪声的实现

训练噪声通过 `noise_scale` 参数控制，在 `update()` 方法中注入：

```python
# 在 update() 的 mini-batch 循环中
if self.noise_scale is not None:
    p_batch = p_batch + (2 * torch.rand_like(p_batch) - 1) * self.noise_scale
    o_batch = o_batch + (2 * torch.rand_like(o_batch) - 1) * self.noise_scale
```

- 默认 `noise_scale=None`（关闭）
- 设置时给 policy 和 offline 观测都加均匀噪声 `U(-noise_scale, +noise_scale)`
- 在归一化之前注入（先加噪声，再归一化，再送入判别器）
- 作用：正则化判别器，防止过拟合，提高泛化能力

### 11.3 可能的运行问题及注意事项

**（1）离线数据必须提供**

AMPPlugin 需要离线参考运动数据。可通过三种方式提供：
- 环境配置实现 `env.cfg.load_amp_data()` 方法
- 配置中设置 `data_loader_func`（callable）
- 配置中设置 `motion_files`（PKL 文件路径列表）

如果三种都未配置，会打印警告但不会崩溃——只是判别器没有参考数据，`_offline_sequences` 为 None，`update()` 会出错。

**（2）环境需要输出 AMP 观测组**

环境配置必须包含与 `amp_cfg.obs_group`（默认 `"amp"`）对应的观测组。如果使用条件 AMP，还需要 `amp_cfg.condition_obs_group`（默认 `"amp_condition"`）。

**（3）obs_groups 配置**

Runner 配置的 `obs_groups` 必须包含 AMP 相关组，以便 RolloutStorage 存储这些观测：
```python
obs_groups = {
    "policy": ["policy"],
    "critic": ["critic"],
    "amp": ["amp"],                      # 必须
    "amp_condition": ["amp_condition"],   # 条件 AMP 时必须
}
```
AMPPluginRunner 的 `_get_default_obs_sets()` 已自动将 AMP 组加入默认集合。

**（4）G1 body_names 需根据实际 URDF 调整**

环境配置中 `amp_body_pos_b` 的 `body_names` 需要与机器人 URDF 中的 link 名称一致。当前配置的 `left_knee_link` / `right_knee_link` / `left_shoulder_roll_link` / `right_shoulder_roll_link` 需要验证是否与 G1 的实际 URDF 匹配。

**（5）AMP 观测特征与离线数据特征必须一致**

在线观测 `[joint_pos, joint_vel, base_ang_vel, projected_gravity, body_pos_b]` 的维度和语义必须与离线 PKL 中提取的特征完全对齐。如果离线数据缺少 `key_points_b`，在线观测不应包含 `body_pos_b`。
