# 实验记录：AMP Plugin 首次训练诊断与修复

**日期：** 2026-03-22
**任务：** G1 AMP Locomotion (Rough Terrain + Conditional LAFAN)
**配置：** `UnitreeG1RoughPluginRunnerCfg` → `AMPPluginRunner`
**运行 ID：** wandb `run-20260321_031044-der9fl3w`

---

## 1. 实验背景

首次使用 Plugin 架构的 AMP 训练 G1 rough terrain locomotion。之前的训练使用 `OnPolicyRunner`（无 AMP），同样硬件和类似配置下 10000 轮约 7-8 小时，每迭代约 2.5-2.9 秒。

### 硬件环境

- GPU: NVIDIA RTX 3060 12GB
- 训练命令: `bash train.sh`（task: `unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-v0`）

### 训练配置

| 参数 | 值 |
|------|------|
| num_envs | 4096 |
| num_steps_per_env | 24 |
| max_iterations | 200000 |
| actor/critic | MLP [1024, 512, 256], ELU |
| PPO epochs/mini-batches | 5 / 4 |
| AMP disc hidden | [1024, 512] |
| AMP disc epochs/mini-batches | 5 / 4 |
| loss_type | LSGAN |
| style_reward_scale | 1.0 |
| task_style_lerp | 0.5 |
| num_conditions | 2 (walk/run) |
| sim_dt | 0.005, decimation=4 (policy_dt=0.02) |

### AMP 离线数据

| 类别 | PKL 文件 | Clips | 帧数 | 时长 |
|------|---------|-------|------|------|
| Walk | `lafan_walk_clips.pkl` | 12 | 144,776 | 48 min |
| Run | `lafan_run_clips.pkl` | 6 (含 sprint) | 75,576 | 25 min |
| 合计 | | 18 | 220,352 | 73 min |

镜像增强后：440,704 帧 → 440,668 有效序列。

---

## 2. 发现的问题

### 2.1 问题一：训练速度极慢（12 倍）

**现象：** 每迭代 ~34.6 秒，而之前无 AMP 训练每迭代仅 ~2.5-2.9 秒。

**时间分布：**

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Collection time (仿真采样) | ~5.1s | 15% |
| Learning time (PPO + AMP) | ~29.5s | **85%** |

**根因定位：** `rsl_rl/plugins/amp.py` 中的 `_match_conditions()` 方法。

```python
# 原始代码：纯 Python for 循环，逐样本执行 CUDA 操作
def _match_conditions(self, target_conds, offline_conds, batch_size):
    indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
    for i in range(batch_size):          # batch_size ≈ 23,552
        cond = target_conds[i]
        pool = torch.nonzero(offline_conds == cond, as_tuple=False).squeeze(-1)
        if pool.numel() == 0:
            pool = torch.arange(len(offline_conds), device=self.device)
        indices[i] = pool[torch.randint(pool.numel(), (1,), device=self.device)]
    return indices
```

**影响分析：**
- `mini_batch_size ≈ 23,552`（4096 × 23 / 4）
- 每迭代调用 20 次（5 epochs × 4 mini-batches）
- 总计 ~471,040 次 Python→CUDA 往返
- 按 ~60μs/次估算：471,040 × 60μs ≈ **28.3 秒**（与 29.5s 实测吻合）

### 2.2 问题二：训练效果差——速度跟踪 error 持续增大

**指标趋势（2200 次迭代后）：**

| 指标 | iter 0 | iter 100 | iter 500 | iter 1000 | iter 2000 | iter 2200 |
|------|--------|----------|----------|-----------|-----------|-----------|
| Mean Reward | -2.53 | -0.12 | 1.19 | 5.44 | 12.49 | 14.24 |
| Episode Length | 19.7 | 2.6 | 63.8 | 238.8 | 441.1 | 468.8 |
| error_vel_xy | 0.027 | 0.005 | 0.067 | 0.250 | 0.389 | 0.389 |
| error_vel_yaw | 0.108 | 0.013 | 0.177 | 0.620 | 0.980 | 0.980 |
| Terrain Level | 0 | 0 | 0 | 0 | 0 | 0 |
| time_out 比例 | 2% | 0% | 0% | 0.1% | 85% | 85% |

**根因：`task_style_lerp=0.5` 导致 style reward 压制 task reward**

奖励融合方式为：`r_total = 0.5 × r_task + 0.5 × r_style`

量化分析（iter 2200）：

| 分量 | 每步值 |
|------|--------|
| task reward per step | ~0.003 |
| style reward per step | ~0.007（step_dt × scale × LSGAN_score） |
| **style / task 比值** | **2.3×** |

在 0.5/0.5 的 lerp 下，PPO 看到的总 reward 中 style 信号占大头。结果：
- 策略学会了**站稳不摔倒、模仿人形姿态**（满足 AMP）
- 但**不积极追踪速度指令**（task reward 被稀释）
- error_vel_xy "增大" 的本质：早期 episode 极短（2-20步，还没偏离就摔倒了），error 看似小；后期 episode 468 步，有足够时间暴露跟踪偏差

### 2.3 问题三：地形课程完全卡住（terrain level = 0）

**晋升条件分析：**

```python
# terrain_levels_vel 中的晋升条件
move_up = (distance > terrain_size/2)
        & (lin_track_reward_sum / max_ep_s > weight * 0.7)   # 需 > 1.75
        & (ang_track_reward_sum / max_ep_s > weight * 0.7)   # 需 > 1.05
```

当前 track reward 远达不到阈值（无论如何解释 `_episode_sums`，归一化后的值都远低于 1.75）。

同时，降级条件（< weight × 0.6）频繁触发：`avg_terrain_level_down_count = 1752`。

### 2.4 AMP 判别器状态（正常）

| 指标 | iter 0 | iter 2200 | 评估 |
|------|--------|-----------|------|
| disc_score (policy) | -0.79 | -0.62 | 策略在逐渐欺骗判别器 ✓ |
| disc_demo_score (offline) | 0.46 | 0.62 | 两者分数在收敛 ✓ |
| disc_loss | 0.275 | 0.164 | 稳定下降 ✓ |

判别器本身工作正常，策略确实在学习模仿参考运动。

---

## 3. 修复方案

### 3.1 修复训练速度：向量化 `_match_conditions`

**文件：** `rsl_rl/plugins/amp.py`

**方案：** 在 `set_offline_data()` 时预建每个 condition 到离线序列索引的映射（`dict[int, Tensor]`），在 `_match_conditions` 中按 condition 值分组向量化采样。

```python
# 新增：预建索引映射
def _rebuild_cond_index_map(self, offline_conds):
    self._cond_index_map = {}
    for cond_val in offline_conds.unique().tolist():
        self._cond_index_map[cond_val] = torch.nonzero(
            offline_conds == cond_val, as_tuple=False
        ).squeeze(-1)

# 向量化实现：从 ~47 万次循环变为 2 次（2 个 condition）
def _match_conditions(self, target_conds, offline_conds, batch_size):
    indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
    matched = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
    for cond_val, pool in self._cond_index_map.items():
        mask = target_conds == cond_val
        count = mask.sum().item()
        if count == 0:
            continue
        indices[mask] = pool[torch.randint(pool.numel(), (count,), device=self.device)]
        matched |= mask
    if not matched.all():
        fallback = torch.arange(len(offline_conds), device=self.device)
        n = (~matched).sum().item()
        indices[~matched] = fallback[torch.randint(fallback.numel(), (n,), device=self.device)]
    return indices
```

**预期效果：** Learning time 从 ~29.5s 降到 < 1s，每迭代从 ~34.6s 降到 ~6-7s。

### 3.2 修复奖励融合：lerp → 直接相加（bfm_training 风格）

**文件：** `rsl_rl/plugins/amp.py`, `rsl_rl/runners/amp_plugin_runner.py`, `rsl_rl_ppo_cfg.py`

**改动：**
- 去掉 `task_style_lerp` 参数
- `lerp_reward()` → `combine_reward()`：`r_total = r_task + r_style`
- `style_reward_scale`: 1.0 → 2.0（与 bfm_training 一致）
- style_reward 已内含 `step_dt × scale`，直接叠加到 task reward 上

**对比：**

| 方式 | 公式 | task 信号保留 |
|------|------|-------------|
| 之前 (lerp) | `0.5 × task + 0.5 × style` | 50% |
| 现在 (additive) | `task + style` | **100%** |

### 3.3 修复课程学习：基于距离（bfm_training 风格）

**文件：** `source/unitree_lab/unitree_lab/tasks/locomotion/mdp/curriculums.py`

**改动：** 去掉对 `reward_weight` 阈值的依赖，改为纯距离驱动。

```python
# 之前：依赖 reward weight 阈值（无法达到）
move_up = (distance > size/2) & (lin_track > weight*0.7) & (ang_track > weight*0.7)
move_down = (lin_track < weight*0.6) | (ang_track < weight*0.6)

# 现在：纯距离驱动（bfm_training 风格）
move_up = distance > terrain_size / 2     # 走过半个地形 → 晋升
move_down = distance < terrain_size / 4   # 走不到 1/4 → 降级
```

`command_levels_vel`（速度课程）也做了同样改动。

### 3.4 配置调整

| 参数 | 之前 | 之后 | 原因 |
|------|------|------|------|
| `max_iterations` | 200000 | **15000** | 200000 需要 ~80 天 |
| `style_reward_scale` | 1.0 | **2.0** | 对齐 bfm_training |
| `task_style_lerp` | 0.5 | **删除** | 改为 additive |

---

## 4. 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `rsl_rl/plugins/amp.py` | 向量化 `_match_conditions`；新增 `_rebuild_cond_index_map`；`lerp_reward()` → `combine_reward()`；去掉 `task_style_lerp`；`style_reward_scale` 默认 2.0 |
| `rsl_rl/runners/amp_plugin_runner.py` | `lerp_reward()` → `combine_reward()` |
| `rsl_rl_ppo_cfg.py` | `max_iterations` 200000→15000；去掉 `task_style_lerp=0.5`；`style_reward_scale` 1.0→2.0 |
| `curriculums.py` | `terrain_levels_vel` 和 `command_levels_vel` 改为纯距离驱动课程 |

---

## 5. 关键教训

1. **Conditional AMP 的 `_match_conditions` 必须向量化。** 逐样本 Python 循环在 mini_batch_size > 20000 时，CUDA kernel launch 开销会导致数量级的性能退化。

2. **AMP 奖励融合方式对训练效果影响巨大。** lerp(0.5) 在 style reward 本身就大于 task reward 的情况下，会严重削弱任务目标的学习信号。additive 方式（bfm_training）保留完整 task reward 信号，style reward 作为额外加分。

3. **课程学习的晋升条件不应依赖 reward weight 的绝对值。** 不同奖励融合方式下，`_episode_sums` 的尺度完全不同。基于距离的课程更稳健，直接衡量 robot 的实际行走能力。

4. **`max_iterations` 需要根据硬件实际算力设定。** RTX 3060 上每迭代 ~35s（修复前），200000 次需要 ~80 天。

---

## 6. 后续实验计划

- [ ] 停止当前训练，使用修复后的配置重新训练
- [ ] 验证训练速度改善（预期每迭代 ~6-7s）
- [ ] 观察 velocity error 趋势是否改善
- [ ] 观察地形课程是否能正常晋升
- [ ] 如果 style_reward_scale=2.0 导致步态质量下降，尝试调到 1.5 或 1.0
