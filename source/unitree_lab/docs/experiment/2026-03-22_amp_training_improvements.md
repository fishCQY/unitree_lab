# 实验记录：AMP 训练改进 — 奖励融合、课程学习、Sim2Sim 架构、传感器精简

**日期：** 2026-03-22
**基于运行：** wandb `run-20260322_003535-l8gqr2wk`（修复 `_match_conditions` 后的首次训练）
**训练配置：** `UnitreeG1RoughPluginRunnerCfg`, RTX 3060, 4096 envs

---

## 1. 上一轮训练观察

修复 `_match_conditions` 向量化后，训练速度从 ~35s/iter 降到 ~7s/iter，提速 5 倍。训练到 5500 迭代后观察到以下现象：

### 1.1 训练指标趋势

| 迭代 | Reward | Ep Length | Terrain Level | error_vel_xy | error_vel_yaw |
|------|--------|-----------|--------------|-------------|--------------|
| 0 | -2.56 | 19.8 | 0.80 | 0.028 | 0.107 |
| 500 | 1.26 | 99.5 | 0.00 | 0.115 | 0.368 |
| 1500 | 6.84 | 362 | 0.37 | 0.371 | 0.905 |
| 2000 | **7.95** | 368 | 1.41 | 0.361 | 0.898 |
| 3000 | **4.93** | 266 | 3.31 | 0.341 | 0.776 |
| 3500 | 5.47 | 295 | 3.82 | 0.350 | 0.763 |
| 5000 | 6.93 | 352 | 4.53 | 0.394 | 0.855 |
| 5500 | 8.03 | 390 | 4.59 | 0.392 | 0.853 |

### 1.2 问题诊断

**问题 1：velocity error 持续增大**

不是策略退步，而是环境同时在两个维度变难：
- 地形课程 0 → 4.6（阶梯 level 6-7、boxes level 6）
- 速度指令范围 `avg_vel_lin_x` 从 1.0 → 1.18

在同等难度下 tracking reward 实际持平（track_lin_vel: 1.06 → 1.11），说明策略在进步，但 error 绝对值因环境变难而上升。

**问题 2：iter 2000-3000 reward 下降**

terrain level 从 1.4 快速爬到 3.3（2 级跨度），episode length 从 368 降到 266。纯距离驱动的课程推得太快——robot 在简单地形走得远，被迅速推到更难的地形，还没适应就被降级了。

**问题 3：Sim2Sim 视频全部失败**

所有 checkpoint 的 sim2sim 子进程都因 `gladLoadGL error` 退出。原因是 MuJoCo Renderer 在 tmux headless 环境下无法初始化 OpenGL。`train.sh` 中虽有 `MUJOCO_GL=egl`，但当前训练进程是旧 shell 启动的，且子进程的 env 传递不稳定。

**问题 4：速度课程不必要**

bfm_training 不使用速度课程——速度范围从一开始就固定。速度课程增加了额外的不稳定因素，让 error 同时受地形和速度两个变量影响，难以分析。

### 1.3 时间分布

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Collection (仿真) | 5.2s | 74% |
| Learning (PPO+AMP) | 1.8s | 26% |

Collection 是瓶颈。其中 `height_scanner`（160×100 ray cast grid）和 `contact_information`（26 body 接触查询）占据大量 GPU 计算，而 policy 不使用 height_scan，critic 的 `contact_information` 边际价值低。

---

## 2. 改进方案

### 2.1 奖励融合：lerp → 直接相加（已在上轮实施）

```
之前: r_total = 0.5 * r_task + 0.5 * r_style    (task 信号被稀释)
现在: r_total = r_task + style_reward_scale * r_style * dt  (task 信号 100% 保留)
```

### 2.2 课程学习改进

**去掉速度课程**（与 bfm_training 一致）

- 从 `G1CurriculumCfg` 中移除 `command_levels`
- 速度指令范围从训练一开始就固定为 vx ±1.0 / vy ±0.5 / wz ±2.0
- error 只受地形难度一个因素影响，更容易分析

**地形课程改为混合门槛**

```python
# 之前（纯距离，推得太快）
move_up = distance > terrain_size / 2

# 现在（距离 + tracking 质量双重门槛）
lin_track_mean = episode_sums["track_lin_vel_xy_exp"] / episode_length
ang_track_mean = episode_sums["track_ang_vel_z_exp"] / episode_length
move_up = (distance > terrain_size / 2) & (lin_track_mean > 0.5) & (ang_track_mean > 0.3)
move_down = (distance < terrain_size / 4) | (lin_track_mean < 0.2)
```

per-step tracking 均值范围 [0, 1]，阈值 0.5/0.3 确保策略走得远且跟得准才晋升。

### 2.3 Sim2Sim 架构改造（子进程 → Runner 内置）

**之前（子进程外挂）：**
```
train.py monkey-patch runner.save()
  → subprocess.Popen(run_sim2sim_locomotion.py)
  → 后台线程 _sim2sim_poller 等待 → upload wandb
涉及: queue + threading.BoundedSemaphore + 信号量竞争 + 日志分散
```

**现在（bfm_training 风格，Runner 内置）：**
```
AMPPluginRunner.learn()
  → save() → _maybe_run_sim2sim(it)
    → _export_onnx()
    → _build_deploy_yaml()
    → _record_sim2sim_video()  (MUJOCO_GL=egl)
    → _log_sim2sim_to_wandb()  (commit=False)
全部同步执行，无子进程/线程/信号量
```

关键改进：
- 直接在 Runner 内调用 MuJoCo 评估，无进程间通信
- 显式设置 `MUJOCO_GL=egl` 确保 tmux/headless 正常渲染
- wandb 使用 `commit=False` 不打乱训练 step
- `run_sim2sim_locomotion.py` 顶部加 `os.environ.setdefault("MUJOCO_GL", "egl")`

### 2.4 传感器精简

| 移除项 | 类型 | 原开销 | 原因 |
|--------|------|--------|------|
| `height_scan` | critic 观测 | 高（ray cast grid） | policy 不用，critic 也不需要 |
| `height_scanner` | 场景传感器 | 高（每 0.08s 全量更新） | 无观测引用 |
| `contact_information` | critic 观测 | 中-高（26 body 查询） | `feet_contact_force` 已提供足部信息 |

预期 Collection time 从 ~5.2s 降低（主要来自去掉 height_scanner ray casting）。

---

## 3. 修改文件清单

| 文件 | 改动 |
|------|------|
| `rsl_rl/runners/amp_plugin_runner.py` | sim2sim 内置到 Runner：新增 `_maybe_run_sim2sim`、`_run_sim2sim`、`_export_onnx`、`_build_deploy_yaml`、`_record_sim2sim_video`、`_log_sim2sim_to_wandb` 等方法 |
| `scripts/rsl_rl/train.py` | 删除 ~150 行子进程外挂代码（线程/信号量/monkey-patch），改为设置 `runner.sim2sim_cfg` |
| `scripts/mujoco_eval/run_sim2sim_locomotion.py` | 顶部加 `os.environ.setdefault("MUJOCO_GL", "egl")` |
| `source/.../rough_env_cfg.py` | critic 移除 `height_scan` + `contact_information`；场景移除 `height_scanner`；课程移除 `command_levels` |
| `source/.../curriculums.py` | `terrain_levels_vel` 改为混合门槛（距离+tracking）；`command_levels_vel` 的初始/fallback 也同步 |

---

## 4. 预期效果

| 指标 | 改进前 | 改进后预期 |
|------|--------|-----------|
| Collection time | ~5.2s | ~3-4s（去掉 ray caster） |
| 每迭代总时间 | ~7s | ~5-6s |
| Sim2Sim 视频 | 全部失败（gladLoadGL） | 正常录制上传 wandb |
| Reward 稳定性 | iter 2000-3000 大幅下降 | 混合门槛防止过快晋升 |
| velocity error | 受地形+速度双重影响 | 只受地形影响，更好分析 |

---

## 5. 后续计划

- [ ] 停止当前训练，用新配置重新训练
- [ ] 验证 sim2sim 视频在 wandb 上正常显示
- [ ] 观察混合门槛课程下 reward 是否更平稳
- [ ] 对比去掉 height_scanner 后的 Collection time 降幅
- [ ] 如果地形课程太慢，可调低阈值（lin_track 0.5 → 0.4）
