# 实验记录：Sim2Sim 两阶段评估、地形重建、Yaw 跟踪修复

**日期：** 2026-03-22
**基于运行：** wandb `run-20260322_122036-3dj1a03n`（修复后的第二次训练）

---

## 1. 观察到的问题

### 1.1 Sim2Sim 视频是平地（无地形）

之前将 sim2sim 从子进程外挂改为 Runner 内置调用时，漏掉了地形注入步骤（`_generate_course_xml` 和 `_setup_terrain`）。MuJoCo 加载原始 XML 后只有一个空的 heightfield，没有实际地形几何体，看起来就是平地。

### 1.2 机器人出生在楼梯上，脚嵌入地形

赛道从 `x_start = -total_len/2` 开始沿 +X 建造，spawn 固定在 x=0（赛道中心）。原来的赛道以 `rough_ground` 开头，x=0 正好落在 `stairs_down` 段上，导致机器人出生在台阶中间。

### 1.3 Yaw 跟踪极差（error_yaw = 1.1 rad/s，持续增大）

| 迭代 | error_vel_yaw | track_ang_vel_z_exp | terrain_level |
|------|--------------|-------------------|--------------|
| 0 | 0.107 | 0.005 | 0 |
| 500 | 0.541 | 0.141 | 0 |
| 1000 | 0.836 | 0.346 | 0 |
| 1500 | 1.053 | 0.506 | 0 |
| 2000 | 1.115 | 0.589 | 0 |
| 3000 | 1.096 | 0.644 | 0 |

**根因分析：**
- `ang_vel_z` 范围 ±2.0 rad/s（每秒 115°），对双足 robot 在粗糙地形上几乎不可能
- LAFAN walk/run 参考动作中 yaw rate 基本在 ±0.5 rad/s 以内，AMP style reward 惩罚大角速度
- 策略学会了"放弃 yaw 跟踪、专注直线走"
- 均匀采样 ±2.0 的平均 |ωz| = 1.0，如果 robot 不转，error ≈ 1.0

### 1.4 地形课程 3000+ 迭代 0 次晋升

混合门槛中 `ang_track_mean > 0.3` 永远无法满足：
- `track_ang_vel_z_exp` per-step 均值远低于 0.3
- 地形课程被 yaw tracking 的硬性条件完全阻塞

---

## 2. 修复方案

### 2.1 Sim2Sim 改为两阶段评估（bfm_training 风格）

**Phase 1：全任务无渲染评估**
- 对配置的所有 eval_tasks（默认 `rough_forward`）运行 headless 评估
- 每个任务跑 10 个 episode，收集 survival_rate / velocity_error 等指标

**Phase 2：选中任务录视频**
- `rough_forward`（固定录）+ 表现最差的 2 个任务
- 录制 20 秒视频，含正确的地形注入

**wandb 上传：**
```
sim2sim_eval/{task}/survival_rate
sim2sim_eval/{task}/mean_velocity_error
sim2sim_eval/{task}/velocity_error_x
sim2sim_eval/{task}/velocity_error_y
sim2sim_video          → rough_forward 视频
sim2sim_video_worst_1  → 最差任务视频
```

全部使用 `commit=False`，不打乱训练 step。

### 2.2 地形注入修复

在 `_run_sim2sim` 中加入完整的地形注入逻辑：
- import `_generate_course_xml` 和 `_setup_terrain`
- 对 `course` 类型地形动态生成带 box geom 的临时 XML
- 对 heightfield 类型地形注入 heightfield 数据
- 录制完成后清理临时 XML 文件

### 2.3 赛道重新设计（IsaacLab 风格）

**布局（总长 63m）：**

```
rough(11m) → rough(2m) → stairs_up → platform → stairs_down
           → rough(2m) → slope_up → platform → slope_down
           → flat(3m, SPAWN x=0)
           → rough(2m) → stairs_up → stairs_down → slope_up → slope_down
           → rough(2m) → stairs_up → slope_down → rough → slope_up → stairs_down → rough(2m)
```

设计要点：
- spawn 在唯一的 flat 区域（x: -1.5 → +1.5），机器人在平地上初始化
- 所有地形段之间用 rough_ground 连接（灰色，与白色 flat 区分）
- 包含金字塔楼梯、金字塔斜坡、back-to-back、楼梯↔斜坡交叉衔接 4 种组合
- step_height=0.12m, slope_angle=0.18 rad (10.3°), platform_height=0.60m

**防脚嵌入：**
- 每个 box geom 增加 2cm x-overlap，相邻段在接缝处微小重叠
- MuJoCo 碰撞系统在重叠区域自动取最高接触面

**rough_ground 可视化：**
- heightfield 使用 `rgba="0.45 0.45 0.45 1"`（灰色），与白色 flat 明确区分

### 2.4 Yaw 跟踪修复（bfm_training 6 机制）

| 修改 | 之前 | 之后 |
|------|------|------|
| `ang_vel_z` 初始范围 | ±2.0 | **±1.0** |
| wz 课程 | 无 | **`ang_vel_curriculum`：delta=0.1 渐进扩展到 ±2.0** |
| 地形门槛含 ang_track | `ang_track_mean > 0.3` | **去掉**，只保留 lin_track |
| resampling_time | (4.0, 6.0) | (4.0, 6.0)（已对齐） |
| rel_standing_envs | 0.1 | 0.1（已对齐） |
| heading_command | stiffness=2.0 | stiffness=2.0（已有） |

**`ang_vel_curriculum` 工作方式：**

```python
初始: ang_vel_z = (-1.0, 1.0)

每次 episode 结束时:
  走得远 (distance > terrain_size/2) → wz 范围 ±0.1 扩展
  走不动 (distance < terrain_size/4) → wz 范围 ±0.1 收缩（不低于初始 ±1.0）

最终: ang_vel_z → (-2.0, 2.0)
```

从 ±1.0 到 ±2.0 需要 10 次成功扩展，独立于地形课程，不互相干扰。

---

## 3. 修改文件清单

| 文件 | 改动 |
|------|------|
| `rsl_rl/runners/amp_plugin_runner.py` | `_run_sim2sim` 改为两阶段评估；加入地形注入（`_create_simulator_with_terrain`）；Phase 2 选 rough_forward + worst N 录视频；wandb 上传结构化指标 + 视频；修复 `parents[1]` → `parents[2]` 路径问题 |
| `scripts/mujoco_eval/run_sim2sim_locomotion.py` | 地形生成加 2cm x-overlap 防脚嵌入；rough_ground heightfield 改为灰色 rgba；顶部加 `MUJOCO_GL=egl` |
| `mujoco_utils/evaluation/eval_task.py` | `rough_forward` 赛道重新设计：spawn 在中间 flat 区，包含楼梯↔斜坡交叉衔接，rough_ground 连接各段 |
| `curriculums.py` | `terrain_levels_vel` 去掉 ang_track 门槛；新增 `ang_vel_curriculum` 函数（只扩展 wz，delta=0.1） |
| `rough_env_cfg.py` | `ang_vel_z` ±2.0 → ±1.0；`G1CurriculumCfg` 加入 `ang_vel_levels` 课程 |

---

## 4. 预期效果

| 指标 | 改进前 | 改进后预期 |
|------|--------|-----------|
| error_vel_yaw | 1.1 rad/s（持续增大） | 初始更小（±1.0 范围），随课程渐进增大 |
| 地形课程 | 3000 迭代 0 晋升 | 不再被 yaw 阻塞，只看 lin_track + 距离 |
| Sim2Sim 地形 | 平地（无地形） | 完整赛道（楼梯+斜坡+粗糙地面） |
| Sim2Sim 视频 | 机器人出生在楼梯上 | 出生在 flat spawn zone |
| wandb 指标 | 只有视频 | 结构化指标（survival/vel_error per task）+ 视频 |
| wz 范围 | 固定 ±2.0 或固定 ±1.0 | ±1.0 → ±2.0 渐进扩展 |

---

## 5. 后续观察项

- [ ] 确认地形课程能正常晋升（不再被 yaw 阻塞）
- [ ] 观察 error_vel_yaw 趋势——应先在 ±1.0 范围内下降，再随课程扩展缓慢上升
- [ ] 确认 sim2sim 视频在 wandb 上显示正确地形
- [ ] 观察 ang_vel_curriculum 的扩展速度——`avg_vel_ang_z` 应缓慢从 1.0 上升
- [ ] 如果 wz 课程扩展太快，可将 delta 从 0.1 降到 0.05
