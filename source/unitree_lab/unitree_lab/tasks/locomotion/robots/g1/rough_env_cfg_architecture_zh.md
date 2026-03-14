# `rough_env_cfg.py` 架构与模块实现详解

本文档面向文件 `source/unitree_lab/unitree_lab/tasks/locomotion/robots/g1/rough_env_cfg.py`，目标是说明：

- 配置层级与对象装配关系
- 场景、观测、指令、奖励、终止、课程学习等模块的具体实现
- 训练配置与 Play 配置的行为差异
- 关键参数如何共同影响策略学习

---

## 1. 文件在工程中的定位

该文件是 **G1 人形机器人粗糙地形行走任务** 的环境配置入口，使用 Isaac Lab 的 `@configclass` 机制组织模块化配置。

文件顶部注释给出的继承关系如下：

- `LocomotionEnvCfg`（基础环境模板，定义通用 locomotion 环境组件）
- `UnitreeG1RoughEnvCfg`（本文件主配置，绑定 G1+rough terrain+AMP）
- `UnitreeG1RoughEnvCfg_PLAY`（推理/演示配置，减少规模并关闭部分扰动）

它本质上是“把 MDP 所需的一切配置项拼装到一起”，包括：

- 场景（机器人、地形、传感器）
- 观测（policy/critic/debug/image/AMP）
- 指令（速度命令采样）
- 事件（域随机化、reset、外力冲击）
- 奖励、终止条件、课程学习

---

## 2. 顶层依赖与设计模式

## 2.1 依赖来源

- `isaaclab.*`：仿真、场景、传感器、配置容器
- `unitree_lab.tasks.locomotion.mdp`：所有 MDP 函数实现（观测项函数、奖励函数、重置函数、随机化函数）
- `UNITREE_G1_CFG`：G1 机器人关节与动力学资产配置
- `ROUGH_TERRAINS_CFG`：粗糙地形生成器配置
- 自定义传感器：
  - `NoiseRayCasterCameraCfg`（带噪声与延迟的深度射线相机）
  - `DelayedImuCfg`（可配置延迟 IMU）

## 2.2 配置模式

该文件大量使用：

- `@configclass`：把类当作配置对象（非传统业务逻辑类）
- `ObsTerm / RewTerm / EventTerm / DoneTerm / CurrTerm`：声明式定义 MDP 组成项
- `SceneEntityCfg`：通过名字绑定场景实体（robot/sensor/body/joint 正则筛选）

这是一种“声明式 MDP 编排”：本文件定义“用什么”，具体“怎么算”由 `mdp` 模块负责。

---

## 3. 场景模块：`G1SceneCfg`

`G1SceneCfg` 继承自 `InteractiveSceneCfg`，负责定义仿真世界中的实体。

## 3.1 地形 `terrain`

- 使用 `TerrainImporterCfg`，`terrain_type="generator"`，地形生成器为 `ROUGH_TERRAINS_CFG`
- `max_init_terrain_level=0`：训练初始从较低难度地形开始（配合 curriculum 逐步提高）
- 材质配置：
  - 物理材质：静摩擦/动摩擦均为 1.0，恢复系数 1.0
  - 视觉材质：瓷砖 MDL 贴图，`texture_scale=(0.25, 0.25)`

这部分决定了接触动力学和视觉外观，尤其摩擦参数会显著影响步态稳定性。

## 3.2 机器人占位 `robot`

- `robot: ArticulationCfg = MISSING`
- 在最终环境类 `UnitreeG1RoughEnvCfg.__post_init__` 中再注入 `UNITREE_G1_CFG`

这种“延后绑定”让 `G1SceneCfg` 可复用：场景类先声明需要机器人，具体机器人资产后续装配。

## 3.3 高度扫描器 `height_scanner`

- 挂载到 `torso_link`
- 网格采样：`resolution=0.1`，覆盖区域 `size=[1.6, 1.0]`
- `ray_alignment="yaw"`：随机器人航向对齐
- 仅对 `/World/ground` 射线检测

作用：提供局部地形高度信息，常用于 critic 的特权观测，帮助价值函数理解地形几何。

## 3.4 深度相机 `depth_camera`

- 自定义 `NoiseRayCasterCameraCfg`，挂在 torso 附近，采用 ROS 坐标约定
- 分辨率较低（32x24），典型用于轻量深度感知
- 噪声与退化模型：
  - 高斯噪声 `std=0.03`
  - dropout `0.1`
  - 有效范围 `[0.2, 3.0]`
  - 延迟 `latency_steps=2`

这部分是在训练阶段主动注入传感器不确定性，缩小 sim-to-real 差距。

## 3.5 接触传感器 `contact_forces`

- 跟踪机器人全身接触，`history_length=3`
- `track_air_time=True`：可用于步态相关奖励（如 `feet_air_time`）

## 3.6 光照与 IMU

- `sky_light`：DomeLight HDR 环境光
- `imu`：`DelayedImuCfg` 挂载 pelvis，默认延迟范围 `(0.0, 0.0)`（当前不启用延迟）

---

## 4. 指令模块：`G1CommandsCfg`

目前只定义 `base_velocity`（`UniformVelocityCommandCfg`）：

- 每 10s 重新采样一次命令
- 5% 环境站立（`rel_standing_envs=0.05`）
- 开启 heading 指令（`heading_command=True`）
- 范围：
  - `lin_vel_x`: `[-2.0, 2.0]`
  - `lin_vel_y`: `[-0.5, 0.5]`
  - `ang_vel_z`: `[-2.0, 2.0]`
  - `heading`: `[-pi, pi]`

作用是构建“多任务速度跟踪”训练分布，覆盖前进、侧移、转向。

---

## 5. 观测模块：`G1ObservationsCfg`

这个模块是文件中最关键的部分之一，定义了多个观测组（不同网络/用途）。

## 5.1 `PolicyCfg`（策略输入）

包含：

- IMU 角速度、投影重力
- 当前速度命令
- 相对关节位置/速度
- 上一时刻动作

特点：

- 全部可拼接（`concatenate_terms=True`）
- 开启观测扰动（`enable_corruption=True`）
- 对若干项设置 clip 和均匀噪声

这是“仅本体感知 + 命令”的主策略输入，不含地形高度图（后面主配置中显式置空）。

## 5.2 `CriticCfg`（价值网络特权输入）

在 policy 基础上增加大量特权信息：

- `base_lin_vel`、`height_scan`
- 关节力矩、关节加速度
- 足端速度/接触力
- 机身质量变化、接触材质、质心
- 各执行器动作延迟
- 外力/外力矩
- 全身关键部位接触信息

特点：

- `enable_corruption=False`（通常不对 critic 额外加噪）
- 包含域随机化“隐变量”的显式观测，有助于稳定训练与 credit assignment

这是典型 asymmetric actor-critic：actor 用有限观测，critic 用更多状态信息。

## 5.3 `DebugCfg` 与 `ImageCfg`

- `DebugCfg.height_scan` 实际调用 `mdp.depth_scan`，用于调试深度流
- `ImageCfg.depth_image`：输出归一化深度图（范围映射到 `[0,1]`）

可以用于可视化、感知策略、或后续多模态扩展。

## 5.4 AMP 观测组：`DiscriminatorCfg` / `DiscriminatorDemoCfg`

用于 Adversarial Motion Priors（AMP）：

- `disc_agent.amp_agent_obs`：从当前机器人状态提取 AMP 特征
- `disc_demo.amp_demo_obs`：从大量动作文件（`walk/run/turn/side step`）读取示例特征
- 两者都使用 `disc_obs_steps=2`（短时序堆叠）

这部分支持“任务奖励 + 风格判别”联合训练，提升动作自然性。

---

## 6. 事件模块：`G1EventCfg`

事件是按生命周期触发的干预项：

## 6.1 `startup`（环境创建时）

- 刚体材质随机化（摩擦、恢复系数）
- torso 增减质量
- 非 torso 链节质量缩放
- torso 质心扰动
- 执行器刚度/阻尼缩放
- 关节 armature 缩放

目的：系统化域随机化，提高鲁棒性与泛化。

## 6.2 `reset`（每回合重置）

- 机体位姿与速度随机重置
- 关节位置按比例扰动（`0.5~1.5`），速度归零

## 6.3 `interval`（时间间隔触发）

- 随机外力冲击 `apply_external_force_torque_stochastic`
- 当前 `interval_range_s=(0.0, 0.0)`，但由 `probability=0.002` 决定触发概率

该设计相当于“稀疏冲击扰动”，训练抗干扰恢复能力。

---

## 7. 奖励模块：`G1RewardsCfg`

奖励由“任务项 + 约束项 + 稳定项 + 惩罚项”组成。

## 7.1 任务核心

- `track_lin_vel_xy_exp`：跟踪线速度指令
- `track_ang_vel_z_exp`：跟踪角速度指令

## 7.2 运动稳定与能耗

- `lin_vel_z_l2`、`ang_vel_xy_l2`：抑制不必要机身抖动
- `energy`、`dof_acc_l2`、`action_rate_l2`：抑制过激控制与能耗

## 7.3 接触与步态质量

- `undesired_contacts`：惩罚非足部接触
- `feet_air_time`：鼓励合理摆动期
- `feet_slide`、`feet_force`、`feet_stumble`：约束落足质量
- `feet_too_near`：限制双足间距过小

## 7.4 姿态与关节偏置

- `body_orientation_l2`、`flat_orientation_l2`
- `dof_pos_limits`：关节越界惩罚
- `joint_deviation_*`：限制部分关节偏离默认姿态

## 7.5 回合终止惩罚

- `termination_penalty`：提前失败给大惩罚（-200）

---

## 8. 终止与课程学习

## 8.1 `G1TerminationsCfg`

- `time_out`：时间到自动结束
- `base_contact`：torso 接触地面超过阈值即失败

## 8.2 `G1CurriculumCfg`

- `terrain_levels = mdp.terrain_levels_vel`

常见行为是根据速度跟踪表现调整地形难度层级，实现自动“由易到难”训练。

---

## 9. 环境总装：`UnitreeG1RoughEnvCfg`

该类把前面所有子配置汇总到基础环境中：

- `scene/observations/commands/rewards/terminations/events/curriculum`

并在 `__post_init__` 进行关键二次改写：

1. 注入机器人资产  
   `self.scene.robot = UNITREE_G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")`

2. 调整动作接口  
   `joint_pos.scale=0.25`，并放宽 clip

3. 关闭 policy 高度扫描  
   `self.observations.policy.height_scan = None`  
   （说明 actor 不直接用地形扫描）

4. 调整奖励权重（AMP 场景重点）
   - 强化/平衡速度项与脚步项
   - 将若干关节姿态偏置项置 0（由判别器学习动作风格）
   - 降低部分非核心接触/足力惩罚

这一段是“从通用配置到实验配方”的关键：同一套模块，不同权重会得到显著不同步态风格。

---

## 10. Play 配置：`UnitreeG1RoughEnvCfg_PLAY`

`PLAY` 继承训练配置，并做推理友好化修改：

- 同步深度相机更新周期到控制周期
- `num_envs` 从 4096 降到 50（节省资源）
- 地形网格降为 `5x5`，关闭 curriculum
- 关闭 policy 观测扰动（`enable_corruption=False`）
- 关闭随机外力事件（`events.base_external_force_torque=None`）
- 保持速度命令范围与训练一致

目标是：减少随机性与规模，提升可视化和评估稳定性。

---

## 11. 数据流与执行顺序（概念）

每一步可概括为：

1. 命令生成器采样/保持 `base_velocity`
2. 传感器更新（IMU、接触、高度/深度射线）
3. 组装 observation group（policy/critic/AMP）
4. 策略输出动作，按 action scale 写入关节目标
5. 仿真推进并触发事件（reset/interval）
6. 计算奖励项与终止项
7. curriculum 依据统计调整地形难度

---

## 12. 关键实现思想总结

- **模块化声明**：配置与算法函数解耦，本文件负责编排，`mdp` 负责计算细节
- **非对称观测**：actor 简化、critic 特权，提升可训练性
- **强域随机化**：质量、摩擦、COM、增益、外力共同提升鲁棒性
- **AMP 融合**：任务奖励负责“会走”，判别器负责“走得像”
- **训练/推理分离**：同源配置通过 `PLAY` 做轻量和去随机化切换

---

## 13. 如果你要改这个文件，优先关注哪些杠杆

- 想提升速度跟踪：先调 `track_lin_vel_xy_exp` / `track_ang_vel_z_exp` 权重与 `std`
- 想减少摔倒：看 `base_contact`、`body_orientation_l2`、`lin_vel_z_l2`
- 想动作更自然：调 AMP 观测步长、demo 动作库、`joint_deviation_*` 权重
- 想泛化更强：扩展 `G1EventCfg` 随机化范围与概率
- 想加视觉策略：在 `policy` 中引入 `image.depth_image`（并处理网络结构变化）

---

## 14. 分阶段实验手册（参数改动 -> 预期现象 -> 排障建议）

本手册面向你当前 rough + AMP 训练，按常见曲线阶段拆分。  
建议每次只改 1~2 个变量，跑固定步数后再比较（避免多变量耦合导致无法归因）。

---

## 14.1 阶段 A：训练前 0~5k（快速建立“能走+不炸”）

这一阶段目标不是最优回报，而是先拿到稳定上升趋势和较低方差。

### 建议改动 1：先关闭随机外力冲击

- 参数改动：
  - `self.events.base_external_force_torque = None`
- 预期现象：
  - `Train/mean_reward` 抖动显著减小
  - 负向尖刺次数下降
  - `Train/mean_episode_length` 更快靠近上限
- 若不符合预期，排障建议：
  - 检查是否误用了 `PLAY` 任务或旧 checkpoint 继续训练
  - 对比 `events` 日志项是否确实被禁用

### 建议改动 2：将侧向速度先收敛到 0

- 参数改动：
  - `lin_vel_y: (-0.0, 0.0)`（临时）
- 预期现象：
  - 早期更容易学到稳定前进 gait
  - `track_lin_vel_xy` 更快上升
- 若不符合预期，排障建议：
  - 检查是否同时开了过强风格约束（见下一条）
  - 检查重置姿态随机范围是否过大（`reset_base` / `reset_robot_joints`）

### 建议改动 3：降低 AMP 对任务奖励的干扰

- 参数改动（推荐二选一）：
  - `task_style_lerp: 0.5 -> 0.7`（更偏任务）
  - 或 `style_reward_scale: 1.0 -> 0.5`
- 预期现象：
  - `Train/mean_reward` 更连续，出现“先涨后崩”的概率降低
  - `Policy/mean_noise_std` 回落更稳
- 若不符合预期，排障建议：
  - 关注 `amp/disc_score` 与 `amp/disc_demo_score` 是否快速饱和
  - 若判别器学得太快，可再降 `disc_learning_rate`

---

## 14.2 阶段 B：5k~50k（稳定提升，抑制中期回退）

这一阶段常见问题是“能跑但 reward 回落”，本质多为课程学习加速后的非平稳。

### 建议改动 1：降低 PPO 主学习率

- 参数改动：
  - `learning_rate: 1e-3 -> 3e-4`
  - 保留 `schedule="adaptive"` 不变先观察
- 预期现象：
  - reward 曲线斜率略慢，但回退幅度变小
  - `Policy/mean_noise_std` 不再明显二次上升
- 若不符合预期，排障建议：
  - 记录 `kl_mean`（若可见）；若 KL 长期极低，可适当回调到 `5e-4`
  - 若 value loss 偏大，考虑减小 critic 网络宽度（次优先）

### 建议改动 2：恢复 actor 的地形观测（仅 rough 任务）

- 参数改动：
  - 去掉 `self.observations.policy.height_scan = None`
- 预期现象：
  - 地形难度上升后，reward 下滑幅度减小
  - 足端滑移和躯干姿态惩罚项更可控
- 若不符合预期，排障建议：
  - 检查 `height_scanner` 更新周期是否正确（`decimation * dt`）
  - 若观测维度突增导致学习慢，可先降低噪声或减小扫描分辨率

### 建议改动 3：温和化课程学习（可选）

- 参数改动（任选其一）：
  - 将 `max_init_terrain_level` 暂设为 0（已是 0 可保持）
  - 临时关闭 `terrain_generator.curriculum`
- 预期现象：
  - reward 更平滑，尤其是 10k 以后不易掉头
- 若不符合预期，排障建议：
  - 若过于平滑但上限偏低，说明任务挑战不足，再逐步恢复 curriculum

---

## 14.3 阶段 C：收敛期 50k+（提高上限与泛化）

目标从“稳定”转向“上限 + 迁移”。

### 建议改动 1：逐步恢复训练扰动

- 参数改动：
  - 重新开启外力冲击，但降低强度：
    - `probability: 0.002 -> 0.0005`
    - `force_range` 先减半
- 预期现象：
  - reward 短期小幅下降，随后恢复
  - sim2sim 失败姿态减少，抗扰动能力增强
- 若不符合预期，排障建议：
  - 若奖励断崖式下滑，说明恢复过猛，先只恢复概率或只恢复某一轴力

### 建议改动 2：放开侧向命令并分阶段扩展

- 参数改动：
  - `lin_vel_y: 0 -> +-0.2 -> +-0.35 -> +-0.5`（分 2~3 次实验）
- 预期现象：
  - 每次扩展后短期回报下滑，再次爬升
  - 横向机动能力提升
- 若不符合预期，排障建议：
  - 同步检查 AMP demo 是否覆盖侧移动作（若覆盖不足，优先补 demo）
  - 或临时再提高 `task_style_lerp`，减弱风格冲突

### 建议改动 3：回调风格比重，追求“自然步态”

- 参数改动：
  - 在任务稳定后，把 `task_style_lerp` 从 `0.7` 回调到 `0.6` 或 `0.5`
- 预期现象：
  - 动作更自然，足端冲击更柔和
  - 任务回报可能略降，但视觉质量提升
- 若不符合预期，排障建议：
  - 若任务明显退化，说明 demo 分布和命令分布仍不匹配，应优先补充 demo 而非继续降 lerp

---

## 14.4 建议的最小实验矩阵（3 组）

- **实验 A（稳态基线）**：
  - 关外力 + `lin_vel_y=0` + `task_style_lerp=0.7` + `lr=3e-4`
- **实验 B（恢复 rough 感知）**：
  - 在 A 基础上恢复 `policy.height_scan`
- **实验 C（泛化增强）**：
  - 在 B 基础上逐步放开 `lin_vel_y`，并小幅恢复外力扰动

每组建议至少跑到同一里程碑（如 30k 或 50k），比较：

- `Train/mean_reward`（趋势、方差、是否中期回撤）
- `Train/mean_episode_length`（是否稳定靠近上限）
- `Policy/mean_noise_std`（是否出现中后期反弹）
- AMP 指标（`disc_score` / `disc_demo_score` 是否过早饱和）

---

## 14.5 常见故障到参数定位（速查）

- **现象：reward 先升后明显回落，但 episode_length 仍高**
  - 优先排查：课程学习变难 + actor 缺地形观测 + AMP 比重过高
- **现象：reward 经常出现深度负尖刺**
  - 优先排查：外力冲击事件过猛、终止惩罚过重
- **现象：`mean_noise_std` 中后期上升**
  - 优先排查：学习率偏大、任务与风格目标冲突、命令分布扩展过快
- **现象：sim2sim 比训练内评估差很多**
  - 优先排查：观测噪声/延迟建模不匹配、随机化覆盖不均匀、视觉/本体输入不一致

---

## 14.6 执行原则（避免无效试验）

- 一次只改 1~2 项，保留对照组
- 固定随机种子跑至少 2 次再下结论
- 先追求“稳定可复现”，再追求“高上限”
- 任何“先涨后崩”都优先怀疑目标非平稳，而不是单看某一奖励权重
