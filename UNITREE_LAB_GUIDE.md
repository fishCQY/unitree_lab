## 这份文档写给“RL/AMP 小白”

你可以把这个仓库理解成：**给 Unitree G1 机器人准备的一套 Isaac Lab 强化学习（RL）训练任务 + 训练/回放脚本**。

如果你现在对 RL、AMP、locomotion 都不熟，没关系：下面按“我只想跑起来 → 我想知道代码怎么组织 → 我想改配置/奖励”的顺序讲清楚。

---

## 你最需要先知道的 5 个概念（超简版）

- **环境（Environment / Env）**：一个“可交互的模拟器接口”。你给它动作（action），它返回观测（obs）、奖励（reward）、是否结束（done）。
- **策略（Policy）**：神经网络，输入 obs 输出 action。
- **训练（Train）**：让策略在环境里反复试错，最大化奖励。
- **Locomotion（运动/行走）**：这里指“让 G1 按给定速度指令走稳、走快、少摔、少耗能”。
- **AMP（Adversarial Motion Priors）**：一种把“动作风格/动作先验”融合进 RL 的方法。对初学者来说：**它仍然是 RL**，只是奖励/观测里额外加入了一些“像不像参考动作”的约束与信号（通常会有判别器 discriminator 等组件）。

---

## 仓库结构（你应该从哪看起）

仓库最关键的两块：

- **训练/回放入口脚本**：`/home/chloe/unitree_lab/scripts/rsl_rl/`
  - `train.py`：训练
  - `play.py`：加载 checkpoint 回放（可导出 onnx/jit）
- **真正的任务与配置**：`/home/chloe/unitree_lab/source/unitree_lab/unitree_lab/`
  - `tasks/`：任务（locomotion、motion_tracking）
  - `assets/`：机器人 USD/关节/执行器配置（G1、PR、AB 等）
  - `actuators/`：执行器（比如延迟、PD 等）
  - `sensors/`：传感器（IMU、ray-caster 深度相机）
  - `terrain/`：地形生成/组合（rough、random、自定义 mesh）

你跑 `train/play` 时，脚本会 `import unitree_lab.tasks`，从而把任务注册到 Gym registry 里（也就是“用字符串 task id 找到 env_cfg / agent_cfg”）。

---

## 任务是如何“被找到并启动”的（关键机制）

### 1）任务注册（Gym id）

以 locomotion 为例，G1 的任务 id 在：

- `source/unitree_lab/unitree_lab/tasks/locomotion/robots/g1/__init__.py`

里面用 `gym.register(...)` 注册了一组 id，例如：

- `unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-v0`
- `unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-Play-v0`
- `unitree_lab-Isaac-Velocity-Flat-Unitree-G1-AMP-v0`
- `unitree_lab-Isaac-Velocity-Rough-Unitree-G1-Depth-v0`
- 以及 GRU 版本（`...-GRU-...`）

每个注册项都带两个关键入口：

- **env 配置入口**：`env_cfg_entry_point`（例如 `...rough_env_cfg:UnitreeG1RoughEnvCfg`）
- **算法/agent 配置入口**：`rsl_rl_cfg_entry_point`（例如 `...rsl_rl_ppo_cfg:UnitreeG1RoughPPORunnerCfg`）

### 2）`train.py` 怎么拿到配置并创建环境

`scripts/rsl_rl/train.py` 的核心流程（概念上）：

1. 启动 Isaac Sim（`AppLauncher`）
2. 用 `@hydra_task_config(args_cli.task, args_cli.agent)` 根据 `--task` 去 registry 里加载：
   - env_cfg（环境配置类）
   - agent_cfg（PPO/Runner 配置）
3. `gym.make(args_cli.task, cfg=env_cfg)` 创建环境
4. 用 `RslRlVecEnvWrapper` 包一层，让 rsl-rl 能批量并行训练
5. `OnPolicyRunner(...).learn(...)` 开始训练
6. 日志写到 `logs/rsl_rl/<experiment_name>/<时间戳>_.../`

`play.py` 类似，只是改成加载 checkpoint 并循环推理。

---

## Locomotion（G1）实现是怎么拼起来的

### 1）Env 配置的“拼装方式”（推荐你理解这张图）

Locomotion 基类配置在：

- `tasks/locomotion/config/envs/base_env_cfg.py`

它把一个任务拆成 7 块（你以后改实验基本都改这里的某一块）：

- **Scene**：地形、灯光、传感器、机器人 USD 放在哪里
- **Commands**：给机器人什么目标（比如目标速度）
- **Actions**：策略输出怎么变成关节控制（比如 joint position action）
- **Observations**：给策略/价值网络看什么（角速度、关节状态、高度扫描…）
- **Rewards**：奖励函数项（速度跟踪、能耗、平衡、接触惩罚…）
- **Terminations**：什么时候提前结束（摔倒、超时、越界…）
- **Curriculum/Events**：课程学习与随机化（摩擦、质量、推力、reset 随机…）

G1 的 rough/flat 配置在：

- `tasks/locomotion/robots/g1/rough_env_cfg.py`
- `tasks/locomotion/robots/g1/flat_env_cfg.py`

它们是继承基类后做“覆盖/微调”，例如：

- 把 `scene.robot` 换成 G1 的 `ArticulationCfg`
- 选择 rough / random 的 terrain 组合
- 调整 observation group（比如是否启用 depth camera）
- 调整 reward 权重
- Play 版本减少 env 数、关闭随机化、放宽指令范围等

### 2）MDP（观测/奖励/终止/随机化）代码在哪里

Locomotion 的 MDP 函数集中在：

- `tasks/locomotion/mdp/observations.py`
- `tasks/locomotion/mdp/rewards.py`
- `tasks/locomotion/mdp/terminations.py`
- `tasks/locomotion/mdp/events.py`
- `tasks/locomotion/mdp/curriculums.py`
- `tasks/locomotion/mdp/commands/`（速度指令采样）
- `tasks/locomotion/mdp/symmetry/`（G1 对称映射，常用于数据增强/正则）

配置文件里看到的 `mdp.xxx` 基本都来自这里。

### 3）机器人/执行器/传感器/地形

- **机器人配置**：`assets/robots/`
  - `unitree.py`：G1（以及关节/执行器参数）
  - `unitree_parallel.py`：PR/AB 变体
  - `unitree_beyondmimic.py`：另一套 G1 资产配置
- **执行器**：`actuators/`
- **传感器**：`sensors/imu/`、`sensors/ray_caster/`
- **地形**：`terrain/`（并在 `terrain/__init__.py` 导出 `ROUGH_TERRAINS_CFG`、`RANDOM_TERRAINS_CFG` 等组合）

---

## 如何启动 Locomotion 的训练（train）

下面假设你的 Python 环境能 `import isaaclab` 与 `import isaacsim`。

### 0）先确认“可用 task id”

这个仓库自带的 `scripts/list_envs.py` 目前按 `"Template-"` 过滤，**不会显示**你现在的 `unitree_lab-...` 任务。

最稳的办法是直接用 Python 打印 registry（推荐复制执行）：

```bash
cd /home/chloe/unitree_lab
python -c "import gymnasium as gym; import unitree_lab.tasks; print('\n'.join(sorted([k for k in gym.registry.keys() if 'unitree_lab-' in k and 'Locomotion' not in k])))"
```

或者你也可以直接看文件：

- `source/unitree_lab/unitree_lab/tasks/locomotion/robots/g1/__init__.py`

### 1）训练 rough AMP（常用起点）

```bash
cd /home/chloe/unitree_lab
python scripts/rsl_rl/train.py \
  --task unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-v0 \
  --headless \
  --device cuda
```

常用可选项：

- `--num_envs 1024`：减少并行环境数（显存不够时很有用）
- `--seed 42`：固定随机种子
- `--max_iterations 20000`：限制训练迭代次数
- `--video`：训练中录视频（会自动开启 cameras）

训练输出位置（非常重要）：

- `logs/rsl_rl/<experiment_name>/<时间戳>_.../`
  - `params/env.yaml`、`params/agent.yaml`
  - `checkpoints/`（模型文件名以你实际生成的为准）
  - `videos/train/`（如果开启 `--video`）

### 2）训练 flat AMP

```bash
python scripts/rsl_rl/train.py \
  --task unitree_lab-Isaac-Velocity-Flat-Unitree-G1-AMP-v0 \
  --headless \
  --device cuda
```

### 3）训练 Depth 版本（需要 ray-caster 深度相机）

```bash
python scripts/rsl_rl/train.py \
  --task unitree_lab-Isaac-Velocity-Rough-Unitree-G1-Depth-v0 \
  --headless \
  --device cuda
```

---

## 如何启动 Locomotion 的回放（play）

### 1）最简单：用最近一次训练的 checkpoint（按 run 名查找）

```bash
cd /home/chloe/unitree_lab
python scripts/rsl_rl/play.py \
  --task unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-Play-v0 \
  --load_run <把这里换成 logs 下某个 run 目录名> \
  --checkpoint <把这里换成该 run 下的 checkpoint 文件名> \
  --device cuda
```

> `--load_run` 和 `--checkpoint` 都是为了让脚本能在 `logs/rsl_rl/<experiment_name>/` 下拼出正确路径。你也可以直接给 `--checkpoint /abs/path/to/model.pt`（play.py 会优先用它）。

### 2）想看画面（非 headless）

play 通常不加 `--headless`，并可加：

- `--real-time`：尽量按真实时间跑（否则可能“跑得很快”）
- `--video`：只录一段 play 视频（默认从第 0 步开始录）

示例：

```bash
python scripts/rsl_rl/play.py \
  --task unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-Play-v0 \
  --checkpoint /home/chloe/unitree_lab/logs/rsl_rl/<experiment>/<run>/checkpoints/<ckpt> \
  --video --video_length 400 \
  --real-time
```

play 会自动导出：

- `exported/policy.pt`
- `exported/policy.onnx`

---

## 常见问题（第一次跑最容易卡在这里）

### 1）报错 `ModuleNotFoundError: isaacsim`

这说明你在一个“没有 Isaac Sim runtime”的 Python 里跑。

你需要满足至少一个条件：

- **conda/venv 的 python 里装了 Isaac Sim pip 包**（例如 `isaacsim-rl`），并已接受 EULA；
- 或者用 Isaac Sim 自带的 `python.sh` 来运行脚本。

### 2）EULA 交互提示卡住

第一次 import `isaacsim` 需要接受协议。终端里出现 `Do you accept the EULA? (Yes/No):` 时输入 `Yes`。

### 3）不知道 checkpoint 文件名是什么

去 `logs/rsl_rl/<experiment_name>/<run>/` 里看 `checkpoints/` 文件夹，实际文件名以你训练生成的为准。

---

## 我想改 reward / observation / 地形 / 机器人，应该从哪下手？

- **想改训练任务整体结构**：从 `tasks/locomotion/config/envs/base_env_cfg.py` 入手
- **想改 G1 rough/flat 细节**：看 `tasks/locomotion/robots/g1/rough_env_cfg.py`、`flat_env_cfg.py`
- **想改 reward/obs 的具体数学**：看 `tasks/locomotion/mdp/rewards.py`、`observations.py`
- **想改速度指令采样范围**：看 `BaseCommandsCfg`（以及 `mdp/commands/velocity_command.py`）
- **想换机器人资产/执行器参数**：看 `assets/robots/unitree*.py` 与 `actuators/`
- **想换地形组合**：看 `terrain/rough.py`、`terrain/__init__.py`

---

## 只关注 Locomotion：推荐你先跑的 3 个命令

```bash
cd /home/chloe/unitree_lab

# 1) 先训练一个最基础的 rough AMP（headless）
python scripts/rsl_rl/train.py --task unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-v0 --headless --device cuda

# 2) 再用 Play 任务回放（需要你把 checkpoint 路径改成实际存在的）
python scripts/rsl_rl/play.py --task unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-Play-v0 --checkpoint /abs/path/to/model.pt

# 3) 想录视频的话
python scripts/rsl_rl/play.py --task unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-Play-v0 --checkpoint /abs/path/to/model.pt --video --video_length 400
```

