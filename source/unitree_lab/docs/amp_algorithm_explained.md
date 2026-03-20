# AMP（Adversarial Motion Priors）算法详解

本文档详细解释 bfm_training 项目中 AMP 算法的设计、实现原理和参数调优指南。面向零基础读者，从 GAN 的基本概念讲起。

---

## 目录

1. [AMP 的作用与整体框架](#1-amp-的作用与整体框架)
2. [GAN 公式详解](#2-gan-公式详解)
3. [梯度惩罚与 Lipschitz 连续性](#3-梯度惩罚与-lipschitz-连续性)
4. [AMP 奖励公式](#4-amp-奖励公式)
5. [奖励乘以 dt 的原理](#5-奖励乘以-dt-的原理)
6. [数据镜像的实现与原理](#6-数据镜像的实现与原理)
7. [Lambda 与梯度惩罚对判别器的影响](#7-lambda-与梯度惩罚对判别器的影响)
8. [判别器网络架构](#8-判别器网络架构)
9. [AMP 观测定义](#9-amp-观测定义)
10. [条件式 AMP（Conditional AMP）](#10-条件式-ampcondditional-amp)
11. [运动参考数据](#11-运动参考数据)
12. [训练流程总结](#12-训练流程总结)
13. [全部参数详解与调参指南](#13-全部参数详解与调参指南)
14. [关键源文件索引](#14-关键源文件索引)

---

## 1. AMP 的作用与整体框架

### 1.1 AMP 是什么

AMP（Adversarial Motion Priors）是一种基于对抗学习的运动风格模仿方法，核心目的是：

1. **让策略产生的运动"像人/参考动作"**：通过训练一个判别器（Discriminator），区分"策略生成的运动"和"动捕参考运动"，再把判别器的输出转化为奖励信号反馈给策略。
2. **替代手工设计的 style reward**：传统方法需要人工设计步态、摆臂等大量 reward term，AMP 用一个判别器自动从数据中学习"什么是好运动"。
3. **支持条件式模仿（Conditional AMP）**：本项目扩展了原始 AMP，支持根据不同条件（如 walk/run）匹配不同的参考动作。

### 1.2 整体框架：GAN + RL

AMP 本质上是一个 GAN（生成对抗网络）嵌入到 RL 训练循环中的框架：

```
                    ┌──────────────────────┐
                    │   Motion Capture Data │  (专家/离线数据)
                    │   (PKL files)         │
                    └──────────┬───────────┘
                               │
                               v
┌──────────┐    观测序列    ┌──────────────┐    AMP 奖励    ┌──────────┐
│  Policy   │ ──────────> │ Discriminator │ ──────────>  │   PPO    │
│ (生成器)   │             │  (判别器)      │              │  更新策略  │
└──────────┘             └──────────────┘              └──────────┘
     ^                                                        │
     └────────────────────────────────────────────────────────┘
```

- **生成器 = RL 策略**：产生运动轨迹
- **判别器 = AMP Discriminator**：判断运动是"真实参考动作"还是"策略生成动作"
- **奖励信号**：判别器越认为是"真实动作"，给策略的奖励越高

---

## 2. GAN 公式详解

### 2.1 什么是 GAN

GAN（Generative Adversarial Network，生成对抗网络）的核心思想可以用一个比喻来理解：

- **生成器（Generator）** = 造假币的人
- **判别器（Discriminator）** = 验钞机

造假币的人不断改进技术让假币更像真币，验钞机不断升级来识别假币。两者互相对抗，最终造假币的人能造出以假乱真的"假币"。

在 AMP 中：
- **生成器 = RL 策略**（生成机器人运动）
- **判别器 = AMP Discriminator**（判断运动是"真实动捕"还是"策略生成"）

### 2.2 原始 GAN 的公式

判别器 D(x) 输出一个 0~1 的值，表示"认为输入 x 是真实数据的概率"。

**判别器的目标**——把真假分开：
- 对真实数据 x_expert：让 D(x_expert) 接近 1（正确判为真）
- 对生成数据 x_policy：让 D(x_policy) 接近 0（正确判为假）

**生成器的目标**——骗过判别器：
- 让 D(x_policy) 接近 1（让判别器误以为是真的）

### 2.3 BFM 中使用的 LSGAN（最小二乘 GAN）

原始 GAN 用 log 函数，容易出现**梯度消失**（当判别器太强时，生成器几乎得不到梯度信号）。BFM 项目使用的是 LSGAN（Least Squares GAN，最小二乘 GAN），用平方误差代替 log。

对应代码（`source/rsl_rl/rsl_rl/algorithms/amp.py` 第 213-214 行）：

```python
offline_loss = (offline_d - 1).pow(2).mean()
policy_loss = (policy_d + 1).pow(2).mean()
```

写成公式：

```
L_disc = mean( (D(x_expert) - 1)^2 ) + mean( (D(x_policy) + 1)^2 )
         \_________________________/   \__________________________/
          希望专家数据得分 -> +1           希望策略数据得分 -> -1
```

判别器像一把"打分尺"：

```
      策略数据        中间        专家数据
        |              |           |
  ──────┼──────────────┼───────────┼──────>  D(x) 的分数
       -1              0           +1
```

- 判别器给专家数据打 **+1 分**
- 判别器给策略数据打 **-1 分**

### 2.4 为什么用 LSGAN 而不是原始 GAN

原始 GAN 的 log 函数在判别器很自信时梯度接近 0（梯度消失）。LSGAN 的平方项 (D(x)-1)^2 **永远有梯度**——即使 D(x) 远离目标值，梯度信号依然存在，训练更稳定。

#### 详细解释"永远有梯度"

训练神经网络的核心方法是**梯度下降**：
- 有一个损失函数 L（衡量"现在有多差"）
- 计算 L 对网络参数的**梯度**（告诉你"往哪个方向调参数能让 L 变小"）
- 梯度越大 → 调整幅度越大 → 学得越快
- **梯度为 0 → 完全没有方向 → 学不了**

#### 原始 GAN 的 log 函数

```
L_原始 = log(1 - D(x))
```

log 函数的形状：

```
L
 |
 |\
 |  \
 |    \_____________________________   <-- 当 D(x) 很小时，曲线变平
 |                                         梯度接近 0
 └─────────────────────────────────── D(x)
  0                                1
```

当 D(x) 接近 0 或者很小的时候（判别器很自信地说"这是假的"），log 曲线**变得非常平**，斜率（梯度）接近 0。

这意味着：当策略表现很差时，恰恰是最需要学习信号的时候，**反而得不到梯度**。这就是"梯度消失"。

#### LSGAN 的平方函数

```
L_LSGAN = (D(x) - 1)^2
```

平方函数的形状（抛物线）：

```
L
 |
 |*                           *
 | *                         *
 |  *                       *
 |   *                     *
 |    *                   *
 |      *               *
 |        *           *
 |           *     *
 |              *              <-- 最低点在 D(x) = 1
 └──────────────┼──────────── D(x)
              目标=1
```

平方函数的梯度是：

```
dL/dD = 2 * (D(x) - 1)
```

用具体数字说明：

```
D(x) 当前值    距离目标(1)    梯度 = 2*(D-1)    梯度大小
-----------    ----------    ---------------    --------
D = -3         差 4          2*(-3-1) = -8      很大 -> 学得快
D = -1         差 2          2*(-1-1) = -4      较大 -> 学得较快
D = 0          差 1          2*(0-1)  = -2      中等 -> 正常学习
D = 0.5        差 0.5        2*(0.5-1)= -1      较小 -> 微调
D = 1          差 0          2*(1-1)  = 0       零   -> 到达目标，不用学了
```

关键发现：
- **D(x) 离目标越远，梯度越大** → 越差的时候学得越快
- **只有到达目标 D=1 时梯度才为 0** → 这正是我们想要的
- **不存在"半路梯度消失"的问题**

对比总结：

```
原始 GAN：策略很差 -> 判别器很自信 -> log 曲线变平 -> 梯度消失 -> 学不了
LSGAN：  策略很差 -> 离目标很远 -> 平方项很大 -> 梯度很大 -> 学得最快
```

---

## 3. 梯度惩罚与 Lipschitz 连续性

### 3.1 什么是 Lipschitz 连续性

用日常例子理解。想象你在山上走：

- **Lipschitz 连续** = 山坡的坡度有上限，不会出现垂直悬崖
- **不 Lipschitz 连续** = 山上有悬崖，走一小步就掉下去

更直白的定义：**输入变化一点点，输出最多变化 K 倍**。

```
|f(x1) - f(x2)| <= K * |x1 - x2|
```

K 越小，函数越"平滑"。

对应到判别器 D(x)：
- **平滑的 D**：输入运动稍微变一点，打分也只变一点 → 稳定
- **不平滑的 D**：输入运动稍微变一点，打分突然从 +1 跳到 -1 → 不稳定

### 3.2 为什么需要梯度惩罚

#### 问题：没有约束的判别器会"过拟合"

如果不约束判别器，它可以学出极端尖锐的决策边界：

```
没有梯度惩罚：                          有梯度惩罚：

D(x)                                   D(x)
 +1 | ████                              +1 |    /‾‾‾‾‾‾‾
    |     █                                |   /
    |     █    （悬崖式跳变）                 |  /   （平滑过渡）
 -1 |     █████                          -1 |/
    └──────────> x                          └──────────> x
      专家  策略                              专家  策略
```

左边（没有梯度惩罚）的问题：
1. 策略几乎得不到有用的信号——在策略分布区域 D(x) 几乎是常数 -1，梯度为 0
2. 策略不管怎么改善运动，判别器都说"你是假的"，给 0 奖励
3. 直到某天运动突然跨过"悬崖"，奖励突然从 0 跳到最大 → 不稳定

右边（有梯度惩罚）的好处：
1. 策略能获得**持续的改善信号**——运动越接近专家，得分逐渐增加
2. 运动的微小改善都能反映在奖励上 → 稳定训练

### 3.3 梯度惩罚计算详解

#### 前置知识：什么是"函数对输入的梯度"

先用一个简单例子。假设有一个函数：

```
f(x, y) = 3x + 2y
```

这个函数对输入 (x, y) 的梯度是：

```
梯度 = (df/dx, df/dy) = (3, 2)
```

意思是：
- x 增加 1，f 增加 3
- y 增加 1，f 增加 2

梯度的**大小**（L2 范数）：

```
||梯度|| = sqrt(3^2 + 2^2) = sqrt(13) ≈ 3.6
```

这个值代表"在最陡方向上，输入变 1 单位，输出最多变 3.6"。

#### 应用到判别器

判别器 D 是一个神经网络，输入是一个 AMP 观测向量 x（比如 100 维），输出是一个标量分数。

```
x = [关节角度1, 关节角度2, ..., 角速度x, 角速度y, ...]  (100维向量)
                        |
                   D(x) = 0.7  (一个数字)
```

"D 对输入 x 的梯度"就是：

```
梯度 = (dD/dx_1, dD/dx_2, dD/dx_3, ..., dD/dx_100)
```

每一项表示"对应那个输入维度变一点点，D 的输出变多少"。

#### 逐行拆解代码

代码位于 `source/rsl_rl/rsl_rl/algorithms/amp.py` 第 244-264 行：

```python
def grad_pen(self, obs):
    # 第一步：准备输入，告诉 PyTorch "我要对这个输入求梯度"
    obs = obs.clone().detach().requires_grad_(True)

    # 第二步：前向传播，得到每条数据的分数
    disc = self.discriminator(obs)

    # 第三步：计算 D 的输出相对于输入 obs 的梯度
    grad = torch.autograd.grad(
        disc, obs,
        grad_outputs=torch.ones_like(disc),
        create_graph=True,
        only_inputs=True,
    )[0]

    # 第四步：计算每条数据的梯度大小，然后平方取平均
    grad_pen = (grad.norm(2, dim=1) - 0).pow(2).mean()
    return grad_pen
```

各步骤的张量形状：

```
obs    形状: (256, 100)   -- 256 条专家数据，每条 100 维
disc   形状: (256, 1)     -- 每条数据一个分数
grad   形状: (256, 100)   -- 每条数据一个 100 维的梯度向量
grad.norm(2, dim=1) 形状: (256,)  -- 每条数据一个梯度大小
grad_pen 形状: 标量         -- 最终的惩罚值
```

公式：

```
GP = mean( ||梯度_x D(x)||^2 )

即：对每条专家数据 x，计算 D 对 x 的梯度大小的平方，然后取平均。
```

#### 用具体数字走一遍

假设只有 3 条专家数据，每条 4 维（简化）：

```
obs = [[1.0, 2.0, 0.5, -0.3],   <-- 第 0 条
       [0.8, 1.5, 0.7, -0.1],   <-- 第 1 条
       [1.2, 1.8, 0.3, -0.5]]   <-- 第 2 条
```

判别器输出：

```
disc = [0.95, 0.88, 0.92]   <-- 都接近 1（因为是专家数据）
```

计算梯度（每条数据得到一个 4 维梯度向量）：

```
grad[0] = [0.3, -0.1, 0.5, 0.2]    <-- D 对第 0 条数据各维度的敏感度
grad[1] = [0.8, 0.4, -0.2, 0.6]
grad[2] = [0.1, 0.05, -0.1, 0.03]
```

计算每条数据的梯度大小：

```
||grad[0]|| = sqrt(0.3^2 + 0.1^2 + 0.5^2 + 0.2^2)
            = sqrt(0.09 + 0.01 + 0.25 + 0.04)
            = sqrt(0.39)
            ≈ 0.62

||grad[1]|| = sqrt(0.8^2 + 0.4^2 + 0.2^2 + 0.6^2)
            = sqrt(0.64 + 0.16 + 0.04 + 0.36)
            = sqrt(1.20)
            ≈ 1.10

||grad[2]|| = sqrt(0.1^2 + 0.05^2 + 0.1^2 + 0.03^2)
            = sqrt(0.01 + 0.0025 + 0.01 + 0.0009)
            = sqrt(0.023)
            ≈ 0.15
```

计算惩罚：

```
grad_pen = mean( 0.62^2 + 1.10^2 + 0.15^2 )
         = mean( 0.38 + 1.21 + 0.023 )
         = 1.613 / 3
         ≈ 0.54
```

然后这个值乘以 0.5 * lambda 加到总损失里：

```
如果 lambda = 10：
梯度惩罚贡献 = 0.5 * 10 * 0.54 = 2.7
```

判别器优化时会试图减小这个值，也就是让 grad[0], grad[1], grad[2] 的梯度大小都接近 0——迫使判别器在专家数据附近变得平坦。

#### 为什么只在专家数据上算梯度惩罚

注意代码中（第 216 行）：

```python
amp_grad_pen_loss = self.grad_pen(offline_disc_in)
```

只传入了 `offline_disc_in`（专家数据），而不是策略数据。

原因：目标是让判别器在专家数据附近形成**平坦的高原**（D 约等于 +1），这样：
- 策略运动从远处向专家方向接近时，D(x) 平滑上升 → 持续的奖励改善信号
- 已经很像专家的运动不会因为微小差异导致分数剧烈波动 → 稳定

不需要在策略数据上也约束，因为策略数据分布会不断变化，而专家数据分布是固定的。

### 3.4 总损失公式

```
L_total = (D(x_expert) - 1)^2 + (D(x_policy) + 1)^2 + 0.5 * lambda * GP
          \______________________________________________/   \______________/
                    判别能力（区分真假）                        平滑约束
```

对应 `amp.py` 第 218 行：

```python
amp_loss = offline_loss + policy_loss + 0.5 * self.amp_lambda * amp_grad_pen_loss
```

---

## 4. AMP 奖励公式

策略通过判别器输出获得的奖励（`amp.py` 第 163-166 行）：

```python
disc = self.discriminator(disc_input).squeeze(-1)
reward = torch.clamp(1 - 0.25 * torch.square(disc - 1), min=0)
return reward * self.reward_weight
```

公式：

```
r_AMP = clamp(1 - 0.25 * (D(x) - 1)^2, min=0) * reward_weight
```

不同 D(x) 值对应的奖励：

```
D(x) 值     含义                        AMP 奖励
-------     ----                        --------
D = +1      判别器认为是专家动作          1.0（最大）
D = 0       介于二者之间                 0.75
D = -1      判别器认为是策略动作          0.0（最小）
D = +3      极端"像专家"                 0.0（被 clamp 截断）
```

---

## 5. 奖励乘以 dt 的原理

代码（`on_policy_runner.py` 第 149 行）：

```python
amp_rewards = self.amp.reward(obs, self.alg.storage) * self.env.unwrapped.step_dt
```

### 5.1 直觉解释

想象你在跑步，速度是 5 米/秒。如果按不同频率记录位置：

```
记录频率    每步时间间隔 dt    每步位移         10秒总步数    10秒总距离
--------    --------------    --------         ----------    ----------
10 Hz      0.1 秒            5 * 0.1 = 0.5米    100 步      100 * 0.5 = 50 米
50 Hz      0.02 秒           5 * 0.02 = 0.1米   500 步      500 * 0.1 = 50 米
100 Hz     0.01 秒           5 * 0.01 = 0.05米  1000 步     1000 * 0.05 = 50 米
```

无论记录频率多少，10 秒跑的总距离都是 50 米。关键是**每步位移 = 速度 * dt**。

AMP 奖励也是同样的道理：
- `amp.reward()` 返回的是一个**"瞬时奖励率"**（类比速度，单位：奖励/秒）
- RL 需要的是**"这一步的奖励"**（类比位移，单位：奖励/步）
- 所以要乘 dt 来转换：`每步奖励 = 奖励率 * dt`

### 5.2 如果不乘 dt 会怎样

```
仿真频率     dt       每episode步数(10s)    不乘dt的累积奖励    乘dt的累积奖励
--------    ----      ----------------     ---------------    ---------------
50 Hz       0.02s     500 步               500 * r = 500r     500 * r * 0.02 = 10r
100 Hz      0.01s     1000 步              1000 * r = 1000r   1000 * r * 0.01 = 10r
```

- **不乘 dt**：频率越高，AMP 累积奖励越大 → 换频率就要重新调所有参数
- **乘 dt**：无论频率多少，相同时间段的累积奖励相同 → **参数和仿真频率解耦**

本质是对"奖励率"做时间积分的离散近似：

```
总 AMP 奖励 ≈ r_0 * dt + r_1 * dt + r_2 * dt + ... + r_N * dt
```

### 5.3 与 RL 奖励的融合

最终的 RL 奖励（`on_policy_runner.py` 第 161 行）：

```python
self.alg.process_env_step(obs, rewards + additional_reward, dones, extras)
```

```
r_total = r_task + r_AMP * dt + r_symmetry * dt
```

---

## 6. 数据镜像的实现与原理

### 6.1 什么是数据镜像

人体是左右对称的。如果有一段"先迈左脚"的走路动作，把它左右镜像翻转，就得到"先迈右脚"的走路动作——同样是合理的运动。

```
    原始动作              镜像动作

      O                     O
     /|\                   /|\
    / | \                 / | \
   L  |  R    >>>>>>>    R  |  L
     / \                   / \
    L   R                 R   L
    ^                     ^
  左脚在前               右脚在前
```

镜像平面是身体的正中面（xOz 平面，即前后-上下平面），翻转 y 轴（左右方向）。

### 6.2 不同特征的镜像方式

代码位于 `source/light_gym/light_gym/utils/light_amp_data_loader.py` 第 82-119 行。

#### (a) 关节角度和角速度（dof_pos, dof_vel）

```python
mirrored = data[:, joint_mirror_indices] * joint_mirror_signs
```

两步操作：

**第一步：交换左右关节的索引**

```
原始：  [左肩=30°, 右肩=10°, 左膝=20°, 右膝=-5°]
         index 0    index 1    index 2    index 3

joint_mirror_indices = [1, 0, 3, 2]  -> 交换 0<->1, 2<->3

交换后：[右肩=10°, 左肩=30°, 右膝=-5°, 左膝=20°]
```

**第二步：某些关节符号取反**

有些关节的旋转方向在镜像后要反过来。比如"左髋外展 20°"镜像后变成"右髋外展 20°"，但在关节坐标下符号可能要取反。

```
joint_mirror_signs = [1, 1, -1, -1]

最终：  [10°, 30°, 5°, -20°]
```

#### (b) 角速度（root_angle_vel）

```python
mirrored = data * [-1, 1, -1]
```

坐标系：x = 前, y = 左, z = 上

```
绕 x 轴旋转（roll，身体左右摇）  -> 镜像后方向反转 -> 乘 -1
绕 y 轴旋转（pitch，前后点头）    -> 镜像后方向不变 -> 乘  1
绕 z 轴旋转（yaw，左右转头）      -> 镜像后方向反转 -> 乘 -1
```

为什么 roll 和 yaw 反转：因为它们的旋转方向和 y 轴有关。镜像翻转 y 轴后，右手定则下旋转方向就反了。pitch 绕 y 轴旋转，是在 xOz 平面内的运动，镜像不影响它。

#### (c) 投影重力（proj_grav）

```python
mirrored = data * [1, -1, 1]
```

重力在体坐标下的投影，镜像后只有 y 分量取反（因为 y 是左右方向）。

#### (d) 关键点坐标（key_points_b）

```python
points = points[:, body_mirror_indices, :]   # 交换左右肢体的点
points[:, :, 1] *= -1.0                      # y 坐标取反
```

先交换左右对应的身体部位索引（如左手点和右手点互换），再把 y 坐标取反。

### 6.3 镜像的三个作用

1. **数据量翻倍**：每条动捕数据变成两条（原始 + 镜像），训练样本加倍
2. **消除偏向性**：如果参考数据中"先迈左脚"的样本更多，判别器会偏好左脚先迈。镜像后两边等价
3. **提升对称性**：帮助策略学出左右对称的运动模式

---

## 7. Lambda 与梯度惩罚对判别器的影响

### 7.1 总损失回顾

```
L_total = (D(x_expert) - 1)^2 + (D(x_policy) + 1)^2 + 0.5 * lambda * ||梯度 D||^2
          \______________________________________________/   \______________________/
                    判别能力（区分真假）                            平滑约束
```

lambda 控制的是**判别能力 vs 平滑度**的权衡。

### 7.2 三种情况对比

#### lambda 太小（比如 0.1）——判别器过强

```
梯度惩罚几乎不起作用，判别器可以随意"尖锐"

D(x)
 +1 | ████
    |     █         悬崖式决策边界
    |     █
 -1 |     █████
    └──────────> x
      专家  策略
```

后果：
- 判别器轻松区分策略和专家，amp_loss 极低
- 但策略的 amp_reward 几乎一直为 0（D(x_policy) 稳定在 -1 附近）
- 策略无论怎么改善运动，判别器都说"假的" → **策略学不到东西**

#### lambda 适中（比如 10，默认值）——平衡

```
D(x)
 +1 |    /‾‾‾‾‾‾‾
    |   /            能区分 + 保持平滑
    |  /
 -1 |/
    └──────────────> x
      策略    专家
```

好处：
- 判别器能有效区分专家和策略数据
- 同时保持足够平滑
- 运动从差到好，奖励逐渐增加 → **策略有持续的改善方向**

#### lambda 太大（比如 1000）——判别器过弱

```
梯度惩罚极强，判别器被迫几乎变成常数

D(x)
 0.1 | ─────────────    几乎一条水平线
     |                   完全分不清真假
-0.1 |
     └──────────────> x
```

后果：
- 判别器太弱，丧失判别能力，amp_loss 居高不下
- 给所有运动的打分都差不多 → **奖励信号没有区分度**

### 7.3 总结表格

```
lambda     判别器状态          奖励信号        策略能否学习    诊断特征
------     --------          --------        ----------    --------
太小(<1)   过强，悬崖式边界    几乎一直为 0     学不了         amp_loss极低, amp_reward极低
适中(~10)  能区分 + 平滑      连续渐变         稳定改善       两者都在合理范围
太大(>100) 过弱，几乎是常数    无区分度         没方向感       amp_loss较高, grad_pen极低
```

### 7.4 实际调参建议

1. 从默认值 lambda=10 开始
2. 观察 wandb 中的 `amp_loss` 和 `grad_pen_loss` 曲线：
   - `amp_loss` 震荡剧烈或发散 → **增大 lambda**（如 10 -> 20）
   - `amp_reward` 长期为 0 → **减小 lambda**（如 10 -> 5），判别器可能太强了
   - `amp_loss` 居高不下 → **减小 lambda**（如 10 -> 5），判别器太弱了
3. 通常 **5 ~ 20** 是合理范围

---

## 8. 判别器网络架构

### 8.1 网络结构

代码（`amp.py` 第 89 行）：

```python
self.discriminator = MLP(self.disc_input_dim, 1, disc_hidden_dims, activation).to(self.device)
```

默认配置 `disc_hidden_dims = [1024, 512]` 下，网络结构为：

```
Input (amp_input_dim + cond_emb_dim)
  -> Linear(input, 1024) -> ReLU
  -> Linear(1024, 512)   -> ReLU
  -> Linear(512, 1)               <-- 无激活函数，输出原始分数
```

### 8.2 输入维度计算

- **非条件 AMP**：`disc_input_dim = num_disc_obs * num_frames`
- **条件 AMP**：`disc_input_dim = num_disc_obs * num_frames + condition_embedding_dim`

### 8.3 MLP 实现

代码位于 `source/rsl_rl/rsl_rl/networks/mlp.py`。

网络是一系列 Linear + Activation 层的顺序堆叠，最后一层没有激活函数（输出原始分数，范围不限于 0~1）。

---

## 9. AMP 观测定义

### 9.1 AMP 观测组

定义在 `source/light_gym/light_gym/tasks/locomotion/config/base_env_cfg.py` 第 275-286 行：

```python
class AmpCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos)          # 关节角度
    joint_vel = ObsTerm(func=mdp.joint_vel)          # 关节角速度
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)    # 基座角速度
    projected_gravity = ObsTerm(func=mdp.projected_gravity)  # 投影重力向量
    body_pos_b = ObsTerm(func=mdp.body_pos_b)        # 关键点体坐标
```

包含 5 种特征：

```
特征              含义                维度
----              ----                ----
joint_pos         关节角度            num_joints
joint_vel         关节角速度          num_joints
base_ang_vel      基座角速度          3
projected_gravity 投影重力向量        3
body_pos_b        关键点体坐标        num_points * 3
```

### 9.2 序列构建

AMP 使用连续 `num_frames` 帧（默认 2 帧）的观测拼接为判别器输入，这样判别器可以感知**运动的时序特征**（速度、加速度模式等），而非仅仅静态姿态。

---

## 10. 条件式 AMP（Conditional AMP）

### 10.1 条件嵌入

代码（`amp.py` 第 72-77 行）：

```python
self.num_conditions = int(offline_dataset.get("num_conditions", 2))
self.condition_embedding_dim = condition_embedding_dim
self.condition_embedding = nn.Embedding(self.num_conditions, self.condition_embedding_dim)
self.disc_input_dim = self.amp_input_dim + self.condition_embedding_dim
```

条件 ID（walk=0, run=1）通过 `nn.Embedding` 映射到 16 维向量，拼接到 AMP 观测后面输入判别器。

### 10.2 条件判定

定义在 `source/light_gym/light_gym/tasks/locomotion/mdp/observations.py`：

```python
def vel_cmd_condition_id(env, command_name="base_velocity", vx_index=0, vx_threshold=1.1):
    cmd = env.command_manager.get_command(command_name)
    vx = cmd[:, vx_index]
    cond_id = (vx >= vx_threshold).to(torch.long)  # 0=walk, 1=run
    return cond_id.unsqueeze(-1)
```

当 vx >= 1.1 m/s 时为 run（ID=1），否则为 walk（ID=0）。

### 10.3 条件匹配采样

在 mini-batch 构造时，保证策略数据和离线数据的条件匹配——walk 的策略轨迹只和 walk 的参考数据配对，run 同理。

---

## 11. 运动参考数据

### 11.1 数据格式

参考数据存储为 PKL 文件，包含 retarget 后的动捕数据。使用条件式加载：

```python
amp_conditions = {
    "walk": ["lafan_walk_clips.pkl"],
    "run":  ["lafan_run_clips.pkl"],
}
load_conditional_amp_data(
    amp_conditions,
    keys=["dof_pos", "dof_vel", "root_angle_vel", "proj_grav", "key_points_b"]
)
```

### 11.2 数据加载器

代码位于 `source/light_gym/light_gym/utils/light_amp_data_loader.py`。

加载后返回的数据结构：

```
{
    "motion_data":    Tensor (N, D)    -- N 个帧，每帧 D 维特征
    "motion_step":    Tensor (N,)      -- 每帧在原动作中的位置(0,1,2...)
    "motion_ids":     Tensor (N,)      -- 每帧所属的动作 ID
    "num_motions":    int              -- 动作总数
    "feature_dim":    int              -- 特征维度
    "condition_ids":  Tensor (N,)      -- 每帧的条件 ID（条件 AMP 时）
    "num_conditions": int              -- 条件数量
}
```

---

## 12. 训练流程总结

```
每个 training iteration:

  === Rollout 阶段 =========================================
  for each env_step:
    1. policy 生成 action
    2. env.step() 获得 obs, task_reward
    3. AMP: discriminator(obs_序列) -> amp_reward
    4. total_reward = task_reward + amp_reward * dt
    5. 存入 RolloutStorage
  ============================================================

  === Update 阶段 ============================================
    1. PPO 更新策略（alg.update）
    2. AMP 更新判别器（amp.update）
       - 从 storage 和离线数据各采样 mini-batch
       - 计算 LSGAN loss + 梯度惩罚
       - 反向传播更新判别器参数
    3. 同步 normalizer（多 GPU 时）
  ============================================================
```

---

## 13. 全部参数详解与调参指南

### 13.1 核心参数表

```
参数                    默认值       含义
----                    ------       ----
reward_weight           2.0          AMP 奖励权重（最关键参数）
amp_lambda              10.0         梯度惩罚权重
num_frames              2            连续帧数
disc_hidden_dims        [1024, 512]  判别器隐藏层
obs_normalization       True         观测归一化
lr_scale                1.0          判别器 LR 缩放因子（相对 PPO LR）
learning_rate           1e-4         判别器固定学习率（lr_scale=None 时使用）
num_learning_epochs     2            每个 PPO 更新中判别器训练轮数
num_mini_batches        2            判别器 mini-batch 数
noise_scale             None         观测噪声
condition_embedding_dim 16           条件嵌入维度
```

### 13.2 调参优先级与策略

**第一优先：`reward_weight`（AMP 奖励权重）**

- 运动不自然但任务完成度高 → 增大 reward_weight（如 2.0 -> 3.0 -> 5.0）
- 运动好看但任务完不成 → 减小 reward_weight（如 2.0 -> 1.0 -> 0.5）
- 推荐范围：0.5 ~ 5.0

**第二优先：`amp_lambda`（梯度惩罚权重）**

- amp_loss 震荡剧烈或发散 → 增大 amp_lambda
- amp_loss 很低但 amp_reward 也很低（判别器太强）→ 减小 amp_lambda
- 推荐范围：5 ~ 20

**第三优先：`lr_scale` / `num_learning_epochs`（判别器训练强度）**

- 判别器和策略需要平衡
- 减小 lr_scale 或 num_learning_epochs 让判别器训练更慢

**第四优先：`noise_scale`（输入噪声）**

- 判别器过拟合时可以加噪声

### 13.3 诊断方法

```
监控指标         理想状态                  异常及对策
--------         --------                  ----------
amp_loss         稳定在较低值，缓慢下降     震荡/发散 -> 增大 lambda, 减小 LR
grad_pen_loss    保持较低且稳定             持续增大 -> 减小 lambda
amp_reward       随训练逐渐增大             始终接近0 -> 判别器太强, 减小训练强度
实际运动效果      自然、像参考动作           抖动/不自然 -> 检查 AMP 观测是否匹配参考数据
```

---

## 14. 关键源文件索引

```
组件                文件路径
----                --------
AMP 算法主体        source/rsl_rl/rsl_rl/algorithms/amp.py
AMP 配置类          source/rsl_rl/rsl_rl/isaaclab_rl/amp_cfg.py
数据加载器          source/light_gym/light_gym/utils/light_amp_data_loader.py
Runner 集成         source/rsl_rl/rsl_rl/runners/on_policy_runner.py
MLP 网络            source/rsl_rl/rsl_rl/networks/mlp.py
AMP 观测定义        source/light_gym/light_gym/tasks/locomotion/config/base_env_cfg.py
条件观测函数        source/light_gym/light_gym/tasks/locomotion/mdp/observations.py
训练配置示例        source/light_gym/light_gym/tasks/locomotion/config/agents/rsl_rl_ppo_cfg.py
```

---

## 附录：关键设计亮点

1. **LSGAN 而非标准 GAN**：更稳定，不需要 sigmoid 输出，不存在梯度消失
2. **梯度惩罚目标为 0（非 1）**：鼓励判别器在专家分布附近梯度为零（plateau），提供更平滑的奖励信号
3. **条件 AMP 用 Embedding 而非 One-hot**：固定维度，可以轻松添加新条件而不改变网络结构
4. **奖励 * step_dt**：对齐不同仿真频率下的 reward 尺度
5. **条件匹配采样**：mini-batch 中保证策略和离线数据的条件一致，避免 walk 动作和 run 参考配对
6. **镜像数据增强**：左右对称镜像，数据量翻倍，提升对称性和泛化能力
