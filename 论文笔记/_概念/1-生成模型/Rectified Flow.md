---
type: concept
aliases: [Rectified Flow, 矫正流]
---

# Rectified Flow

## 定义

Flow Matching 的一种特殊形式，使用线性插值路径连接噪声和数据分布，使 flow 尽可能"直"，从而简化采样过程。

## 数学形式

$$
x_t = t x_1 + (1-t) x_0, \quad t \in [0, 1]
$$

其中 $x_1$ 为数据样本，$x_0$ 为噪声样本。

## 核心要点

1. 线性路径使 ODE 求解更简单，减少采样步数
2. 可通过"reflow"操作进一步拉直轨迹
3. 在图像生成任务上表现优异

## 代表工作

- [[Tuna-2]]: 采用 rectified flow 进行像素空间生成
- [[FLUX.1]]: 使用 rectified flow 的高质量生成模型
- [[InstaFlow]]: 一步生成的 rectified flow 模型

## 相关概念

- [[Flow Matching]]
- [[Diffusion Model]]
- [[ODE Solver]]
