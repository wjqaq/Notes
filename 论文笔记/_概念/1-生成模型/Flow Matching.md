---
type: concept
aliases: [Flow Matching, 流匹配]
---

# Flow Matching

## 定义

一种生成模型训练范式，通过学习从噪声分布到数据分布的连续变换（flow）来生成样本，是扩散模型的替代方案。

## 数学形式

$$
\mathcal{L}_{FM} = \mathbb{E}_{t, x_0, x_1} \|v_\theta(x_t, t) - v_t\|^2
$$

其中 $v_t = x_1 - x_0$ 为真实速度场，$v_\theta$ 为模型预测。

## 核心要点

1. 相比扩散模型，flow matching 训练更稳定
2. Rectified flow 使用线性插值路径，简化训练
3. 可在像素空间或潜在空间应用

## 代表工作

- [[Tuna-2]]: 在像素空间应用 flow matching 进行图像生成
- [[Rectified Flow]]: 提出线性路径的 flow matching
- [[Stable Diffusion 3]]: 采用 flow matching 替代扩散

## 相关概念

- [[Rectified Flow]]
- [[Diffusion Model]]
- [[Score Matching]]
