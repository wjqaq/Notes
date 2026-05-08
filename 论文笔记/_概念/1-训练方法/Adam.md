---
type: concept
aliases: [Adam optimizer, AdamW, 自适应矩估计]
---

# Adam

## 定义
自适应矩估计（Adaptive Moment Estimation）优化器，结合动量（Momentum）和 RMSProp 的自适应学习率。

## 数学形式

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
$$

## 核心要点
1. 一阶矩估计 $m_t$ 提供动量方向，二阶矩估计 $v_t$ 提供自适应步长
2. 默认超参数：$\beta_1=0.9, \beta_2=0.999, \varepsilon=10^{-8}$
3. AdamW 变体将权重衰减与自适应学习率解耦
4. [[LIME]] 中用于推理时每步的 KV 扰动优化

## 代表工作
- [[LIME]]: 推理时 7 步 Adam 优化

## 相关概念
- [[Inference-time Optimization]]
