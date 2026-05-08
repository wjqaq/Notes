---
type: concept
aliases: [EMA, 指数移动平均, momentum encoder]
---

# Exponential Moving Average

## 定义
指数移动平均，通过动量更新维护参数的平滑版本：$\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$，常用于 SSL 中防止坍塌。

## 数学形式
$$\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$$

其中 $\lambda \in [0, 1]$ 为动量系数（通常接近 1，如 0.996）。

## 核心要点
1. 在 SSL/DINO 中，EMA 教师提供稳定的软目标，防止表示坍塌
2. TIPSv2 提出 Head-only EMA：仅对投影头做 EMA，视觉编码器共享权重
3. Head-only EMA 减少 42% 训练参数，性能持平甚至略优
4. 完全移除 EMA 会导致训练严重不稳定

## 代表工作
- [[TIPSv2]]: Head-only EMA 的首次提出
- [[DINO]]: 全模型 EMA 教师的经典应用
- BYOL: 使用 EMA 教师进行无监督表示学习

## 相关概念
- [[Self-Distillation]]
- [[DINO]]
- [[Knowledge Distillation]]
