---
type: concept
aliases: [First Logit Boosting, 首 token Logit 增强]
---

# 首词Logit增强

## 定义
[[FLB]] 的核心机制：存储首个解码步的 logit $l_0$，在后续每一步解码中通过[[时变权重函数]]加权叠加到当前 logit 上，持续注入最强的视觉定位信号。

## 数学形式

**存储**: $l_{0} = \mathrm{logit}_{\theta}(y \mid x, v)$

**叠加**: $y_{t} \sim \operatorname{softmax}\!\Big[\mathrm{logit}_{\theta}(y \mid v, x, y_{<t}) + w_{t}\,l_{0}\Big]$

**约束**: 需满足 $y_t \in \mathcal{V}_{\text{head}}(y_{<t})$（[[自适应合理性约束]]）

## 核心要点
1. **为什么是首 token**：首 token 离视觉 token 最近，[[长程衰减]]最轻，视觉信息最强
2. **单次计算**：$l_0$ 只需计算一次，后续步骤零额外前向传播
3. **双重效应**：直接视觉定位（视觉信号保持）+ 隐式视觉引用（"The" 效应稳定预测）

## 代表工作
- [[FLB]]: 提出者

## 相关概念
- [[时变权重函数]]
- [[自适应合理性约束]]
- [[长程衰减]]
- [[视觉定位]]
