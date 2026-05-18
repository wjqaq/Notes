---
type: concept
aliases: [Regularization Loss, 正则化损失, L2 Regularization]
---

# Regularization Loss

## 定义
约束模型参数的损失项，防止过拟合或产生过大的修正。

## 数学形式
$$\mathcal{L}_{\text{reg}} = \|\Delta\mathbf{A}\|_{2}^{2}$$

## 核心要点
1. MHSA 中约束 $\Delta\mathbf{A}$ 的大小使其尽可能小
2. 保持预训练注意力结构，仅做针对性局部修正
3. 过强会抑制有效修正，过弱会导致方差增大

## 代表工作
- [[MHSA]]: 三目标训练中的正则化项

## 相关概念
- [[Detector-Guided Loss]]
- [[LVLM Output Quality Loss]]
