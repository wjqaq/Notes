---
type: concept
aliases: [Attention Steering Generator, 注意力引导生成器]
---

# Attention Steering Generator

## 定义
MHSA 中的轻量级生成器 $G$，输入原始跨模态注意力，输出修正量 $\Delta\mathbf{A}$，通过对注意力模式的修正来抑制 LVLM 幻觉。

## 数学形式
$$\Delta\mathbf{A} = G(\mathbf{A}), \quad \mathbf{A}' = \mathbf{A} + \Delta\mathbf{A}$$

## 核心要点
1. 三层 MLP（hidden=512），输入输出维度相同
2. 残差设计：只学习偏移量而非从头预测注意力
3. 权重初始化为 $\mathcal{U}(-10^{-5}, 10^{-5})$，零偏置
4. 由检测器引导损失、正则化损失和 LVLM 质量损失联合训练
5. 仅在检测器判定为幻觉时触发修正

## 代表工作
- [[MHSA]]: 引入可学习的注意力引导生成器，超越启发式注意力操纵

## 相关概念
- [[Cross-Modal Attention]]
- [[Hallucination Detector]]
- [[Adversarial Training]]
- [[Detector-Guided Loss]]
