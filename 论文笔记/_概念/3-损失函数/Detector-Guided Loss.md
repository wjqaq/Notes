---
type: concept
aliases: [Detector-Guided Loss, 检测器引导损失]
---

# Detector-Guided Loss

## 定义
MHSA 中受对抗训练启发的损失函数，鼓励生成器 $G$ 产生使判别器 $D$ 判定为非幻觉的注意力修正。

## 数学形式
$$\mathcal{L}_{\text{dg}} = -\log D_{0}(\mathbf{A}') = -\log D_{0}(\mathbf{A} + G(\mathbf{A}))$$

## 核心要点
1. 对抗训练风格设计：$G$ 试图"欺骗" $D$ 使其判定修正后注意力为非幻觉
2. 仅对 hallucinatory 样本计算（非幻觉样本无需修正）
3. 是 MHSA 幻觉抑制的核心驱动力

## 代表工作
- [[MHSA]]: 引入 detector-guided loss 实现可学习的注意力修正

## 相关概念
- [[Adversarial Training]]
- [[Hallucination Detector]]
- [[Attention Steering Generator]]
