---
type: concept
aliases: [LVLM Output Quality Loss, LVLM 输出质量损失]
---

# LVLM Output Quality Loss

## 定义
MHSA 中确保修正后注意力产生的 LVLM 输出质量不下降的损失函数，通过交叉熵对比修正后模型的输出分布与 ground-truth。

## 数学形式
$$\mathcal{L}_{\text{LVLM}} = \text{CE}(f_{\text{LVLM}}(\mathbf{A}'), y_{\text{gt}})$$

## 核心要点
1. 使用修正后注意力 $\mathbf{A}'$ 替换原始注意力的 LVLM 输出
2. 与 ground-truth label $y_{\text{gt}}$ 计算交叉熵
3. 确保修正不仅降低幻觉特征，还保持或提升实际输出质量
4. 生成式（caption）任务中被设为 0（离线训练，避免过约束）

## 代表工作
- [[MHSA]]: 三目标联合训练中的输出质量保留项

## 相关概念
- [[Cross-Entropy Loss]]
- [[Attention Steering Generator]]
- [[Detector-Guided Loss]]
