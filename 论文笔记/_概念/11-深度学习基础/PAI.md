---
type: concept
aliases: [PAI, Paying more Attention to Image]
---

# PAI

## 定义
一种训练无关的 LVLM 幻觉缓解方法，通过对注意力重新加权将更多 focus 分配给图像 token。

## 核心要点
1. 启发式规则：重新加权注意力而非可学习修正
2. 训练无关（inference-time method）
3. 与 OPERA 类似，使用固定策略不随样本自适应

## 代表工作
- Liu et al. (2024, ECCV): PAI 原始提出
- [[MHSA]]: 用可学习的数据驱动修正取代 PAI 的启发式规则

## 相关概念
- [[OPERA]]
- [[Cross-Modal Attention]]
- [[多模态幻觉]]
