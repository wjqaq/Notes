---
category: 1-训练方法
aliases: [Hard Negative Mining, Hard Mining]
---

# Hard Negative Mining (难负样本挖掘)

一种训练技术，通过向模型展示与正样本难以区分的负样本（hard negatives）来增强其辨别能力。在 TPR-CL 中，Hard-Mining 阶段通过逐步提高 rejected 响应 $y_l$ 中替代单元的得分（从 bottom 20% 到 bottom 20-40%），使 $y_l$ 中的幻觉越来越难以与 $y_w$ 区分，从而让模型学习辨别细微的错误。

## 代表工作
- [[TPR]]: TPR-CL 的 Hard-Mining 阶段在 Quantities (+20.8) 和 Spatial Relations (+35.7) 上带来显著提升
