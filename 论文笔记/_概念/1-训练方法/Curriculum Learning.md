---
category: 1-训练方法
aliases: [Curriculum Learning]
---

# Curriculum Learning (课程学习)

一种训练策略，通过从简单样本开始逐步增加训练样本的难度来训练模型。在 TPR 中，课程学习体现为 Warm-Up 阶段用贪心策略（最高/最低分单元的巨大 reward gap）和 Hard-Mining 阶段逐步提高 rejected 响应中替代单元的得分（使 reward gap 逐渐缩小）。

## 代表工作
- [[TPR]]: TPR-CL 变体利用课程学习策略系统性优化 reward gap 配置
- [[π0.5]]: 应用课程学习进行机器人策略训练
