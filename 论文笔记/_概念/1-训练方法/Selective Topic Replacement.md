---
category: 1-训练方法
aliases: [Selective Topic Replacement]
---

# Selective Topic Replacement (选择性 Topic 替换)

TPR 框架的核心机制。从排序后的 topic 候选池中，根据策略（Greedy/Curriculum）选择替代语义单元，替换响应模板中对应的单元，从而在 preferred 和 rejected 响应之间精确控制语义差异，塑造期望的 reward gap。

## 代表工作
- [[TPR]]: 作为 TPR 偏好对构造的核心机制
