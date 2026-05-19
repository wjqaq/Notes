---
category: 3-损失函数
aliases: [Reward Gap]
---

# Reward Gap Configuration (Reward Gap 配置)

DPO 训练中偏好对 $(y_w, y_l)$ 的 reward gap $\Delta r^* = r^*(y_w|x) - r^*(y_l|x)$ 的整体配置设计。核心洞察是 DPO 的训练效率取决于如何系统性设计每个偏好对的 reward gap 大小和难度分布。TPR 通过 topic 级控制实现了对 reward gap configuration 的精细塑造。

## 代表工作
- [[TPR]]: 首次系统研究 reward gap configuration 的优化问题并通过 topic 级控制实现
- [[DPO]]: DPO 论文中隐含提出的 reward modeling 视角
