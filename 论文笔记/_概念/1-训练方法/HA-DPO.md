---
category: 1-训练方法
aliases: [HA-DPO, Hallucination-Aware DPO]
---

# HA-DPO (Hallucination-Aware DPO)

一种早期的 VLM 幻觉缓解偏好优化方法。在 preference 数据中引入幻觉感知（hallucination-aware）的对比信号，用于 DPO 训练。

## 代表工作
- [[HA-DPO]]: Zhao et al. (2023)
- [[TPR]]: HA-DPO 属于 rewriting 类方法，TPR 证明其幻觉类型分布与模型自身 failure mode 存在差异
