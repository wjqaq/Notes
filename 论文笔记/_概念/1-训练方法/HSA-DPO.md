---
category: 1-训练方法
aliases: [HSA-DPO]
---

# HSA-DPO (Hallucination Severity-Aware DPO)

一种 VLM 幻觉缓解方法。首先使用 GPT-4V 构建的幻觉检测数据集训练幻觉检测模型，然后采用 detect-then-rewrite pipeline 构建 6k 偏好数据，最后使用幻觉严重度感知 (severity-aware) 的 DPO 进行对齐。

## 代表工作
- [[HSA-DPO]]: Xiao et al. (AAAI 2025)
- [[TPR]]: 在 ObjectHal-Bench 上 TPR 取得更低幻觉率 (3.4 vs 5.3)
