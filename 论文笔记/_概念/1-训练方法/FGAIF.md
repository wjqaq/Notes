---
category: 1-训练方法
aliases: [FGAIF]
---

# FGAIF (Fine-Grained AI Feedback)

一种 VLM 对齐方法，利用细粒度 AI 反馈生成偏好数据。将响应分解为子片段，对每个片段进行评分，然后构造偏好对用于 DPO 训练。

## 代表工作
- [[FGAIF]]: Jing and Du (2024)
- [[TPR]]: TPR 通过 topic 级选择性替换实现更精细的语义控制
