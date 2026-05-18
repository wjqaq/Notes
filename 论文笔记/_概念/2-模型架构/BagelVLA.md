---
aliases: []
tags: [concept, vla, model, world-model]
created: 2026-05-18
---

# BagelVLA

交织语言规划、视觉预测和动作生成的 VLA 模型 [Hu et al., 2026 (RSS)]。

- 论文: BagelVLA: Enhancing Long-Horizon Manipulation via Interleaved Vision-Language-Action Generation (RSS 2026)
- 特点: 单步去噪机制，双流 Flow Matching 框架
- 关系: [[UAM]] 直接继承其单步去噪和输入格式设计

## 代表工作

- [[BagelVLA]]: 原始提出
- [[UAM]]: 简化 BagelVLA 输入格式，丢弃大规模预训练，直接训练于动作数据
