---
aliases: []
tags: [concept, vla, model, co-training]
created: 2026-05-18
---

# ChatVLA

统一多模态理解和机器人控制的 VLA 模型 [Zhou et al., 2025]，使用 VQA 和动作数据联合训练。

- 论文: [ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model](https://arxiv.org/abs/...) (2025, EMNLP)
- 特点: VL co-training 部分保留 VLM 能力（MMMU=37.4）
- 局限: co-training 数据集和 QA-动作目标差距限制通用理解保留
- 关系: [[UAM]] 的对比基线

## 代表工作

- [[ChatVLA]]
- [[UAM]]: 对比 co-training vs 架构分离策略
