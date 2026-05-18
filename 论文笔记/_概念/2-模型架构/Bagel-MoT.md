---
category: 模型架构
tags: [mixture-of-experts, text-encoder, byteDance]
related_works: ["Bagel"]
created: 2026-05-18
---

# Bagel-MoT

Bagel-MoT 是 ByteDance 为 Bagel 模型设计的 Mixture-of-Experts 文本编码器。

## 特点

- 作为 Bagel 统一多模态模型的文本编码器
- 支持 Thinking 推理模式
- Bagel 从 Bagel 到 Bagel-Think 的 30.3% 提升说明 MoT 架构对迭代推理的良好支持

在 Entity-Rubrics 论文中，Bagel 是最低分的开源模型，但 Thinking 模式提升幅度最大。
