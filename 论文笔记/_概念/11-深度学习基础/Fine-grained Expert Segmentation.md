---
type: concept
aliases: [Fine-grained Expert Segmentation]
---

# Fine-grained Expert Segmentation

## 定义
将 MoE 中的大专家拆分为更多数量的小专家，增加路由灵活性和专家专业化程度，同时在相同激活参数量下提升模型表达能力。

## 核心要点
1. 将少量大专家替换为大量小专家（如 128 个专家而非 8 个）
2. 每个 token 激活更多专家以保持总激活参数量
3. Qwen3 MoE: 128 专家 / 8 激活，相比传统 8 专家 / 2 激活更灵活
4. 由 DeepSeekMoE 论文推广

## 代表工作
- [[Qwen3]]: 128 专家 / 8 激活
- [[DeepSeek-V3]]: 细粒度 MoE

## 相关概念
- [[Mixture of Experts]]
- [[Global-batch Load Balancing Loss]]
