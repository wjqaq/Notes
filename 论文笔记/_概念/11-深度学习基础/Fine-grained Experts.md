---
type: concept
aliases: [Fine-grained Experts, 细粒度专家]
---

# Fine-grained Experts

## 定义
MoE 架构中的一种设计策略：将专家 FFN 拆分为数量更多、尺寸更小的专家单元，同时增加每次激活的专家数量。

## 核心要点
1. 相比传统 MoE（如 Mixtral 8x7B 中 8 个专家激活 2 个），细粒度专家使用更多更小的专家
2. Qwen2 MoE: 64 个路由专家 + 8 个共享专家，每 token 激活 8 个
3. 在同等总参数量和激活参数量下，提供更丰富的专家组合可能性
4. 由 DeepSeekMoE (Dai et al., 2024) 推广

## 代表工作
- [[Qwen2]]: 64 + 8 专家配置，每个细粒度专家从密集模型复制并打乱后初始化

## 相关概念
- [[Mixture-of-Experts]]
- [[Shared Expert Routing]]
- [[Upcycling]]
