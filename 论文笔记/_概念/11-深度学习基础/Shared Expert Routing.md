---
type: concept
aliases: [共享专家路由, shared experts]
---

# Shared Expert Routing

## 定义
MoE 架构中，部分专家被所有 token 共享（始终激活），而其余专家通过路由选择性激活。共享专家捕获通用知识，路由专家处理特定模式。

## 核心要点
1. 源自 DeepSpeed-MoE 的设计
2. 与细粒度专家分割配合使用
3. 提升训练效率和下游任务性能
4. Qwen2.5-Turbo 和 Plus 采用此设计

## 代表工作
- [[Qwen2.5]]: MoE 变体采用共享专家路由
- DeepSpeed-MoE (Rajbhandari et al., 2022): 首次提出

## 相关概念
- [[Mixture of Experts|混合专家模型]]
- [[Fine-grained Expert Segmentation|细粒度专家分割]]
