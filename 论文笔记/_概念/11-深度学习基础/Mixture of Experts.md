---
type: concept
aliases: [Mixture of Experts, MoE, Sparse MoE]
---

# Mixture of Experts

## 定义
一种神经网络架构，由多个"专家"子网络和一个路由机制组成，每个输入 token 只激活部分专家，从而实现参数规模与计算量的解耦。

## 数学形式
$$y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)$$
其中 $g(x)$ 为 gating/routing 函数，$E_i$ 为第 $i$ 个专家，通常使用 top-k 路由选择激活专家。

## 核心要点
1. 参数总数大但每 token 激活参数少，推理效率高
2. Fine-grained Expert Segmentation: 将大专家拆分为更多小专家以增强专业化
3. Qwen3 MoE: 128 专家 / 8 激活，去除共享专家，使用 global-batch load balancing loss

## 代表工作
- [[Qwen3]]: 30B-A3B 和 235B-A22B 两个 MoE 变体
- [[DeepSeek-V3]]: 671B/37B MoE
- [[Qwen2.5-MoE]]: 前代 MoE 模型

## 相关概念
- [[Fine-grained Expert Segmentation]]
- [[Global-batch Load Balancing Loss]]
- [[Dense Model]]
