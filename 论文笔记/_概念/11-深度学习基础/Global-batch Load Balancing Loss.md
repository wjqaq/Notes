---
type: concept
aliases: [Global-batch Load Balancing Loss, Global Batch Load Balance]
---

# Global-batch Load Balancing Loss

## 定义
在整个 global batch 上计算 MoE 中专家的负载均衡损失，鼓励 token 均匀分配，促进专家专业化，避免部分专家过载或闲置。由 Qwen3 团队改进并采用。

## 数学形式
$$\mathcal{L}_{balance} = \alpha \cdot \sum_{i=1}^{N} f_i \cdot P_i$$
其中 $f_i$ 为专家 $i$ 被路由到的 token 比例，$P_i$ 为专家 $i$ 的 gating 概率均值，$\alpha$ 为权重。

## 核心要点
1. 在 global batch 粒度而非 micro batch 粒度计算，统计更稳定
2. Qwen3 MoE 去除共享专家后依赖此损失确保所有专家被激活
3. 相比传统辅助损失，在高负载不均衡场景下更有效

## 代表工作
- [[Qwen3]]: MoE 训练中使用，128 专家 / 8 激活

## 相关概念
- [[Mixture of Experts]]
- [[Fine-grained Expert Segmentation]]
