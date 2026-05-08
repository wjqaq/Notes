---
type: concept
aliases: [推理时优化, inference-time learning, test-time optimization, 推理时学习]
---

# Inference-time Optimization

## 定义
在推理阶段对模型内部表示或输出分布进行基于梯度的优化，而不修改模型参数的一类方法。

## 核心要点
1. 与训练时优化不同：不更新模型权重，仅调整中间表示（如 [[KV Cache]]、hidden states）
2. 通常需要定义推理时目标函数（如 [[Modality Relevance|relevance]] 最大化、约束满足等）
3. 每次推理独立优化，优化状态不跨样本保持
4. 优势：无需训练数据、即插即用、可适配任意冻结模型
5. 代价：额外推理时间（通常数倍减速）

## 代表工作
- [[LIME]]: 每步解码优化 KV cache 以增强模态 relevance
- [[V-ITI]]: Inference-time intervention 方法

## 相关概念
- [[KV Cache]]
- [[Layer-wise Relevance Propagation|LRP]]
