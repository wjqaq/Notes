---
title: "Mixture-of-Experts Discriminator"
created: 2025-05-08
tags: [discriminator, moe, multimodal]
---

# MoE Discriminator

## 定义

混合专家（Mixture-of-Experts）判别器是一种专门化判别器架构，通过多个专家网络分别评估不同方面的质量，然后加权组合最终判别结果。

## 核心思想

在多模态推理任务中，不同类型的错误（如感知错误 vs 推理错误）具有不同的特征，单一判别器难以有效区分。MoE 判别器通过专门化专家提供解耦反馈。

## 典型结构

$$
r(x,y) = \alpha \cdot D_v(x,c) + (1-\alpha) \cdot D_r(x,t)
$$

- **感知专家 $D_v$**: 评估视觉描述质量
- **推理专家 $D_r$**: 评估推理轨迹质量
- **权重 $\alpha$**: 平衡两个专家的贡献

## 应用场景

- [[PRISM]]: 分布对齐阶段的判别器
- 多模态模型训练中的质量评估
- 对抗性训练中的判别器

## 优势

1. **专门化**: 每个专家专注于特定类型的质量评估
2. **解耦反馈**: 为不同错误类型提供针对性信号
3. **灵活性**: 可根据任务需求调整专家数量和类型

## 相关概念

- [[Mixture-of-Experts]]: 基础架构
- [[Adversarial Training]]: 训练范式
- [[On-Policy Distillation]]: 应用场景
