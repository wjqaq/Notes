---
title: "Mixture-of-Experts"
created: 2025-05-08
tags: [architecture, moe, scaling]
---

# Mixture-of-Experts (MoE)

## 定义

混合专家（Mixture-of-Experts, MoE）是一种模型架构，通过多个专门化的"专家"网络和路由机制，根据输入动态选择激活哪些专家。

## 核心思想

$$
y = \sum_{i=1}^n G(x)_i \cdot E_i(x)
$$

- **专家 $E_i$**: 专门化的子网络
- **门控 $G(x)$**: 路由函数，决定每个专家的权重
- **稀疏激活**: 通常只激活 top-k 个专家

## 路由策略

- **Top-k 路由**: 选择得分最高的 k 个专家
- **负载均衡**: 防止某些专家过载
- **专家容量**: 限制每个专家处理的样本数

## 应用场景

- [[PRISM]]: MoE 判别器架构
- 大语言模型（如 Mixtral, DeepSeek-MoE）
- 多任务学习
- 大规模模型扩展

## 优势

1. **参数效率**: 增加参数但不增加计算量
2. **专门化**: 不同专家学习不同模式
3. **可扩展**: 易于扩展到更大规模

## 挑战

1. **训练不稳定**: 需要负载均衡
2. **内存占用**: 所有专家都需加载
3. **路由学习**: 需要学习好的路由策略

## 相关概念

- [[MoE Discriminator]]: PRISM 中的应用
- [[Sparse Activation]]: 稀疏激活
- [[Model Scaling]]: 模型扩展
