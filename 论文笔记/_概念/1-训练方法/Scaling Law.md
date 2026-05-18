---
type: concept
aliases: [缩放定律, scaling laws]
---

# Scaling Law

## 定义
描述模型性能（损失）与模型参数量、训练数据量、计算量之间幂律关系的经验规律，用于指导模型超参数和规模选择。

## 数学形式

$$L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_{\infty}$$

其中 $N$ 为参数量，$D$ 为数据量，$L_{\infty}$ 为不可约损失。

## 核心要点
1. 性能随模型规模和数据量呈幂律改善
2. 可用于预测最优批大小和学习率
3. 指导 MoE 与 Dense 模型的性能对比
4. Chinchilla 缩放定律建议训练 token 数约为参数量的 20 倍

## 代表工作
- [[Qwen2.5]]: 利用缩放定律确定不同规模模型的最优超参数和 MoE 配置
- [[Qwen2]]: Qwen 系列的预训练数据从 7T 扩展到 18T

## 相关概念
- [[Pre-training]]
- [[Mixture of Experts|MoE]]
