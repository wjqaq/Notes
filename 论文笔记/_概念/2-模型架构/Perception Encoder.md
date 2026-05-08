---
type: concept
aliases: [PE, PE-core, Perception Encoder core]
---

# Perception Encoder

## 定义
Google DeepMind 的视觉编码器系列，侧重图文对齐，但在密集任务上表现不足。TIPSv2 在参数更少、训练数据更少的情况下仍能超越。

## 核心要点
1. 强调视觉-语言的全局对齐
2. PE-core G/14 有 56% 更多参数和 47x 更多训练数据
3. TIPSv2-g 在 3/5 共享评估上超越 PE-core G/14
4. 局限：密集视觉任务（分割、深度）表现不如 TIPSv2

## 代表工作
- [[TIPSv2]]: 在多数 shared evals 上超越 PE-core
- Bolya et al. (2025): Perception Encoder 论文

## 相关概念
- [[CLIP]]
- [[Contrastive Learning]]
- [[TIPSv2]]
