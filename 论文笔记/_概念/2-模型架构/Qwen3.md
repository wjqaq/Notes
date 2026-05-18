---
type: concept
aliases: [Qwen3 LLM, Qwen3系列]
---

# Qwen3

## 定义
阿里 Qwen 团队的第三代大语言模型系列，是 [[Qwen3-VL]] 的文本骨干基础，提供 Dense (1.7B/4B/8B/32B) 和 MoE (30B-A3B, 235B-A22B) 多种尺寸。

## 核心要点
1. 支持 Thinking (CoT) 和 Non-thinking 两种推理模式
2. MoE 架构实现高效的推理-质量权衡
3. Qwen3-VL 在 VLM 训练中不仅不损害、甚至增强了 Qwen3 的纯文本能力

## 代表工作
- [[Qwen3-VL]]: 以 Qwen3 为 LLM backbone 的视觉-语言模型
- Yang et al. (2025a): Qwen3 技术报告

## 相关概念
- [[Qwen3-VL]]
- [[Qwen2.5-VL]]
- [[Mixture-of-Experts]]
