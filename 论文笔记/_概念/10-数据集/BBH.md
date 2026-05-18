---
type: concept
aliases: [BIG-Bench Hard, 大基准难题]
---

# BBH

## 定义
BIG-Bench 数据集中的 23 个难题子集，用于评估 LLM 的高阶推理能力，包含逻辑推理、数学、算法等多方面挑战。

## 核心要点
1. 筛选自 BIG-Bench 中当时模型未能超越人类平均水平的 23 个任务
2. 常用 3-shot + Chain-of-Thought 评估
3. GPT-4 3-shot 达 86.7

## 代表工作
- [[Qwen]]: Qwen-14B 3-shot 达 53.4，超越 LLaMA-13B（45.6）
- [[LLaMA 2]]: 70B 3-shot 达 64.9

## 相关概念
- [[MMLU]]
