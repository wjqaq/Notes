---
type: concept
aliases: [自指令, Self Instruction]
---

# Self-Instruct

## 定义
一种利用 LLM 自身 In-Context Learning 能力生成指令-回复对的方法，通过迭代自我生成、筛选和训练，逐步提升模型的指令遵循能力。

## 核心流程
1. 用少量人工示例作为种子，让 LLM 生成新的指令和回复
2. 用规则和人工标注筛选高质量样本
3. 将筛选后的样本加入训练集，微调模型
4. 迭代多次直到获得足够多高质量样本

## 核心要点
1. 大幅降低人工标注成本
2. 可针对特定格式（如 ReAct）生成格式化的训练数据
3. 迭代过程可能遇到质量退化，需要始终保留人工筛选环节

## 代表工作
- [[Qwen]]: 用于 Agent 能力训练，生成约 2000 条高质量 ReAct 格式样本
- [[Alpaca]]: 使用 Self-Instruct 从 GPT-3.5 生成指令数据

## 相关概念
- [[Supervised Fine-Tuning]]
- [[ReAct Prompting]]
