---
type: concept
aliases: [InsTag, 指令标签]
---

# InsTag

## 定义
一种开集细粒度指令标注工具，用于从大规模指令数据集中自动提取语义本体（ontology）。

## 核心要点
1. 对每条指令自动标注多个细粒度标签
2. 基于标签多样性、语义丰富度、复杂度和意图完整性评估指令
3. 通过 InsTag 标注后可选择代表性指令子集进行后续精炼
4. 由 Lu et al. (2024c) 在 ICLR 发表

## 代表工作
- [[Qwen2]]: 后训练数据构建中用于自动本体提取和指令选择

## 相关概念
- [[Instruction Tuning]]
- [[Supervised Fine-Tuning]]
