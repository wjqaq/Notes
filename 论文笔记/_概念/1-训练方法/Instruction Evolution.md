---
type: concept
aliases: [Instruction Evolution, 指令自进化]
---

# Instruction Evolution

## 定义
一种指令数据增强策略：通过提示 LLM 为现有指令添加约束或要求，使其更复杂、更多样化，从而提升指令数据集的难度覆盖范围。

## 核心要点
1. 使用 Qwen 模型自身对已有指令进行"进化"（增加约束或复杂要求）
2. 确保指令数据集中包含不同难度层次的样本
3. 由 Zhao et al. (2024) 提出

## 代表工作
- [[Qwen2]]: 后训练数据构建中用于丰富指令数据集的难度层次

## 相关概念
- [[InsTag]]
- [[Supervised Fine-Tuning]]
