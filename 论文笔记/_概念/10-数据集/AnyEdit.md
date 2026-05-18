---
type: concept
aliases: [AnyEdit Benchmark]
---

# AnyEdit

## 定义
大规模图像编辑基准，聚焦于隐式指令编辑（10k 测试样本），覆盖物理/假设性场景，编辑指令为模板化生成。

## 核心要点
1. 10k 样本，所有指令为模板化生成（非自然语言）
2. 主要覆盖物理推理类隐式指令，维持一对一映射（|K| ~ 1）
3. 相比 AbstractEdit，缺乏真正的抽象一对多映射和自然语言指令

## 代表工作
- Yu et al., "AnyEdit: Mastering Unified High-Quality Image Editing for Any Idea", CVPR 2025
- [[Entity-Rubrics]]: 作为对比基准之一

## 相关概念
- [[AbstractEdit]]
- [[SmartEdit]]
- [[EditWorld]]
