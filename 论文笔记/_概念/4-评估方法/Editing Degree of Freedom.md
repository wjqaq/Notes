---
type: concept
aliases: [eDoF, 编辑自由度]
---

# Editing Degree of Freedom (eDoF)

## 定义
编辑自由度，度量图像编辑任务中模型解释指令所需的自主权/生成性推测程度，与给定上下文图像和文本指令下所有有效编辑解释集的大小成正比。

## 数学形式

$$\text{eDoF} \propto |K(p|I_c)|$$

## 核心要点
1. eDoF 是连续谱系而非离散类别：从一对一映射（Explicit/Implicit 编辑，|K|~1）到一对多映射（Abstract 编辑，|K|>>1）
2. eDoF 是 image-text pair 的函数而非仅文本：同一指令在不同上下文图像中的 eDoF 可以完全不同
3. 由两个正交轴决定：Identification（编辑什么）和 Specificity（怎么编辑）
4. 在 Abstract 编辑中，两轴至少一个模糊，导致高 eDoF

## 代表工作
- [[Entity-Rubrics]]: 首次形式化 eDoF 概念用于抽象图像编辑评估

## 相关概念
- [[Abstract Image Editing]]
- [[Entity-Rubrics]]
