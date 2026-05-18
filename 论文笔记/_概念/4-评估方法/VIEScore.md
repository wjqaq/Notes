---
type: concept
aliases: [VIE, Visual Instruction-following Evaluation]
---

# VIEScore

## 定义
一种基于 VLM 的可解释条件图像合成评估方法，提供全局评分和简短说明，但缺乏实体级细粒度诊断。

## 核心要点
1. 基于 VLM 的零样本评估，在条件图像合成评估中取得较好结果
2. 局限性：仅提供全局评分和简短的文字解释，无法进行实体级诊断
3. Entity-Rubrics 论文中 Spearman's rho=0.54 与人类判断的相关性，低于 Entity-Rubrics 的 0.66

## 代表工作
- Ku et al., "VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation", ACL 2024

## 相关概念
- [[Entity-Rubrics]]
- [[VLM Judge]]
- [[CLIP]]
