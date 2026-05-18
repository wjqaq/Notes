---
type: concept
aliases: []
---

# Entity-Rubrics

## 定义
Entity-Rubrics 是一个 VLM 驱动的细粒度自动评估框架，通过将抽象图像编辑分解为实体级别的原子评估单元，实现对抽象指令遵循的精确、可解释评估。

## 数学形式
三阶段流程：
1. Entity Detection: 识别 Things/Stuff/Global 实体
2. Entity Ranking: Expected Transformation + Execution Alignment
3. Final Scoring: 聚合为 1-10 分评估

## 核心要点
1. 受 NLP 原子事实评估启发，将图像实体视为评估的原子单元
2. Spearman's rho = 0.66 与人类判断相关性，优于 VIEScore (0.54) 和 CLIP (0.41)
3. 开源模型偏向编辑不足（Under-Editing），闭源模型偏向过度编辑（Over-Editing）
4. 可扩展到奖励模型和测试时 critique-and-revise 循环

## 代表工作
- [[Entity-Rubrics|Editor's Choice]]: Ventura et al., arXiv 2026

## 相关概念
- [[Editing Degree of Freedom]]
- [[Atomic Facts Evaluation]]
- [[Abstract Image Editing]]
- [[VLM Judge]]
