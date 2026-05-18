---
type: concept
aliases: [Gemini 3, Gemini Flash]
---

# Gemini 3 Flash

## 定义
Google 的多模态模型（VLM），在 Entity-Rubrics 论文中使用作评估管线的基础 VLM，也作为被评估的图像编辑模型参与基准测试。

## 核心要点
1. 在 Entity-Rubrics 评估流程中作为评估 VLM：处理两步评估调用（上下文图像分析 -> 编辑后对比评估）
2. 也作为被评估模型：Gemini 3.1 Flash 在 AbstractEdit 测试中得分最高（9.52/10，人类评分 9.66）
3. Gemini 3 Pro 得分 9.27，略低于 Flash 版本但在 Logical 域表现最佳

## 代表工作
- [[Entity-Rubrics]]: 评估管线的核心 VLM 引擎

## 相关概念
- [[Gemini 2.5 Pro]]
- [[Entity-Rubrics]]
- [[VLM Judge]]
