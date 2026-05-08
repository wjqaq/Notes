---
type: concept
aliases: [空间定位, spatial relevance, 空间相关性]
---

# Spatial Grounding

## 定义
衡量模型在生成回答时对感知输入空间位置的依赖精度，通过 [[Layer-wise Relevance Propagation|LRP]] 量化为正确区域 relevance 的集中程度。

## 核心要点
1. 高 spatial grounding 表示模型精确关注感知输入中与问题相关的空间位置
2. 低 spatial grounding 意味着模型可能依赖语言先验而非实际感知证据
3. [[LIME]] 通过数值指标首次量化了不同模型的 spatial grounding 水平
4. LLaVA-1.5: 0.27→0.36, Qwen2-Audio: 0.31→0.57

## 代表工作
- [[LIME]]: 提出 quantitative spatial grounding metric

## 相关概念
- [[Modality Reliance]]
- [[Modality Relevance]]
- [[Layer-wise Relevance Propagation|LRP]]
