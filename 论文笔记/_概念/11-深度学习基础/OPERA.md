---
type: concept
aliases: [OPERA decoding, Over-trust Penalty and Retrospection-Allocation]
---

# OPERA

## 定义
基于注意力图的启发式解码策略，通过对过度信任的 token 施加惩罚和回溯重分配来缓解多模态幻觉。

## 核心要点
1. 利用自注意力图中的"知识聚合模式"识别幻觉倾向 token
2. 对过度依赖少量 token 的预测施加惩罚
3. 训练无关，但仅基于注意力启发式，缺乏理论归因基础
4. [[LIME]] baseline 之一，在 POPE 和 CHAIR 上表现中等

## 代表工作
- (Huang et al., 2024): OPERA 原始提出
- [[LIME]]: baseline 对比

## 相关概念
- [[多模态幻觉]]
- [[VCD]]
