---
type: concept
aliases: [Memory-based Visual Rectification, 记忆视觉矫正]
---

# MemVR

## 定义
Memory-based Visual Rectification，利用外部视觉记忆检索来纠正多模态模型幻觉的推理时方法。

## 核心要点
1. 维护视觉记忆库，在推理时检索相似图像进行信息矫正
2. 训练无关，即插即用
3. [[LIME]] 中强 baseline，在多个 benchmark 上表现接近最优
4. 依赖外部记忆库的构建和维护

## 代表工作
- [[LIME]]: baseline 对比

## 相关概念
- [[多模态幻觉]]
- [[VCD]]
- [[V-ITI]]
