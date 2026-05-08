---
type: concept
aliases: [Visual Inference-Time Intervention, 视觉推理时干预]
---

# V-ITI

## 定义
Visual Inference-Time Intervention，在推理时通过干预注意力层输出来增强视觉信息利用的幻觉缓解方法。

## 核心要点
1. 在特定 Transformer 层对注意力输出施加干预以放大视觉信号
2. 训练无关方法
3. [[LIME]] 中强 baseline，在 POPE 某些子任务（如 Adversarial Acc）上达到最优
4. 仅针对视觉模态

## 代表工作
- [[LIME]]: baseline 对比

## 相关概念
- [[多模态幻觉]]
- [[Inference-time Optimization]]
- [[VCD]]
