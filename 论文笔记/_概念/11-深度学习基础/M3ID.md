---
type: concept
aliases: [M3ID, Multi-Modal Mutual-Information Decoding]
---

# M3ID

## 定义
一种[[对比解码]]变体，通过最小化有图像条件输入和无条件输入之间的互信息损失来抑制[[物体幻觉]]。

## 核心要点
1. 利用图像条件输入和无条件输入的 logit 对比，识别并减少视觉信息损失
2. 与 [[VCD]]/[[ICD]] 同为 CD 类方法
3. 局限：同样存在[[长程衰减]]，且 Cover 分数可能虚高（在生成覆盖面广但精度不高的描述时）

## 代表工作
- (Favero et al., 2024): M3ID 原始提出
- [[FLB]]: 实验中的 baseline 之一

## 相关概念
- [[对比解码]]
- [[VCD]]
- [[ICD]]
- [[FLB]]
- [[长程衰减]]
