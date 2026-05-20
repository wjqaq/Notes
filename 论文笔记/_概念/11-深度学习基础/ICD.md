---
type: concept
aliases: [Instruction Contrastive Decoding, 指令对比解码]
---

# ICD

## 定义
Instruction Contrastive Decoding，通过对比有指令和无指令条件下的 logit 分布来减少幻觉的推理时解码方法。

## 核心要点
1. 利用指令扰动来识别语言先验主导的 token
2. 训练无关方法
3. [[LIME]] baseline 之一，但在某些场景（如 CHAIR）可能降低性能

## 代表工作
- [[LIME]]: baseline 对比
- [[FLB]]: 指出 ICD 存在[[长程衰减]]问题

## 相关概念
- [[多模态幻觉]]
- [[VCD]]
- [[FLB]]
- [[长程衰减]]
