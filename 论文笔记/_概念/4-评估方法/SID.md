---
type: concept
aliases: [Self-Introspective Decoding, 自省解码]
---

# SID

## 定义
SID (Self-Introspective Decoding) 是一种推理时 LVLM 幻觉抑制方法，通过自省机制构造额外的解码参考。

## 核心要点
1. 使用自省信号构造额外参考，与原始预测对比
2. LVLM 特定方法，保留原始输入，不使用外部工具
3. 解码开销较高（需要构造额外参考）
4. 在 ICLR 2025 发表

## 代表工作
- [[SIRA]]: 对比基线之一，SIRA 在保持更低开销的同时超越 SID
