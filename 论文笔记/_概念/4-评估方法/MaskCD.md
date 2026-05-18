---
type: concept
aliases: []
---

# MaskCD

## 定义
MaskCD 是一种推理时 LVLM 幻觉抑制方法，通过掩码图像头 (image head) 构造对比解码信号。

## 核心要点
1. 掩码图像头形成对比解码信号，与原始预测对比以减少幻觉
2. LVLM 特定方法，保留原始输入，不使用外部工具
3. 解码开销较高，不如 SIRA 高效
4. 在 POPE/CHAIR 上均表现强劲

## 代表工作
- [[SIRA]]: 对比基线之一，SIRA 在保持更低开销的同时全面超越
