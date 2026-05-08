---
type: concept
aliases: [Text-Image Pretraining with Spatial Awareness]
---

# TIPS

## 定义
Text-Image Pretraining with Spatial awareness，结合 CLIP 对比学习、DINO 自蒸馏和 iBOT 遮蔽图像建模的视觉-语言预训练方法，是 TIPSv2 的前身（ICLR 2025）。

## 数学形式
$$\mathcal{L}_{TIPS} = \mathcal{L}_{CLIP} + \alpha\mathcal{L}_{DINO} + \beta\mathcal{L}_{iBOT}$$

## 核心要点
1. 首次将对比学习、自蒸馏、遮蔽图像建模三合一用于 VLP
2. 使用双 CLS token 分别对齐 web alt-text 和 PaliGemma 合成描述
3. 在 zero-shot 分割和密集预测上超越单纯的对比学习模型
4. 局限：iBOT 损失仅对遮挡 token 计算，大量可见 token 未被利用

## 代表工作
- [[TIPSv2]]: 升级 iBOT 到 iBOT++，加入 head-only EMA 和多粒度文本
- Maninis et al. (ICLR 2025): 原始 TIPS 论文

## 相关概念
- [[CLIP]]
- [[DINO]]
- [[iBOT]]
- [[iBOT++]]
- [[TIPSv2]]
