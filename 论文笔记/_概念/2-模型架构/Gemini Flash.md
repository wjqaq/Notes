---
type: concept
aliases: [Gemini 1.5 Flash]
---

# Gemini Flash

## 定义
Google 的多模态大模型 Gemini 1.5 的轻量版，在 TIPSv2 中用于生成高粒度合成图像描述。

## 核心要点
1. 基于 (图像 + alt-text + PaliGemma 描述) 生成丰富全面的描述
2. 捕捉 PaliGemma 遗漏的细节：姿态、卡通属性、季节背景、物体关系
3. 局限性：描述过于全面会弱化对比学习难度
4. TIPSv2 通过交替使用 PaliGemma 和 Gemini 描述解决此问题

## 代表工作
- [[TIPSv2]]: 利用 Gemini Flash 做多粒度文本增强
- Gemini Team (2024): Gemini 1.5 技术报告

## 相关概念
- [[PaliGemma]]
- [[Multi-Granularity Caption Sampling]]
