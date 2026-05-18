---
type: concept
aliases: [BLIP2, BLIP-2]
---

# BLIP-2

## 定义
Salesforce 提出的视觉语言模型，使用 Q-Former 将冻结的视觉编码器和 LLM 对齐，是高效 VL 对齐的代表工作。

## 数学形式
Q-Former: $H = \text{CrossAttn}(Q_{\text{learnable}}, H_{\text{ViT}})$，再用投影层连接 LLM。

## 核心要点
1. Q-Former 是核心创新：一组可学习 query 通过 Cross-Attention 从 ViT 特征中提取与文本相关的视觉信息
2. 两阶段训练：视觉-语言表示学习 + 视觉-语言生成学习
3. Qwen-VL 的 VL Adapter（单层 Cross-Attention + 可学习 query）可视为 Q-Former 的简化版本

## 代表工作
- [[Qwen-VL]]: Adapter 灵感部分来源于此
- [[InstructBLIP]]: BLIP-2 + 指令微调

## 相关概念
- [[Cross-Attention]]
- [[Q-Former]]
- [[LVLM]]
