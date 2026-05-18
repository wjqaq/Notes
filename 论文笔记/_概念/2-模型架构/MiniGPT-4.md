---
type: concept
aliases: [MiniGPT4, MiniGPT-4]
---

# MiniGPT-4

## 定义
KAUST 提出的轻量级视觉语言模型，仅用一个线性投影层将冻结的 ViT 和 LLM 对齐，以最小成本实现多模态对话。

## 核心要点
1. 最简单的 VL 对齐方案：ViT 输出 -> 单层线性投影 -> LLM
2. 使用 BLIP-2 的预训练 ViT + Q-Former 第一阶段权重，再用高质量图像-文本对做第二阶段的指令微调
3. 证明即使简单的架构配合高质量数据也能产生不错的多模态对话能力
4. Qwen-VL 对比对象之一

## 代表工作
- [[LLaVA]]: 类似简单架构
- [[BLIP-2]]: ViT 权重的来源

## 相关概念
- [[LVLM]]
- [[Instruction Tuning]]
