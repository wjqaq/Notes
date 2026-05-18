---
type: concept
aliases: [InstructBLIP]
---

# InstructBLIP

## 定义
BLIP-2 的指令微调版本，通过在 Q-Former 中加入指令感知的视觉特征提取，提升视觉语言模型的指令遵循能力。

## 核心要点
1. 在 BLIP-2 基础上，Q-Former 接收指令文本作为额外输入，使视觉特征提取对任务指令敏感
2. 在多种视觉语言任务（VQA、captioning、grounding）上进行指令微调
3. Qwen-VL 的 SFT 阶段在思路上与 InstructBLIP 类似，都是通过指令微调提升交互能力

## 代表工作
- [[BLIP-2]]: 前身
- [[Qwen-VL]]: 对标比较

## 相关概念
- [[Instruction Tuning]]
- [[Q-Former]]
- [[LVLM]]
