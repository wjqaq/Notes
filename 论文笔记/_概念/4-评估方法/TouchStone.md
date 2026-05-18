---
type: concept
aliases: [TouchStone]
---

# TouchStone

## 定义
阿里 Qwen 团队提出的用于评估视觉语言模型指令遵循能力的开放域基准，使用 GPT-4 对模型输出进行评分。

## 核心要点
1. 覆盖英文和中文两种语言的指令遵循评估
2. 使用 GPT-4 作为评判者对模型回答打分，关注理解、识别、推理等多个子维度
3. Qwen-VL-Chat 英文 645.2 分、中文 401.2 分，大幅领先同期模型（中文是 VisualGLM 的 1.6 倍）
4. 中文评估是 TouchStone 的独特价值，填补了 LVLM 中文指令遵循评估的空白

## 代表工作
- [[Qwen-VL]]: 在 TouchStone 上获最佳成绩
- [[SEED-Bench]]: 另一 LVLM 评估基准
- [[MME]]: 感知+认知评估

## 相关概念
- [[LVLM]]
- [[Instruction Tuning]]
