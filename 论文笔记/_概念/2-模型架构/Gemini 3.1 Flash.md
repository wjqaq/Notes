---
category: 模型架构
tags: [closed-source, google, gemini, image-editing, vlm]
related_works: ["Entity-Rubrics", "Gemini 3 Flash", "Gemini 2.5 Pro"]
created: 2026-05-18
---

# Gemini 3.1 Flash

Gemini 3.1 Flash（Nano Banana 2）是 Google 的闭源多模态模型，支持图像生成和编辑。

## 在 Entity-Rubrics 中的表现

- Entity-Rubrics 抽象得分: 9.52（所有模型中最高）
- 人类评估指令遵循得分: 9.66
- 在四个领域均表现出色，Physical 领域显著超越 Gemini 3 Pro（9.43 vs 8.46）
- 主要失败模式: 过度编辑（Over-editing）
- 在多样性实验中表现出最高的抽象提示 Vendi Score
