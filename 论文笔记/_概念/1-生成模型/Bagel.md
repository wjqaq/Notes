---
category: 生成模型
tags: [image-editing, unified-multimodal, open-source, thinking-mode, mixture-of-experts]
related_works: ["Entity-Rubrics", "Bagel-MoT", "Step1X-Edit"]
created: 2026-05-18
---

# Bagel

Bagel 是 ByteDance 开发的开源统一多模态模型，支持图像编辑。

## 架构

- **文本编码器**: Bagel-MoT（混合专家文本编码器）
- **图像生成器**: Custom Unified（自定义统一架构）
- **参数**: 14B（7B active，MoE 架构）
- **Thinking 模式**: Bagel-Think

## 在 Entity-Rubrics 中的表现

- Bagel 抽象得分: 4.45（开源模型中最低）
- Bagel-Think 抽象得分: 5.80（Thinking 提升 30.3%，所有模型中相对提升最大）
- Thinking 模式下 Style Transfer 失败率从 34% 降至 9%
- 得分虽低，但人类评估指令遵循得分达 6.61，表明有潜力
