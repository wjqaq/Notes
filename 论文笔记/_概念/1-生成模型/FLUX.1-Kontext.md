---
category: 生成模型
tags: [image-editing, diffusion-transformer, open-source]
related_works: ["Entity-Rubrics", "FLUX.2", "CLIP"]
created: 2026-05-18
---

# FLUX.1-Kontext

FLUX.1-Kontext 是 Black Forest Labs 开发的开源图像编辑模型，基于 Flow Matching 和 DiT 架构。

## 架构

- **文本编码器**: CLIP + T5
- **图像生成器**: Diffusion Transformer (DiT)
- **参数**: ~12B
- **特点**: 支持 in-context 图像编辑（在潜空间中执行流匹配）

## 在 Entity-Rubrics 中的表现

- Entity-Rubrics 抽象得分: 5.10（开源模型中排名较低）
- 主要失败模式: 编辑不足（Under-editing）
- CLIP-only 文本编码器限制了其对抽象指令的理解能力
