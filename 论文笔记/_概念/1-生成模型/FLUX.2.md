---
category: 生成模型
tags: [image-editing, diffusion-transformer, open-source]
related_works: ["Entity-Rubrics", "FLUX.1-Kontext"]
created: 2026-05-18
---

# FLUX.2

FLUX.2 是 Black Forest Labs 开发的 FLUX 系列升级版开源图像编辑模型。

## 架构

- **文本编码器**: Mistral + T5（相比 FLUX.1 的 CLIP-only 大幅升级）
- **图像生成器**: Diffusion Transformer (DiT)
- **参数**: ~32B

## 在 Entity-Rubrics 中的表现

- Entity-Rubrics 抽象得分: 7.26（开源模型第二高）
- 升级的文本编码器（Mistral + T5）是性能提升的关键因素
- 在表面级编辑（Texture: 6%, Lighting: 9% 失败率）上接近闭源模型
- 但在结构性任务（Object Count: 41%, Position: 33%）上仍然大幅落后
