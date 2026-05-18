---
category: 生成模型
tags: [image-editing, diffusion-transformer, open-source]
related_works: ["Entity-Rubrics"]
created: 2026-05-18
---

# HiDream-E1

HiDream-E1 是 HiDream.ai 开发的开源图像生成基础模型，基于稀疏 Diffusion Transformer。

## 架构

- **文本编码器**: T5 + CLIPs + Llama
- **图像生成器**: DiT (Sparse)
- **参数**: ~25B

## 在 Entity-Rubrics 中的表现

- Entity-Rubrics 抽象得分: 5.38
- 表现不稳定：在 Emotional 领域较高（7.91），但 Logical 领域极低（4.18）
- 显式指令下存在严重的过度编辑问题（Figure 19 展示了典型例子）
