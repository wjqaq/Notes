---
category: 生成模型
tags: [image-editing, diffusion-transformer, open-source, thinking-mode]
related_works: ["Entity-Rubrics", "Step1X-Edit"]
created: 2026-05-18
---

# Step1X-Edit

Step1X-Edit 是 Stepfun 开发的开源图像编辑模型，支持 Thinking 推理模式。

## 架构

- **文本编码器**: Qwen-VL
- **图像生成器**: DiT
- **参数**: ~30B
- **Thinking 模式**: Step1X-Think-Reflect（逐步推理 + 反思）

## 在 Entity-Rubrics 中的表现

- Step1X-Edit 抽象得分: 6.55
- Step1X-Think-Reflect 抽象得分: 6.90（Thinking 提升 5.3%）
- Thinking 模式在 Logical 和 Social 领域改进最显著
- 但 Thinking 会引入"精确度税"：在显式提示上反而有轻微退步
