---
type: concept
aliases: [ViT, 视觉Transformer]
---

# Vision Transformer

## 定义
将图像切分为 patch 后经 Transformer Encoder 处理的视觉基础模型。

## 核心要点
1. 将 224×224 图像切成 16×16 patch，线性投影为 token 序列
2. 用 [[CLS Token]] 或 [[Global Average Pooling]] 聚合全局表征
3. 全局自注意力使感受野覆盖整图，但也带来背景 shortcut 等 artifacts

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[CLS Token]]
- [[Patch Embedding]]
- [[Register Tokens]]
- [[DINOv2]]
