---
type: concept
aliases: [register tokens, 寄存器 token]
---

# Register Tokens

## 定义
Darcet et al. ICLR 2024 提出的附加 token，用于吸收 ViT 训练中出现的高范数 outlier，避免其污染 patch 特征。

## 核心要点
1. 在 patch 序列中插入若干个不参与最终预测的可学习 token
2. 能缓解高范数 token 问题，但 [[LaSt-ViT]] 指出 artifacts 根因在于[[Coarse-grained Semantic Supervision|惰性聚合]]而非高范数本身
3. 常与 [[DINOv2]] 组合使用

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[Vision Transformer]]
- [[LaSt-ViT]]
- [[DINOv2]]
