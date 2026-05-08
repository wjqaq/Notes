---
type: concept
aliases: [分类 token, classification token]
---

# CLS Token

## 定义
ViT 中附加在 patch 序列前的可学习 token，其最终输出用作全局图像表征。

## 核心要点
1. 与 patch token 一起参与自注意力，最终读出作为分类 / 对齐特征
2. 实际上通过注意力从 patch token 聚合信息，易出现[[Coarse-grained Semantic Supervision|粗粒度监督]]下的背景偏置
3. 可被 [[Global Average Pooling]] 替代

## 代表工作
- [[LaSt-ViT]]: 提出 / 使用该概念的代表工作

## 相关概念
- [[Vision Transformer]]
- [[Patch Score]]
- [[Register Tokens]]
