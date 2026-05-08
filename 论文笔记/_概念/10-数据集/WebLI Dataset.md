---
type: concept
aliases: [WebLI]
---

# WebLI Dataset

## 定义
Google 内部大规模网络图文数据集，包含 116M 过滤后的图文对，用于 TIPS/TIPSv2 的预训练。

## 核心要点
1. 从网络爬取，含噪声 alt-text
2. TIPSv2 使用 116M 过滤子集（非完整 WebLI）
3. 配合 PaliGemma 和 Gemini Flash 生成合成描述增强
4. 数据集未公开，影响完全复现

## 代表工作
- [[TIPSv2]]: 使用 WebLI 过滤子集进行预训练
- [[TIPS]]: 同样使用 WebLI 数据

## 相关概念
- [[LAION]]
- [[PaliGemma]]
- [[CLIP]]
