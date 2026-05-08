---
type: concept
aliases: [DINO version 3]
---

# DINOv3

## 定义
DINOv2 的后续版本，纯视觉自监督学习方法，在密集视觉任务上表现极强，但不具备文本对齐能力。

## 核心要点
1. 纯视觉 SSL，不依赖文本监督
2. 密集视觉任务（语义分割、深度估计）达到 SOTA
3. 在 ViT-L 规模下，DINOv3 纯视觉任务略优于 TIPSv2
4. 但在图文检索和 zero-shot 分割上远不如 TIPSv2

## 代表工作
- [[TIPSv2]]: Table 8 详细对比 DINOv3 vs TIPSv2 (ViT-L)
- Siméoni et al. (2025): DINOv3 原始论文

## 相关概念
- [[DINOv2]]
- [[DINO]]
- [[TIPSv2]]
- [[Semantic Segmentation]]
