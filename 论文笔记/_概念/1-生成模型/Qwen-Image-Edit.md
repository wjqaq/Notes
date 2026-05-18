---
type: concept
aliases: [Qwen-Image-Edit, Qwen Image]
---

# Qwen-Image-Edit

## 定义
阿里通义千问团队的图像编辑模型，基于 Qwen-VL-2.5 多模态骨干，在 Entity-Rubrics 论文 AbstractEdit 基准测试中为最佳开源模型（得分 7.48/10）。

## 核心要点
1. 开源模型中抽象编辑表现最佳：7.48 分（远超第二名 FLUX.2 的 7.26）
2. 在 Emotional 和 Social 域表现优异（8.00 和 8.14），在 Logical 域较弱（6.89）
3. 仅提供显式编辑指令的评估结果（无抽象编辑人类评分）
4. 利用 Qwen-VL-2.5 多模态骨干作为文本编码器

## 代表工作
- Wu et al., "Qwen-Image Technical Report", arXiv 2025
- [[Entity-Rubrics]]: 在 AbstractEdit 基准中作为最佳开源模型评估

## 相关概念
- [[FLUX.2]]
- [[Entity-Rubrics]]
- [[Abstract Image Editing]]
