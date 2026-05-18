---
type: concept
aliases: [Emu Edit, EMU-Edit]
---

# EMU-Edit

## 定义
Meta 发布的精确图像编辑模型和基准，通过识别和生成任务的统一框架实现图像编辑。包含 440 个全局编辑测试样本。

## 核心要点
1. 440 个全局编辑样本，中等领域多样性（3-4 个域）
2. 使用自然语言指令（非模板化）
3. 主要覆盖全局变换（如"将场景变为夜晚"），是一对多映射但领域范围有限
4. 相比 AbstractEdit，其 eDoF 和覆盖领域较少

## 代表工作
- Sheynin et al., "Emu Edit: Precise Image Editing via Recognition and Generation Tasks", CVPR 2024
- [[Entity-Rubrics]]: 作为对比基准之一

## 相关概念
- [[InstructPix2Pix]]
- [[AbstractEdit]]
