---
aliases: [背侧专家,  dorsal expert]
tags: [concept, vla, architecture, dual-stream]
created: 2026-05-18
---

# Dorsal Expert（背侧专家）

## 定义

VLA 架构中的第二条并行视觉通路，灵感来自生物视觉的背侧通路（dorsal stream），用于处理控制相关的视觉特征（空间布局、物体位姿、场景动力学），从而减轻 VLM 编码器的表征瓶颈。

## 设计要求

由 [[UAM]] 提出的三个关键要求：
1. **生成式初始化**: 由预训练生成式 [[Unified Multimodal Model|UMM]]（如 [[Bagel]]）初始化，自带视觉生成和场景变化的先验
2. **视觉输入**: 接收原始视觉 tokens（VAE 编码），而非可学习查询 tokens
3. **动力学监督**: 辅以视觉动力学损失 $\mathcal{L}_{wm}$（世界模型目标），驱动其进行独立的视觉推理

## 设计空间发现

- 随机初始化 → 无法获得控制所需的视觉结构
- VLM 初始化 → 有改善但语义先验不匹配控制需求
- 生成式初始化（无动力学损失）→ 与 VLM 初始化相当
- **生成式初始化 + 动力学损失** → 唯一同时满足动作性能和语义保留的配置

## 代表工作

- [[UAM]]: 提出 Dorsal Expert 概念和完整设计空间研究
- [[BagelVLA]]: 世界模型专家作为规划信号
