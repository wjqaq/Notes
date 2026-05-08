---
title: "Tuna-2: Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation"
method_name: "Tuna-2"
authors: [Zhiheng Liu, Weiming Ren, Xiaoke Huang, Shoufa Chen, Tianhong Li, Mengzhao Chen, Yatai Ji, Sen He, Jonas Schult, Belinda Zeng, Tao Xiang, Wenhu Chen, Ping Luo, Luke Zettlemoyer, Yuren Cong]
year: 2025
venue: arXiv
tags: [unified-multimodal-model, encoder-free, pixel-embedding, image-generation, visual-understanding]
zotero_collection: _待整理
image_source: mixed
arxiv_html: https://arxiv.org/html/2604.24763
created: 2025-05-08
---

# 论文笔记：Tuna-2: Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Hong Kong, University of California Santa Barbara, Amazon, University of Washington |
| 日期 | April 2025 |
| 项目主页 | https://tuna-ai.org/tuna-2 |
| 对比基线 | [[Tuna]], [[BAGEL]], [[Qwen2.5-VL]], [[FLUX.1]] |
| 链接 | [arXiv](https://arxiv.org/abs/2604.24763) |

---

## 一句话总结

> Tuna-2 证明完全移除预训练视觉编码器，仅用简单的 patch embedding 层直接处理像素，能在多模态理解和生成任务上达到 SOTA 性能。

---

## 核心贡献

1. **Encoder-Free 架构**: 首次证明预训练视觉编码器对多模态建模不是必需的，简单 patch embedding 即可达到 SOTA
2. **Masking-based Feature Learning**: 提出 masking 方案稳定高维像素空间的训练
3. **全面对比分析**: 通过 Tuna-R（保留编码器）和 Tuna-2（完全移除）的对照实验，揭示两种设计的优劣

---

## 问题背景

### 要解决的问题

现有统一多模态模型（UMM）依赖预训练视觉编码器（如 CLIP、SigLIP）和 VAE tokenizer，架构复杂且存在以下问题：
- 编码器的语义先验可能限制细粒度视觉理解
- 模块化设计增加系统复杂度
- 编码器与 LLM 的对齐需要额外训练阶段

### 现有方法的局限

- **基于编码器的方法**（如 LLaVA、Qwen-VL）：依赖预训练编码器，细粒度任务表现不佳
- **潜在空间生成模型**（如 SD3、FLUX）：需要 VAE 压缩，可能丢失高频细节
- **早期 encoder-free 尝试**（如 Fuyu、Chameleon）：缺乏有效的像素空间训练策略

### 本文的动机

作者认为直接在像素空间建模可以：
1. 避免编码器引入的信息瓶颈
2. 保留完整的高频视觉细节
3. 简化架构，消除 connector 对齐阶段

---

## 方法详解

### 模型架构

Tuna-2 采用 **encoder-free unified multimodal** 架构：
- **输入**: 文本指令 + 图像像素（直接 patch embedding）
- **Backbone**: Qwen2.5-7B-Instruct（LLM decoder）
- **核心模块**: [[Patch Embedding]] + [[Flow Matching Head]]
- **输出**: 文本回复 / 生成图像像素
- **总参数**: 7B

#### 架构演进

论文提出两个变体：
- **Tuna-R**: 移除 VAE，保留 SigLIP 2 编码器
- **Tuna-2**: 完全移除所有编码器，仅用 patch embedding

### 核心模块

#### 模块1: Pixel-Space Patch Embedding

**设计动机**: 直接处理原始像素，避免编码器信息瓶颈

**具体实现**:
- 使用简单的 patch embedding 层将图像划分为 patches
- 每个 patch 直接映射到 embedding 空间
- 无需预训练权重，从头学习视觉表示

#### 模块2: Masking-based Feature Learning

**设计动机**: 高维像素空间训练不稳定，需要正则化

**具体实现**:
- 随机选择图像 patches 用可学习的 mask token 替换
- 对于生成任务：模型预测 masked 和 unmasked 区域的干净图像
- 对于理解任务：masking 作为正则化，防止过拟合

#### 模块3: Flow Matching Head

**设计动机**: 在像素空间进行高质量图像生成

**具体实现**:
- 采用 [[Rectified Flow]] 和线性调度
- 使用 x-prediction 和 v-loss 范式
- 直接预测干净图像，再转换为速度场

---

## 关键公式

### 公式1: [[Rectified Flow|Noisy Sample Construction]]

$$
x_t = t x_1 + (1-t) x_0, \quad t \in [0,1]
$$

**含义**: 使用 rectified flow 和线性调度在像素空间构建噪声样本

**符号说明**:
- $x_t$: 时间步 $t$ 的噪声样本
- $x_1$: 干净图像（源图像）
- $x_0$: 从标准正态分布采样的噪声
- $t$: 时间戳，范围 [0,1]

### 公式2: [[Flow Matching|Clean Image Prediction]]

$$
x_\theta = \pi_\theta(x_t, c, t)
$$

**含义**: 模型直接从噪声图像预测干净图像

**符号说明**:
- $x_\theta$: 预测的干净图像
- $\pi_\theta$: 统一模型（vision-language backbone + flow matching head）
- $x_t$: 噪声样本
- $c$: 条件信号（文本 / 文本+图像）
- $t$: 时间戳

### 公式3: Velocity Transformation

$$
v_\theta = \frac{x_\theta - x_t}{1-t}
$$

**含义**: 将预测的干净图像转换为速度项

**符号说明**:
- $v_\theta$: 预测的速度项
- $x_\theta$: 预测的干净图像
- $x_t$: 噪声样本
- $t$: 时间戳

### 公式4: Flow Matching Loss

$$
\mathcal{L}_{flow} = \mathbb{E}_{t,c,x_1,x_0} \|v_\theta - v\|_2^2
$$

**含义**: 回归预测速度与真实速度的学习目标

**符号说明**:
- $\mathcal{L}_{flow}$: Flow matching 损失
- $v_\theta$: 预测速度
- $v$: 真实速度，定义为 $v = x_1 - x_0$
- $\mathbb{E}$: 对 $t, c, x_1, x_0$ 的期望

---

## 关键图表

### Figure 1: Architecture Evolution

![[Tuna-2_fig1.png|600]]

**说明**: 展示从 Tuna 到 Tuna-R 再到 Tuna-2 的架构演进，逐步移除视觉编码组件。Tuna-2 完全 encoder-free，在多个基准测试上达到 SOTA。

### Figure 2: Qualitative Results

![[Tuna-2_fig2.png|600]]

**说明**: Tuna-2 的高质量文本到图像生成和图像编辑示例，证明 encoder-free 设计不会牺牲生成质量。

### Figure 3: Masking-based Feature Learning

![[Tuna-2_fig3.png|600]]

**说明**: 提出的 masking 方案示意图。训练时用可学习 mask token 替换随机 patches，对生成任务执行 masked prediction，对理解任务作为正则化。

### Figure 4: Visual Tokenizer Comparison

![[Tuna-2_fig4.png|600]]

**说明**: 不同视觉 tokenizer 的定性对比。Tuna-2 的像素级重建质量接近专用 tokenizer FLUX.1 VAE。

### Figure 5: Training Dynamics

![[Tuna-2_fig5.png|600]]

**说明**: 不同理解-生成数据比例下的损失曲线。7:3 比例达到最佳平衡。

### Figure 6: Accuracy vs Training Scale

![[Tuna-2_fig6.png|600]]

**说明**: Tuna-R 和 Tuna-2 随训练规模增长的准确率曲线。Tuna-2 在大规模数据下超越 Tuna-R。

### Figure 7: Attention Map Visualization

![[Tuna-2_fig7.png|600]]

**说明**: 注意力图可视化。Tuna-2 展现更准确的视觉-语言对齐，对误导性语言先验更鲁棒。

### Table 1: Multimodal Understanding Benchmarks

| Models | Size | GQA | RealWorldQA | MMVet | MMMU | MMVP | SEED-Bench2+ | AI2D | ChartQA | OCRBench | V* | CountBench | VisuLogic |
|--------|------|-----|-------------|-------|------|------|--------------|------|---------|----------|-----|------------|-----------|
| LLaVA-1.5 | 7B | 62.0 | 54.8 | 32.9 | 35.7 | - | - | 55.5 | 17.8 | 31.8 | - | - | - |
| Qwen2.5-VL | 7B | 60.7 | 69.9 | 61.7 | 58.6 | 78.0 | 70.5 | 82.7 | 83.0 | 83.7 | 71.2 | 74.1 | 20.0 |
| BAGEL | 14B | 66.4 | 72.8 | 67.2 | 55.3 | 85.0 | 71.9 | 89.2 | 78.5 | 73.3 | 70.2 | 82.5 | 41.7 |
| Tuna | 7B | 63.9 | 66.1 | 42.9 | 49.8 | 70.7 | 52.7 | 79.3 | 85.8 | 74.3 | 52.4 | 73.5 | 22.4 |
| Tuna-R | 7B | 63.5 | 67.9 | 46.7 | 51.1 | 74.7 | 58.4 | 79.4 | 85.6 | 78.3 | 57.6 | 77.8 | 26.2 |
| **Tuna-2** | **7B** | **65.0** | **67.7** | **51.7** | **50.7** | **77.3** | **61.1** | **79.6** | **85.6** | **79.7** | **59.2** | **81.7** | **28.8** |

**说明**: Tuna-2 在 7B 规模的 native UMM 中达到 SOTA，尤其在细粒度任务（CountBench、VisuLogic）上优势明显。

### Table 2: Image Generation Results

| Models | Size | GenEval Overall | DPG-Bench Overall |
|--------|------|-----------------|-------------------|
| SD3-M | 2B | 0.74 | 84.08 |
| FLUX.1 [dev] | 12B | 0.82 | 84.00 |
| BAGEL | 14B | 0.88 | 85.07 |
| Tuna | 7B | 0.90 | 86.76 |
| Tuna-R | 7B | 0.88 | 86.35 |
| **Tuna-2** | **7B** | **0.87** | **86.54** |

**说明**: Tuna-2 在生成任务上与 Tuna-R 持平，证明 encoder-free 设计不会牺牲生成质量。

### Table 3: Image Editing Results

| Models | Add | Adj. | Ext. | Rep. | Rm. | Bg. | Sty. | Hyb. | Act. | Total |
|--------|-----|------|------|------|-----|-----|------|------|------|-------|
| FLUX.1 | 4.25 | 4.15 | 2.35 | 4.56 | 3.57 | 4.26 | 4.57 | 3.68 | 4.63 | 4.00 |
| Qwen-Image | 4.38 | 4.16 | 3.43 | 4.66 | 4.14 | 4.38 | 4.81 | 3.82 | 4.69 | 4.27 |
| GPT-Image | 4.61 | 4.33 | 2.90 | 4.35 | 3.66 | 4.57 | 4.93 | 3.96 | 4.89 | 4.20 |
| Tuna | 4.43 | 4.48 | 2.46 | 4.65 | 4.55 | 4.52 | 4.69 | 4.22 | 4.76 | 4.31 |
| Tuna-R | 4.46 | 4.27 | 2.38 | 4.61 | 4.48 | 4.44 | 4.54 | 4.06 | 4.43 | 4.18 |
| **Tuna-2** | **4.34** | **4.13** | **2.22** | **4.53** | **4.42** | **4.36** | **4.58** | **3.91** | **4.28** | **4.09** |

**说明**: Tuna-2 在图像编辑任务上表现强劲，超越多个专用编辑模型。

### Table 4: Image Reconstruction Performance

| Tokenizer | Res. | rFID↓ | PSNR↑ | SSIM↑ |
|-----------|------|-------|-------|-------|
| SD-VAE | 256 | 1.06 | 28.62 | 0.86 |
| FLUX.1[dev]-VAE | 512 | 0.06 | 33.65 | 0.93 |
| Tuna-R | 512 | 0.12 | 32.22 | 0.93 |
| **Tuna-2** | **512** | **0.15** | **32.80** | **0.93** |

**说明**: Tuna-2 的图像重建质量接近专用 VAE tokenizer，证明像素空间建模的有效性。

### Table 5: Masking Ablation Study

| Models | OCRBench | MMVP | CountBench | GenEval |
|--------|----------|------|------------|---------|
| Tuna | 56.9 | 54.0 | 55.6 | 57.2 |
| Tuna-R w/o Masking | 58.3 | 56.7 | 57.2 | 55.7 |
| Tuna-R w/ Masking | 59.2 | 58.0 | 58.2 | 56.0 |
| Tuna-2 w/o Masking | 55.4 | 52.3 | 53.4 | 47.6 |
| **Tuna-2 w/ Masking** | **56.8** | **55.7** | **57.6** | **48.2** |

**关键发现**: Masking 对 Tuna-2 的提升更显著，尤其在理解任务上。这表明 encoder-free 设计更需要正则化。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Stage 1 Data | 550M pairs | 70% captioning + 30% generation | 预训练 |
| FineVision | 13M | 高质量视觉理解数据 | SFT |
| OmniEdit | 2M | 图像编辑数据 | SFT |

### 实现细节

- **Backbone**: Qwen2.5-7B-Instruct
- **Stage 1**: 550M image-text pairs, 300k steps
- **Stage 2**: 13M FineVision + 2M OmniEdit, 50k steps
- **Tuna-R Encoder**: SigLIP 2 So400M
- **训练无需 connector alignment 阶段**

---

## 批判性思考

### 优点

1. **架构简化**: 完全移除视觉编码器，消除 connector 对齐阶段
2. **细粒度理解强**: 在 CountBench、VisuLogic 等细粒度任务上显著优于 encoder-based 方法
3. **生成质量保持**: 在 GenEval、DPG-Bench 上与 SOTA 持平
4. **理论贡献**: 证明预训练编码器对多模态建模不是必需的

### 局限性

1. **训练成本**: 像素空间训练计算量更大，需要 masking 正则化
2. **生成略逊**: Tuna-R 在生成任务上略优于 Tuna-2，编码器语义先验仍有帮助
3. **缺乏代码**: 论文未提供开源代码，复现困难
4. **规模限制**: 仅在 7B 规模验证，更大规模效果未知

### 潜在改进方向

1. 探索更高效的像素空间训练策略
2. 结合 encoder-free 和 encoder-based 的混合设计
3. 扩展到更大规模（70B+）验证可扩展性
4. 探索视频、3D 等其他模态

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整
- [x] 数据集可获取（公开数据集）

---

## 关联笔记

### 基于

- [[Tuna]]: 前作，使用 VAE + 编码器的统一模型
- [[Rectified Flow]]: 采用的生成框架
- [[Flow Matching]]: 训练范式

### 对比

- [[BAGEL]]: 14B 规模的统一模型，使用编码器
- [[Qwen2.5-VL]]: encoder-based VLM 基线
- [[FLUX.1]]: 专用生成模型，使用 VAE

### 方法相关

- [[Patch Embedding]]: 核心视觉编码方式
- [[Masked Image Modeling]]: 特征学习方案
- [[Unified Multimodal Model]]: 模型类别

---

## 速查卡片

> [!summary] Tuna-2
> - **核心**: Encoder-free 统一多模态模型，直接处理像素
> - **方法**: Patch embedding + Masking-based feature learning + Flow matching
> - **结果**: 7B 规模 SOTA，细粒度理解优势明显
> - **项目主页**: https://tuna-ai.org/tuna-2

---

*笔记创建时间: 2025-05-08*
