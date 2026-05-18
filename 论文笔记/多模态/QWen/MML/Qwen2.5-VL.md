---
title: "Qwen2.5-VL Technical Report"
method_name: "Qwen2.5-VL"
authors: [Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, Junyang Lin]
year: 2025
venue: arXiv
tags: [vision-language-model, document-parsing, visual-grounding, video-understanding, gui-agent]
zotero_collection: 多模态/QWen/MML
image_source: mixed
arxiv_html: https://arxiv.org/html/2502.13923
created: 2025-05-18
---

# 论文笔记：Qwen2.5-VL Technical Report

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Alibaba Group, Qwen Team |
| 日期 | February 2025 |
| 项目主页 | https://chat.qwenlm.ai |
| 对比基线 | [[GPT-4o]], [[Claude 3.5 Sonnet]], [[InternVL2|InternVL2.5-78B]], [[Gemini 1.5 Pro]] |
| 链接 | [arXiv](https://arxiv.org/abs/2502.13923) / [GitHub](https://github.com/QwenLM/Qwen2.5-VL) / [HF](https://huggingface.co/Qwen) |

---

## 一句话总结

> Qwen2.5-VL 在原生动态分辨率 ViT、绝对时间 MRoPE 和窗口注意力三项架构创新的基础上，通过 4.1T tokens 预训练实现文档全解析、精确目标定位和小时级长视频理解，72B 版本在文档/图表理解上匹配 GPT-4o 和 Claude 3.5 Sonnet。

---

## 核心贡献

1. **高效视觉编码器**: 在 [[Vision Transformer|ViT]] 中引入 [[Window Attention]]（仅4层全注意力），使计算从 $O(n^2)$ 降至线性，支持原生分辨率输入。
2. **时空动态处理**: 在空间维度使用 [[Native Dynamic Resolution]]，在时间维度引入 [[Dynamic FPS Sampling]]，实现变尺寸图像和变帧率视频的统一处理。
3. **绝对时间对齐的 MRoPE**: 将 [[MRoPE]] 的时间维度从帧索引改为 [[Absolute Time Encoding|绝对时间戳]]，使模型感知事件节奏和精确时刻定位。
4. **大规模高质量数据**: 预训练数据从 1.2T 扩展到 4.1T tokens，涵盖图文交错、定位、文档解析、视频、Agent 等多模态数据。
5. **三项全能**: 文档全解析（[[Document Omni-Parsing]]）+ 精确目标定位（[[Visual Grounding|Box+Point Grounding]]）+ 超长视频理解（[[Long Video Understanding]]），且保持纯文本能力不退化。

---

## 问题背景

### 要解决的问题

现有 [[LVLM|大规模视觉语言模型]] 虽然在通用任务上表现良好，但在三个关键维度存在瓶颈：
- **细粒度视觉感知不足**: 无法精确描述对象位置（点/框级别）
- **文档理解碎片化**: 文本提取、图表解析、版面分析依赖分离的专用模型
- **长视频建模薄弱**: 难以处理小时级视频，时间定位精度差

### 现有方法的局限

- [[Qwen2-VL]] 的 MRoPE 时间 ID 绑定到帧序号，无法表达视频内容的真实节奏变化
- 大多数 VLM 使用归一化坐标，丢失了物体的真实尺度和空间关系
- 传统文档解析需要多模型流水线（版面分析 + OCR + 图表理解），效率低且错误累积
- 视觉编码器使用全局自注意力，处理高分辨率图像时计算复杂度 $O(n^2)$ 过高

### 本文的动机

"当前多模态大模型的能力如同夹心饼干的中间层——胜任各类任务但缺乏卓越表现。" Qwen2.5-VL 致力于从底层夯实细粒度感知能力，为 [[LVLM]] 构建稳固根基，上层则通过 [[Qwen2.5 LLM]] 增强多模态推理。

---

## 方法详解

### 模型架构

Qwen2.5-VL 采用经典的三组件架构，提供 3B / 7B / 72B 三个规模：

- **Large Language Model**: 基于 [[Qwen2.5 LLM]] 预训练权重初始化，将 1D [[RoPE]] 替换为 **MRoPE（对齐绝对时间）**
- **Vision Encoder**: 全新设计的 [[Vision Transformer|ViT]]（从头训练），使用 [[2D-RoPE]] + [[Window Attention]] + [[SwiGLU]] + [[RMSNorm]]
- **Vision-Language Merger**: 两层 [[MLP]]，将相邻 4 个 patch 特征拼接压缩后投影到 LLM 文本嵌入维度

| 配置项 | Qwen2.5-VL-3B | Qwen2.5-VL-7B | Qwen2.5-VL-72B |
|--------|--------------|--------------|----------------|
| ViT Hidden Size | 1280 | 1280 | 1280 |
| ViT Layers | 32 | 32 | 32 |
| ViT Num Heads | 16 | 16 | 16 |
| ViT Patch Size | 14 | 14 | 14 |
| ViT Window Size | 112 | 112 | 112 |
| ViT Full Attn Blocks | {7,15,23,31} | {7,15,23,31} | {7,15,23,31} |
| Merger In/Out | 1280/2048 | 1280/3584 | 1280/8192 |
| LLM Hidden Size | 2048 | 3584 | 8192 |
| LLM Layers | 36 | 28 | 80 |
| LLM KV Heads | 2 | 4 | 8 |
| Vocab Size | 151646 | 151646 | 151646 |
| Trained Tokens | 4.1T | 4.1T | 4.1T |

**关键设计**: 三尺寸共享相同的 ViT，仅 LLM 规模不同，体现 [[Vision Transformer|ViT]] 在跨尺度迁移中的高效性。

### 核心模块

#### 模块1: 快速高效的视觉编码器

**设计动机**: 原生分辨率输入导致自注意力计算量随图像尺寸呈二次增长，需线性化计算同时保持分辨率保真度。

**具体实现**:
- 在 28/32 层中使用 [[Window Attention]]（最大窗口 112x112 = 8x8 patches），仅 4 层（index 7, 15, 23, 31）使用全注意力
- 小于 112x112 的区域不填充，保留原始分辨率
- 位置编码采用 [[2D-RoPE]] 捕获二维空间关系
- 视频输入使用 3D patch 划分：两连续帧分组，减少进入 LLM 的 token 数
- 架构对齐 LLM 设计：[[RMSNorm]] 归一化 + [[SwiGLU]] 激活函数
- 训练：CLIP 预训练 -> 视觉-语言对齐 -> 端到端微调，输入按原始宽高比随机采样

#### 模块2: 原生动态分辨率与帧率

**设计动机**: 传统归一化坐标丢失尺度信息，固定帧率无法适应不同节奏的视频内容。

**具体实现**:
- **空间**: 图像按实际尺寸动态映射为不同长度的 token 序列，边界框和点坐标直接使用实际图像尺寸
- **时间**: [[Dynamic FPS Sampling]] 训练，使模型适应不同帧率的视频
- 直接将 MRoPE 的时间 ID 对齐 [[Absolute Time Encoding|绝对时间戳]]，通过时间维度 ID 间的间隔学习时间节奏

#### 模块3: 绝对时间对齐的 MRoPE

**设计动机**: [[Qwen2-VL]] 的 [[MRoPE]] 将时间位置 ID 绑定到帧序号，无法反映内容变化速度或绝对事件时刻。

**具体实现**:
- MRoPE 分解位置嵌入为三维：$(\text{temporal}, \text{height}, \text{width})$
- 文本输入：三维分量使用相同 ID，等价于 1D [[RoPE]]
- 图像输入：时间 ID 恒定，高/宽 ID 按空间位置分配
- 视频输入：时间 ID 按帧递增，但改进为使用实际时间戳间隔
- 关键创新：**通过时间 ID 之间的间隔（而非绝对序号）让模型感知时间节奏**，无需额外计算开销

---

## 关键公式

### 公式1: [[Window Attention|窗口注意力]] 的计算复杂度

$$
\text{Complexity}_{\text{window}} = O(H \cdot W \cdot w^2) \quad \text{vs} \quad \text{Complexity}_{\text{full}} = O((H \cdot W)^2)
$$

**含义**: 将全局自注意力的二次复杂度降低为线性（对 patch 数量），使原生分辨率处理变为可行。

**符号说明**:
- $H, W$: 图像的高和宽（以 patch 计）
- $w$: 窗口大小（最大 8 = 112/14）

### 公式2: [[MRoPE|多模态旋转位置编码]] 的分解

$$
\Theta = [\theta_t, \theta_h, \theta_w]
$$

**含义**: 将位置编码分解为时间、高度、宽度三个独立分量，统一处理文本（三维相同）、图像（时间恒定）和视频（时间递增）的输入。

**符号说明**:
- $\theta_t$: 时间维度旋转频率
- $\theta_h, \theta_w$: 空间维度（高度、宽度）旋转频率

### 公式3: MLP Vision-Language Merger 的压缩

$$
\mathbf{z} = \text{MLP}(\text{Concat}[\mathbf{f}_{i,j}, \mathbf{f}_{i,j+1}, \mathbf{f}_{i+1,j}, \mathbf{f}_{i+1,j+1}])
$$

**含义**: 将空间相邻的 4 个 ViT patch 特征拼接后通过两层 MLP 压缩，实现 4 倍 token 压缩，降低 LLM 的计算开销。

**符号说明**:
- $\mathbf{f}_{i,j}$: 第 $(i,j)$ 个 patch 的 ViT 输出特征
- $\mathbf{z}$: 压缩后的 token 嵌入

### 公式4: [[Direct Preference Optimization|DPO]] 损失

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma \left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

**含义**: 后训练阶段通过直接偏好优化对齐人类偏好，无需显式训练奖励模型。

**符号说明**:
- $\pi_\theta$: 当前策略模型
- $\pi_{\text{ref}}$: 参考模型（SFT 后）
- $y_w, y_l$: 偏好对中的胜者和败者响应
- $\beta$: 控制偏离参考模型程度的超参数

---

## 关键图表

### Figure 1: Qwen2.5-VL 整体框架

![[Qwen2.5-VL_fig1_framework.png]]

**说明**: Qwen2.5-VL 架构概览。视觉编码器处理原生分辨率的图像和动态 FPS 的视频，通过 [[MRoPE]] 对齐绝对时间，视觉 token 经 [[MLP]] Merger 压缩后送入 [[Qwen2.5 LLM]] 解码器。ViT 引入 [[Window Attention]]、[[SwiGLU]] 激活和 [[RMSNorm]] 归一化。

### Table 1: Qwen2.5-VL 配置表

| 配置项 | 3B | 7B | 72B |
|--------|-----|-----|------|
| ViT Hidden | 1280 | 1280 | 1280 |
| ViT Layers | 32 | 32 | 32 |
| LLM Hidden | 2048 | 3584 | 8192 |
| LLM Layers | 36 | 28 | 80 |
| LLM KV Heads | 2 | 4 | 8 |
| Vocab Size | 151646 | 151646 | 151646 |

**说明**: 三种规模共享相同的 ViT 和 patch size (14)，仅 LLM 和 Merger 输出维度不同，训练数据量均为 4.1T tokens。

### Table 2: 预训练阶段数据构成

| 阶段 | 数据 | Tokens | 序列长度 | 训练参数 |
|------|------|--------|----------|----------|
| Phase 1: Visual Pre-Training | Caption, Knowledge, OCR | 1.5T | 8192 | ViT only |
| Phase 2: Multimodal Pre-Training | +Pure text, Interleaved, VQA, Video, Grounding, Agent | 2T | 8192 | ViT & LLM |
| Phase 3: Long-Context Pre-Training | +Long Video, Long Agent, Long Document | 0.6T | 32768 | ViT & LLM |

**说明**: 渐进式训练策略：先训练 ViT 对齐，再全参数多模态预训练，最后长序列扩展。

### Table 3: SOTA 对比（核心基准）

| Dataset | Qwen2-VL-72B | Qwen2.5-VL-72B | GPT-4o | Claude-3.5 Sonnet | InternVL2.5-78B |
|---------|-------------|----------------|--------|-------------------|-----------------|
| MMMU | 64.5 | **70.2** | 69.1 | 68.3 | 70.1 |
| MMMU-Pro | 46.2 | **51.1** | 51.9 | 51.5 | 48.6 |
| MathVista | 70.5 | **74.8** | - | 67.7 | 72.3 |
| MathVerse | - | **57.6** | - | - | 51.7 |
| MMBench-EN | 87.4 | **88.6** | - | 83.5 | 88.3 |
| MMStar | 69.5 | **70.8** | 64.7 | 65.1 | 69.5 |
| MME | 2494 | **2448** | 2328 | 1920 | 2494 |
| MuirBench | 63.5 | **70.7** | 68.0 | 55.5 | 63.5 |
| HallBench | 57.4 | **55.2** | 55.0 | 51.6 | 57.4 |
| MMVet | 72.3 | **76.2** | 69.1 | - | 72.3 |
| MM-MT-Bench | - | **7.6** | 7.72 | 7.5 | - |

**关键发现**: 72B 版本在大部分基准上超越或匹配闭源顶级模型。小型版本 7B/3B 显著超越同规模竞品。

### Table 4: 纯文本任务性能

| Dataset | Llama-3.1-405B | Qwen2.5-72B | Qwen2.5-VL-72B |
|---------|---------------|-------------|-----------------|
| MMLU-Pro | 64.4 | 71.1 | **71.2** |
| MMLU-redux | 81.6 | 86.8 | **85.9** |
| LiveBench | 41.5 | 52.3 | **57.0** |
| GPQA | 51.1 | 42.4 | **49.0** |
| MATH | 73.8 | 86.6 | **87.8** |
| HumanEval | 89.0 | 86.0 | **86.3** |

**关键发现**: Qwen2.5-VL-72B 在多模态增强的同时保持了领先的纯文本能力，证明视觉训练未损害语言核心能力。

### Table 5: OCR 与文档理解

| Dataset | Qwen2.5-VL-72B | InternVL2.5-78B | GPT-4o | Gemini 1.5 Pro | Claude-3.5 Sonnet |
|---------|---------------|-----------------|--------|----------------|-------------------|
| CC-OCR | **79.8** | 64.7 | 66.9 | 73.0 | 62.5 |
| OmniDocBench en/zh | **0.226/0.324** | 0.275/0.324 | 0.265/0.435 | 0.230/0.281 | 0.330/0.381 |
| OCRBench | **885** | 797 | 736 | 754 | 788 |
| OCRBench_v2 en/zh | **61.5/63.7** | 56.3/57.2 | 54.3/52.1 | 51.9/43.1 | 45.2/39.6 |
| AI2D | **89.1** | 84.6 | 91.1 | 88.4 | 81.2 |
| DocVQA | **95.1** | 91.1 | - | 93.1 | 95.2 |
| ChartQA | **88.3** | 86.7 | 87.2 | 87.2 | 90.8 |
| InfoVQA | **84.1** | 80.7 | - | 81.0 | 74.3 |

**关键发现**: Qwen2.5-VL 在文档和 OCR 任务上全面领先。OCRBench_v2 上大幅超越 Gemini 1.5 Pro（英文 +9.6%，中文 +20.6%）。

### Table 6: 空间理解/Grounding

| Dataset | Qwen2.5-VL-72B | InternVL2.5-78B | Molmo-72B | Grounding DINO | Gemini 1.5 Pro |
|---------|---------------|-----------------|-----------|----------------|----------------|
| RefCOCO val | **92.7** | 93.7 | - | 90.6 | 73.2 |
| RefCOCO+ val | **88.9** | 90.4 | - | 88.2 | 62.5 |
| RefCOCOg val | **89.9** | 92.7 | - | 86.1 | 75.2 |
| ODinW-13 (mAP) | **43.1** | 31.7 | - | 55.0 | 36.7 |
| PointGrounding | **67.3** | - | - | - | - |

**关键发现**: Qwen2.5-VL 缩小了通用模型与专用检测模型在开放词汇检测上的差距，并解锁了点级定位能力。

### Table 7: Counting (计数)

| Dataset | Qwen2.5-VL-72B | GPT-4o | Claude-3.5 Sonnet | Molmo-72B | InternVL2.5-78B | Gemini 1.5 Pro |
|---------|---------------|--------|-------------------|-----------|-----------------|----------------|
| CountBench | **93.6** | 87.9 | 89.7 | 91.2 | 72.1 | 85.5 |

**关键发现**: 使用 "先检测后计数" 提示策略，Qwen2.5-VL 在计数任务上达到领先精度。

### Table 8: 视频理解

| Dataset | Qwen2.5-VL-72B | GPT-4o | Gemini 1.5 Pro |
|---------|---------------|--------|----------------|
| Video-MME (w/o sub.) | **73.3** | 71.9 | 75.0 |
| Video-MME (w/ sub.) | **79.1** | 77.2 | 81.3 |
| Video-MMMU | **60.2** | 61.2 | 53.9 |
| MVBench | **70.4** | 64.6 | 60.5 |
| LVBench | **47.3** | 30.8 | 33.1 |
| MLVU | **74.6** | 73.8 | - |
| EgoSchema | **76.2** | 72.2 | 71.2 |
| Charades-STA (mIoU) | **50.9** | 35.7 | 43.6 |
| LongVideoBench | **60.7** | 66.7 | 64.0 |

**关键发现**: LVBench（长视频）上大幅超越 GPT-4o（+16.5），Charades-STA 时间定位 mIoU=50.9，每视频最多分析 768 帧，总视频 token 不超过 24576。

### Table 9: GUI Agent

| Benchmark | Qwen2.5-VL-72B | GPT-4o | Gemini 2.0 | Claude | Aguvis-72B | Qwen2-VL-72B |
|-----------|---------------|--------|------------|--------|------------|--------------|
| ScreenSpot | **87.1** | 18.1 | 84.0 | 83.0 | 89.2 | 1.6 |
| ScreenSpot Pro | **43.6** | - | 17.1 | - | 23.6 | 1.6 |
| Android Control (High) | **67.36** | 20.8 | 28.5 | 12.5 | 66.4 | 59.1 |
| AndroidWorld (SR) | **35%** | 34.5% | 26% | - | 26.1% | 6% |
| MobileMiniWob++ (SR) | **68%** | 61% | 42% | 61% | 66% | 50% |
| OSWorld | 8.83 | 5.03 | 4.70 | 14.90 | 10.26 | 2.42 |

**关键发现**: ScreenSpot Pro（专业高分辨率）上远超所有基线（43.6% vs 23.6%），并能在真实动态环境中无需辅助标记完成操作。

---

## 训练细节

### 预训练 (Pre-Training)

| 阶段 | 数据重点 | 数据量 | 序列长度 |
|------|----------|--------|----------|
| Phase 1 | 图像描述、知识、OCR | 1.5T | 8,192 |
| Phase 2 | 图文交错、VQA、视频、定位、Agent、纯文本 | 2T | 8,192 |
| Phase 3 | 长视频、长 Agent、长文档 | 0.6T | 32,768 |

- ViT 使用 DataComp + 内部数据从零训练
- LLM 使用 [[Qwen2.5 LLM]] 预训练权重初始化
- Phase 1 仅训练 ViT，Phase 2-3 全参数训练
- 使用动态打包（dynamic packing）平衡 GPU 计算负载

### 数据构成

- **图文交错数据**: 四阶段评分系统筛选（文本质量、图文相关性、图文互补性、信息密度平衡）
- **定位数据**: 绝对坐标，10K+ 类别，使用 [[Copy-Paste Augmentation]] 和 [[Grounding DINO]] + [[SAM]] 合成
- **文档全解析数据**: HTML 格式统一表示（段落、表格、图表、公式、化学式、乐谱），含坐标框和阅读顺序
- **OCR 数据**: 多语言（法/德/意/西/葡/阿/俄/日/韩/越），图表合成 1M 样本，表格处理 6M 真实样本
- **视频数据**: 动态 FPS 采样 + 半小时以上长视频的合成描述 + 时间戳（秒/hmsf 格式）
- **Agent 数据**: 截图描述 + UI 元素定位标注 + 多步操作轨迹 + 推理过程（人工+模型标注）

### 后训练 (Post-Training)

- **SFT**: 约 2M 条目（50% 纯文本 + 50% 多模态），单轮+多轮对话，ChatML 格式，ViT 冻结
- **DPO**: 仅图文+纯文本偏好数据，每个样本仅使用一次
- 使用两阶段数据过滤流水线（领域分层分类 + 领域定制过滤）
- [[Rejection Sampling]] 增强推理：仅保留与 ground truth 匹配的 [[Chain-of-Thought|CoT]] 输出

---

## 实验

### 评估维度

| 能力维度 | 代表基准 |
|----------|----------|
| 大学级推理 | MMMU, MMMU-Pro |
| 数学 | MathVista, MATH-Vision, MathVerse |
| 通用 VQA | MMBench, MMStar, MME, MuirBench, MMVet |
| 文档与OCR | CC-OCR, OmniDocBench, OCRBench, DocVQA, ChartQA, InfoVQA |
| 空间理解 | RefCOCO, ODinW, PointGrounding, CountBench |
| 视频理解 | Video-MME, Video-MMMU, MVBench, LVBench, EgoSchema, MLVU |
| 视频定位 | Charades-STA |
| GUI Agent | ScreenSpot, ScreenSpot Pro, AndroidWorld, OSWorld |
| 纯文本 | MMLU-Pro, GPQA, MATH, HumanEval |

### 关键发现

1. **72B 全面 SOTA**: 在文档/OCR 领域尤为突出，OCRBench_v2 中英双轨大幅领先 Gemini 1.5 Pro
2. **缩放有效性**: 3B/7B 均超越同规模最优模型，证明架构和数据策略的普适性
3. **长视频优势**: LVBench 超越 GPT-4o 16.5 个百分点
4. **Agent 突破**: ScreenSpot Pro 43.6% 远超次优的 23.6%，体现感知能力的实际价值
5. **纯文本无损**: 多模态训练未牺牲语言能力，LiveBench 甚至高于纯文本 Qwen2.5-72B

---

## 批判性思考

### 优点
1. **架构创新实用且高效**: Window Attention + 绝对时间 MRoPE 并非激进改动，但在效率和能力上产生显著增益
2. **数据工程扎实**: 4.1T tokens 的大规模高质量数据（含 HTML 文档格式、多语言 OCR、10K+ 类别定位），数据飞轮清晰
3. **三尺寸覆盖全面**: 从边缘到云端的部署需求均被满足，且小模型性能不俗
4. **能力边界清晰**: 文档、定位、视频三个方面均有具体 benchmark 支撑，对比公平
5. **训练策略健全**: 渐进式三阶段预训练 + SFT + DPO 后训练，数据过滤流水线设计细致

### 局限性
1. **未开源模型权重**: 论文声称 open-source philosophy，但实际未说明权重开源情况（注：后续已开源）
2. **DPO 仅单轮使用**: 每个偏好样本仅用一次，可能未充分利用偏好数据的潜力
3. **幻觉评测一般**: HallBench 上 55.2 低于 Qwen2-VL 的 57.4，说明幻觉问题仍未根本解决
4. **缺乏消融实验**: 论文未展示各架构组件（Window Attention、绝对时间 MRoPE、动态 FPS）的独立消融贡献
5. **纯文本基准对比有限**: Table 4 中与 Llama-3.1-70B/405B 对比不够全面（如 MMLU-Pro 仅部分子集）
6. **CoT 视觉对齐不足**: 论文自述"中间推理步骤可能未能充分整合视觉信息"，且认为这仍是未解决的挑战

### 潜在改进方向
1. 引入更鲁棒的视觉 CoT 机制，确保推理的每一步都真正依赖视觉证据
2. 增强多轮偏好优化（iterative DPO）以充分利用偏好数据
3. 进一步压缩 ViT token（如 Qwen3-VL 中的 deeper merger）以降低推理成本
4. 提升幻觉鲁棒性，特别是在长视频和复杂文档场景

### 可复现性评估
- [x] 代码开源 (GitHub: QwenLM/Qwen2.5-VL)
- [x] 预训练模型 (HuggingFace: Qwen)
- [x] 训练细节完整 (数据构成、训练阶段、超参数均列出)
- [ ] 数据集可获取 (预训练数据未公开，使用专有/合成数据)

---

## 关联笔记

### 基于
- [[Qwen2-VL]]: 直接前身，MRoPE 基础，架构范式延续
- [[Qwen2.5 LLM]]: LLM 初始化权重来源
- [[Vision Transformer|ViT]]: 视觉编码器骨干
- [[MRoPE]]: 多模态位置编码基础

### 对比
- [[GPT-4o]]: 闭源旗舰，Qwen2.5-VL-72B 整体匹配或超越
- [[Claude 3.5 Sonnet]]: 文档理解和 Agent 对比基线
- [[InternVL2|InternVL2.5-78B]]: 最强开源竞品（同期），Qwen2.5-VL 多数基准超越
- [[Gemini 1.5 Pro]]: 视频理解对比基线，OCR 上 Qwen2.5-VL 大幅领先

### 方法相关
- [[Window Attention]]: 视觉编码器高效化核心
- [[Native Dynamic Resolution]]: 动态分辨率处理
- [[Dynamic FPS Sampling]]: 动态帧率训练
- [[Absolute Time Encoding]]: 时间对齐 MRoPE
- [[Document Omni-Parsing]]: 统一文档解析格式
- [[Rejection Sampling]]: CoT 推理增强
- [[Direct Preference Optimization|DPO]]: 偏好对齐
- [[Grounding DINO]]: 定位数据合成
- [[Copy-Paste Augmentation]]: 数据增强

### 后续
- [[Qwen3-VL]]: 下一代，引入 deeper merger 和 Thinking

---

## 速查卡片

> [!summary] Qwen2.5-VL Technical Report
> - **核心**: 三组件 VLM (ViT+Merger+LLM)，Window Attention + 绝对时间 MRoPE + 动态分辨率
> - **规模**: 3B / 7B / 72B，均用 4.1T tokens 训练
> - **三大能力**: 文档全解析 / 精确目标定位 (Box+Point) / 超长视频理解 (小时级)
> - **结果**: 72B 匹配 GPT-4o/Claude 3.5 Sonnet，文档/OCR/视频领先
> - **代码**: https://github.com/QwenLM/Qwen2.5-VL

---

*笔记创建时间: 2025-05-18*
