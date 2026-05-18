---
title: "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond"
method_name: "Qwen-VL"
authors: [Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, Jingren Zhou]
year: 2023
venue: arXiv
tags: [vlm, multimodal, grounding, ocr, vision-language, instruction-tuning]
zotero_collection: 多模态/QWen/MML
image_source: local
arxiv_html: https://arxiv.org/html/2308.12966v3
created: 2026-05-18
---

# 论文笔记：Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Alibaba Group |
| 日期 | October 2023 |
| 项目主页 | https://github.com/QwenLM/Qwen-VL |
| 对比基线 | [[LLaVA]], [[BLIP-2]], [[InstructBLIP]], [[Kosmos-2]], [[Shikra]] |
| 链接 | [arXiv](https://arxiv.org/abs/2308.12966) / [Code](https://github.com/QwenLM/Qwen-VL) |

---

## 一句话总结

> 基于 Qwen-7B 的大规模多语言视觉语言模型，通过三阶段训练和位置感知适配器，统一实现图像描述、视觉问答、定位、文本阅读等多种视觉语言任务。

---

## 核心贡献

1. **多功能统一架构**: 基于 [[Qwen]]-7B LLM + ViT-bigG + 位置感知 VL 适配器，以简洁架构覆盖 captioning、VQA、grounding、OCR 等任务
2. **三阶段训练流水线**: 预训练(1.4B 图文对) -> 多任务预训练(7 类任务) -> 指令微调(350k)，逐步解锁能力
3. **定位与文本阅读能力**: 通过将 bounding box 归一化为文本字符串，使 LLM 原生支持 fine-grained grounding 和 text-reading
4. **多语言多图对话**: 支持中英文、多图输入、多轮对话，在多个 benchmark 上达到 SOTA

---

## 问题背景

### 要解决的问题
现有开源 [[LVLM]] 普遍训练不充分，缺乏细粒度感知能力（如 [[Visual Grounding|object grounding]] 和 text-reading），且落后于闭源模型，限制了多模态应用和社区研究。

### 现有方法的局限
- 开源 LVLM（如 [[LLaVA]]、[[MiniGPT-4]]）主要在粗粒度图像理解上表现尚可，缺乏定位能力
- [[Kosmos-2]]、[[Shikra]] 等虽有 grounding 能力，但在文本阅读、多语言、多图对话等维度不够全面
- 大部分开源模型训练数据量和质量不足，与闭源模型（[[GPT-4o|GPT-4V]]）差距大

### 本文的动机
构建一个在多种视觉语言任务上都有竞争力的开源通用模型，弥补细粒度感知的短板，同时保持多语言和多图理解能力。

---

## 方法详解

### 模型架构

Qwen-VL 采用 **LLM + ViT + Adapter** 的经典 [[LVLM]] 架构，总参数量 **9.6B**：

- **Large Language Model**: [[Qwen]]-7B（decoder-only），用预训练权重初始化，7.7B 参数
- **Visual Encoder**: [[Vision Transformer]] (ViT-bigG from [[OpenCLIP]])，将图像切分为 stride=14 的 patches 生成图像特征，1.9B 参数
- **Position-aware Vision-Language Adapter**: 单层 [[Cross-Attention]] 模块，用一组可训练的 query embeddings（256 个）压缩 ViT 输出的长序列，同时在 query-key pairs 中加入 2D 绝对位置编码以保留空间信息，0.08B 参数

**输入接口**：
- 图像输入：经 ViT + Adapter 压缩为固定长度 256 的序列，两端加 `<img>` 和 `</img>` 特殊 token
- Bounding Box 输入/输出：坐标归一化到 `[0, 1000)`，格式化为字符串 `(X_topleft, Y_topleft), (X_bottomright, Y_bottomright)`，用 `<box>` `</box>` 包裹；用 `<ref>` `</ref>` 标记被引用的描述文本

**设计亮点**：将 grounding 任务完全转化为文本生成任务，无需额外的位置 token vocabulary，充分利用 LLM 的文本理解和生成能力。

### 三阶段训练

#### 阶段1: 预训练（图文对齐）

**目的**: 让视觉编码器和适配器学会将图像特征映射到 LLM 的语义空间。

- **数据**: 1.4B 图文对（清洗后，源自 LAION-en/zh、LAION-COCO、DataComp、Coyo、CC12M 等），77.3% 英文 + 22.7% 中文
- **策略**: 冻结 LLM，只训练 ViT + Adapter；图像分辨率 224x224
- **超参**: lr max 2e-4，batch size 30720，训练 50k steps，消耗约 1.5B 样本

#### 阶段2: 多任务预训练

**目的**: 赋予模型细粒度视觉理解和多任务能力。

- **数据**: 7 类任务混合训练，共约 76.8M 样本
  - Captioning (19.7M)、VQA (3.6M)、Grounding (3.5M)、Referring Grounding (8.7M)、Grounded Captioning (8.7M)、OCR (24.8M)、纯文本自回归 (7.8M)
  - OCR 数据：SynthDoG 合成数据（英文+中文） + Common Crawl 的 PDF/HTML 渲染数据
- **策略**: 解锁 LLM，全模型训练；分辨率提升至 448x448
- **超参**: lr max 5e-5，batch size 4096，训练 19k steps

#### 阶段3: 监督微调（指令对齐）

**目的**: 增强指令遵循和对话能力，得到 Qwen-VL-Chat。

- **数据**: 350k 多模态 + 纯文本对话数据，包括人工标注、模型生成和策略拼接
- **格式**: [[ChatML]] 格式，支持多图对话（`Picture id:` 前缀区分不同图像）
- **策略**: 冻结 ViT，训练 LLM + Adapter
- **超参**: lr max 1e-5，batch size 128，训练 8k steps

---

## 关键公式

### 公式1: [[Cross-Entropy Loss|文本生成损失]]

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(y_t \mid y_{<t}, x)
$$

**含义**: 标准自回归语言建模损失，三个训练阶段统一使用 cross-entropy 最小化文本 token 的预测误差。

**符号说明**:
- $\theta$: 模型参数（不同阶段训练不同模块）
- $y_t$: 第 $t$ 个目标 token
- $x$: 输入（图像特征 + 文本前缀）
- $T$: 目标序列长度

---

## 关键图表

### Figure 1: SOTA Performance Overview / 性能总览

![Qwen-VL_fig1_overview.png]

**说明**: Qwen-VL 在多个视觉语言任务上相比同类规模的通用模型取得 SOTA 性能。覆盖 image captioning、VQA、grounding、text-reading 等多个维度。

### Figure 2: Qualitative Examples / 定性示例

![Qwen-VL_fig2_examples.png]

**说明**: Qwen-VL-Chat 展示多图输入、多轮对话、多语言、文本阅读、定位和细粒度识别能力。示例包括：中文 grounding、图像对比分析、OCR 问答等。

### Figure 3: Three-stage Training Pipeline / 三阶段训练流水线

![[Qwen-VL_fig3_training.png]]

**说明**: 训练流程：阶段1 用低分辨率图文对训练 ViT + Adapter（LLM 冻结）；阶段2 用高分辨率多任务数据全模型训练；阶段3 用对话数据做指令微调（ViT 冻结）。

### Figure 4: Few-shot Learning Results / 少样本学习结果

![[Qwen-VL_fig4_fewshot.png]]

**说明**: Qwen-VL 在 OKVQA、Vizwiz、TextVQA、Flickr30k 上的 few-shot in-context learning 表现。Qwen-VL 的少样本性能优于同参数量模型（Flamingo-9B、OpenFlamingo-9B、IDEFICS-9B），甚至在某些任务上可比肩 80B 模型。

### Figure 5: Grounding and OCR Data Visualization / 训练数据可视化

![[Qwen-VL_fig5_ocr_data.png]]

**说明**: 展示 grounding 数据（上排）和 OCR 数据（下排）的可视化示例。Grounding 数据包含名词短语与对应 bounding box 的对齐；OCR 数据使用 SynthDoG 在自然场景背景上合成文本。

### Figure 6: Convergence of Pre-training Stage / 预训练收敛曲线

![[Qwen-VL_fig6_convergence.png]]

**说明**: 阶段1 预训练的收敛情况：(a) 训练 loss 随图像数量稳步下降；(b) Flickr30K Caption CIDEr 分数上升；(c) VQAv2 zero-shot VQA 分数虽有波动但整体上升（注意阶段1 并未加入 VQA 数据）。

### Figure 7: Learnable Queries Ablation / 可学习查询数量消融

**说明**: 不同压缩特征长度（L64/144/256/400）的初始 loss 和收敛 loss 对比。查询数过少导致信息丢失，过多导致收敛困难。最终选择 **256** 个查询作为平衡点。（图片为矢量图表，未从 PDF 中提取）

### Figure 8: Window vs Global Attention for ViT / 窗口注意力消融

**说明**: 在使用高分辨率 ViT 时，Window Attention 的 loss 显著高于 Global Attention，但训练速度相似。因此 **Qwen-VL 使用 Vanilla Global Attention**。对于 896x896 分辨率，Window Attention 可加速（25s/iter vs 60s/iter），但训练速度仍过慢未采用。（图片为矢量图表，未从 PDF 中提取）

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LAION-5B (en+zh) | 5B 原始/1.4B 清洗后 | 网络爬取，多语言 | 阶段1 预训练 |
| LAION-COCO | 600M/300M 清洗后 | 合成 caption | 阶段1 预训练 |
| DataComp + Coyo | 2.1B/500M 清洗后 | 网络爬取 | 阶段1 预训练 |
| CC12M + CC3M + SBU + COCO Caption | ~16M | 学术 caption 数据集 | 阶段1 预训练 |
| GQA + VGQA + VQAv2 + DVQA + OCR-VQA + DocVQA + TextVQA + ChartQA + AI2D | ~3.6M | VQA 相关 | 阶段2 多任务训练 |
| GRIT + Visual Genome + RefCOCO/+/g | ~20.9M | Grounding 相关 | 阶段2 多任务训练 |
| SynthDoG + CC PDF/HTML | ~24.8M | OCR 数据 | 阶段2 多任务训练 |

### 评估基准

| 任务类别 | 具体数据集 | 指标 |
|----------|-----------|------|
| 图像描述 | Nocaps, Flickr30K | CIDEr |
| 通用 VQA | VQAv2, OKVQA, GQA, ScienceQA-Img, VizWiz | VQA Score / EM / Accuracy |
| 文本导向 VQA | TextVQA, DocVQA, ChartQA, AI2D, OCR-VQA | VQA Score / ANLS / EM |
| 指代表达理解 | RefCOCO/+/g, GRIT | Accuracy |
| 指令遵循 | TouchStone, SEED-Bench, MME | GPT-4 Score / Accuracy |

### 实现细节

- **Backbone**: [[Vision Transformer|ViT]]-bigG ([[OpenCLIP]]) + [[Qwen]]-7B
- **优化器**: [[Adam|AdamW]] ($\beta_1=0.9, \beta_2=0.98, \epsilon=1\mathrm{e}{-6}$)
- **学习率策略**: cosine decay，各阶段 warmup 500/400/3000 steps
- **Weight Decay**: 0.05
- **Gradient Clip**: 1.0
- **数值精度**: BFloat16
- **并行策略**: 阶段2 对 ViT 和 LLM 使用 model parallelism (size 2)

### 主要结果

**Image Captioning & General VQA** (Table 4)：Qwen-VL 在 Flickr30K zero-shot 上获 85.8 CIDEr，超越 Flamingo-80B；在 VQAv2 (79.5)、OKVQA (58.6) 上大幅领先同期 LVLM。

**Text-oriented VQA** (Table 5)：Qwen-VL-Chat 在 TextVQA (63.8)、DocVQA (65.1) 上表现突出，归功于高分辨率训练和丰富的 OCR 数据。

**Referring Expression Comprehension** (Table 6)：在 RefCOCO/+/g 所有 split 上均取得 top-tier 结果。Qwen-VL 在 RefCOCO test-A 上达 92.27，RefCOCOg test 上达 86.32，在通用模型中显著领先。

**Few-shot Learning** (Figure 4)：Qwen-VL 展现强大的上下文学习能力，仅用随机采样的 few-shot exemplars 即可媲美更大的模型（如 Flamingo-80B, IDEFICS-80B）。

**Instruction Following** (Table 7)：Qwen-VL-Chat 在 TouchStone(中文 401.2)、SEED-Bench(58.2)、MME(1487.58) 上均明显优于现有 LVLM。

**纯文本能力** (Table 11)：Qwen-VL 在多模态训练后纯文本能力未退化（MMLU 50.7, C-Eval 51.1），证明多任务训练中混入纯文本数据的有效性。

### 关键表格

### Table 1: Qwen-VL Model Parameters / 模型参数

| 组件 | 参数量 |
|------|--------|
| Vision Encoder (ViT-bigG) | 1.9B |
| VL Adapter | 0.08B |
| LLM (Qwen-7B) | 7.7B |
| **Total** | **9.6B** |

### Table 2: Pre-training Data Details / 预训练数据详情

| 语言 | 数据集 | 原始量 | 清洗后 | 保留率 |
|------|--------|--------|--------|--------|
| English | LAION-en | 2B | 280M | 14% |
| English | LAION-COCO | 600M | 300M | 50% |
| English | DataComp | 1.4B | 300M | 21% |
| English | Coyo | 700M | 200M | 28% |
| English | CC12M | 12M | 8M | 66% |
| English | CC3M | 3M | 3M | 100% |
| English | SBU | 1M | 0.8M | 80% |
| English | COCO Caption | 0.6M | 0.6M | 100% |
| Chinese | LAION-zh | 108M | 105M | 97% |
| Chinese | In-house Data | 220M | 220M | 100% |
| **Total** | | **5B** | **1.4B** | **28%** |

### Table 3: Multi-task Pre-training Data / 多任务预训练数据

| 任务 | 样本量 | 数据来源 |
|------|--------|----------|
| Captioning | 19.7M | LAION-en/zh, DataComp, Coyo, CC12M/3M, SBU, COCO, In-house |
| VQA | 3.6M | GQA, VGQA, VQAv2, DVQA, OCR-VQA, DocVQA, TextVQA, ChartQA, AI2D |
| Grounding | 3.5M | GRIT |
| Ref Grounding | 8.7M | GRIT, Visual Genome, RefCOCO/+/g |
| Grounded Captioning | 8.7M | GRIT, Visual Genome, RefCOCO/+/g |
| OCR | 24.8M | SynthDoG-en/zh, Common Crawl PDF/HTML |
| Pure-text AR | 7.8M | In-house Data |

### Table 4: Image Captioning and General VQA Results

| Model | Nocaps (0-shot) | Flickr30K (0-shot) | VQAv2 | OKVQA | GQA (0-shot) | VizWiz (0-shot) | SciQA-Img (0-shot) |
|-------|-----------------|-------------------|-------|-------|-------------|----------------|-------------------|
| Flamingo-9B | - | 61.5 | 51.8 | 44.7 | - | 28.8 | - |
| Flamingo-80B | - | 67.2 | 56.3 | 50.6 | - | 31.6 | - |
| Kosmos-1 | - | 67.1 | 51.0 | - | - | 29.2 | - |
| Kosmos-2 | - | 80.5 | 51.1 | - | - | 19.6 | - |
| BLIP-2 (Vicuna-13B) | 100.0 | 71.6 | 65.0 | 45.9 | - | 33.4 | - |
| InstructBLIP (Vicuna-13B) | 103.9 | 82.8 | 77.36 | 47.16 | 49.5 | 35.2 | 61.0 |
| Shikra (Vicuna-13B) | 121.9 | 73.9 | 77.9 | 58.6 | 59.3 | 38.9 | 63.1 |
| **Qwen-VL (Qwen-7B)** | **121.4** | **85.8** | **79.5** | **56.6** | **59.3** | **38.9** | **67.1** |
| **Qwen-VL-Chat** | **120.2** | **81.0** | **78.2** | **56.6** | **57.5** | **38.9** | **68.2** |

**关键发现**: Qwen-VL 在 Flickr30K CIDEr (85.8) 上超越所有通用模型（包括参数更多的 Flamingo-80B）。VQAv2 上 Qwen-VL (79.5) 大幅领先。

### Table 5: Text-oriented VQA Results

| Model | TextVQA | DocVQA | ChartQA | AI2D | OCR-VQA |
|-------|---------|--------|---------|------|---------|
| BLIP-2 (Vicuna-13B) | 42.4 | - | - | - | - |
| InstructBLIP (Vicuna-13B) | 50.7 | - | - | - | - |
| mPLUG-DocOwl (LLaMA-7B) | 52.6 | 62.2 | 57.4 | 42.1 | - |
| Pix2Struct-Large (1.3B) | - | 76.6 | 58.6 | 42.1 | 71.3 |
| **Qwen-VL (Qwen-7B)** | **63.8** | **65.1** | **65.7** | **62.3** | **75.7** |
| **Qwen-VL-Chat** | **61.5** | **62.6** | **66.3** | **57.7** | **70.5** |
| PALI-X-55B (Specialist SOTA) | 71.44 | 80.0 | 70.0 | 81.2 | 75.0 |

**关键发现**: Qwen-VL 在 ChartQA (65.7) 和 OCR-VQA (75.7) 上表现优异，体现了高分辨率训练和丰富 OCR 数据的收益。

### Table 6: Referring Expression Comprehension Results

| Model | RefCOCO val | RefCOCO test-A | RefCOCO test-B | RefCOCO+ val | RefCOCO+ test-A | RefCOCO+ test-B | RefCOCOg val | RefCOCOg test | GRIT refexp |
|-------|-------------|-----------------|-----------------|--------------|-----------------|-----------------|--------------|---------------|-------------|
| GPV-2 | - | - | - | - | - | - | - | - | 51.50 |
| OFA-L* | 79.96 | 83.67 | 76.39 | 68.29 | 76.00 | 61.75 | 67.57 | 67.58 | 61.70 |
| Unified-IO | 86.70 | 87.01 | 80.24 | 81.60 | 87.36 | 88.24 | 82.27 | 82.19 | 78.61 |
| VisionLLM-H | - | - | - | - | - | 82.75 | 72.12 | 82.64 | 69.34 |
| Shikra-7B | 87.83 | 91.11 | 81.81 | 82.89 | 87.79 | 91.46 | 82.27 | 83.16 | 69.03 |
| Shikra-13B | 89.36 | 92.26 | 85.34 | 83.12 | 88.25 | 85.24 | 82.64 | 83.23 | 78.22 |
| **Qwen-VL-7B** | **88.55** | **92.27** | **84.51** | **82.82** | **88.59** | **89.26** | **85.58** | **85.48** | - |
| **Qwen-VL-7B-Chat** | **89.36** | **92.26** | **88.25** | **83.12** | **88.25** | **88.77** | **85.96** | **86.32** | **78.22** |
| G-DINO-L (Specialist) | 90.56 | 93.19 | 88.24 | 82.75 | 88.95 | - | - | - | - |
| ONE-PEACE (Specialist) | 92.58 | 94.18 | 89.26 | 88.77 | 92.21 | 89.22 | 89.27 | - | - |

**关键发现**: Qwen-VL-Chat 在 RefCOCOg test (86.32) 上超越所有通用模型，接近 specialist SOTA。Qwen-VL 的定位能力主要来自 grounding 数据的丰富性。

### Table 7: Instruction-following Benchmarks

| Model | TouchStone En | TouchStone Cn | SEED-Bench All | SEED-Bench Img | SEED-Bench Video | MME Perception | MME Cognition |
|-------|--------------|--------------|----------------|----------------|-----------------|----------------|---------------|
| VisualGLM | 488.5 | 247.1 | - | - | - | - | - |
| MiniGPT4 | 531.7 | - | 42.8 | 47.4 | 29.9 | 581.67 | 144.29 |
| InstructBLIP | 552.4 | - | 53.4 | 58.8 | 38.1 | 1212.82 | 291.79 |
| LLaVA | 602.7 | - | 32.7 | 35.2 | 25.8 | 502.82 | 214.64 |
| mPLUG-Owl | 605.4 | - | 34.0 | 37.9 | 23.0 | 967.34 | 276.07 |
| **Qwen-VL-Chat** | **645.2** | **401.2** | **58.2** | **65.4** | **37.8** | **1487.58** | **360.71** |

**关键发现**: Qwen-VL-Chat 在三个指令遵循 benchmark 上全面领先。中文 TouchStone 获 401.2 分，远超第二名 VisualGLM (247.1)。MME Perception 得分 1487.58 为当时最高。

### Table 8: Training Hyperparameters / 训练超参数

| 配置项 | 阶段1: 预训练 | 阶段2: 多任务预训练 | 阶段3: 监督微调 |
|--------|-------------|-------------------|----------------|
| ViT 初始化 | Open-CLIP-bigG | Qwen-VL 阶段1 | Qwen-VL 阶段2 |
| LLM 初始化 | Qwen-7B | Qwen-7B | Qwen-VL 阶段2 |
| VL Adapter 初始化 | random | Qwen-VL 阶段1 | Qwen-VL 阶段2 |
| 图像分辨率 | 224x224 | 448x448 | 448x448 |
| ViT 序列长度 | 256 | 1024 | 1024 |
| LLM 序列长度 | 512 | 2048 | 2048 |
| 可学习查询数 | 256 | 256 | 256 |
| 峰值学习率 | 2e-4 | 5e-5 | 1e-5 |
| 最小学习率 | 1e-6 | 1e-5 | 1e-6 |
| 训练步数 | 50k | 19k | 8k |
| Warmup 步数 | 500 | 400 | 3k |
| 全局 batch size | 30720 | 4096 | 128 |
| ViT lr decay | 0.95 | 0.95 | - |

### Table 10: Window vs Global Attention Training Speed

| 模型输入分辨率与注意力类型 | 训练速度 |
|--------------------------|---------|
| 448x448, Global Attention | 10s / iter |
| 448x448, Window Attention | 9s / iter |
| 896x896, Global Attention | 60s / iter |
| 896x896, Window Attention | 25s / iter |

### Table 11: Pure-text Benchmarks

| Model | MMLU | CMMLU | C-Eval |
|-------|------|-------|--------|
| LLaMA-7B | 35.1 | 26.8 | - |
| LLaMA2-7B | 46.8 | 31.8 | 32.5 |
| Baichuan-7B | 42.3 | 44.4 | 42.8 |
| ChatGLM2-6B | 47.9 | 48.8 | 51.7 |
| InternLM-7B | 51.0 | 51.8 | 52.8 |
| Qwen-7B (最终版) | 58.2 | 62.2 | 63.5 |
| Qwen-7B (中间版) | 49.9 | - | 48.5 |
| **Qwen-VL** | **50.7** | **49.5** | **51.1** |

**关键发现**: Qwen-VL 使用 Qwen-7B 的中间 checkpoint 初始化，经过多模态多任务训练后，纯文本能力不仅未退化，反而有提升（MMLU 49.9 -> 50.7, C-Eval 48.5 -> 51.1）。

---

## 批判性思考

### 优点
1. **架构简洁高效**: 用单一 LLM 统一处理所有任务（captioning、VQA、grounding、OCR），bounding box 文本化是 clever trick
2. **训练策略完备**: 三阶段训练逐步解锁能力，从粗粒度对齐到细粒度感知再到对话，逻辑清晰
3. **数据工程扎实**: 数据清洗和配比值得参考（5B -> 1.4B，保留率 28%），7 类任务的数据量和来源都有明确记录
4. **实验全面**: 覆盖 5 大类评估任务、15+ benchmark，与大量同期工作对比，附有详细的消融实验（查询数、注意力机制）

### 局限性
1. **仅支持单帧图像**: 不支持视频输入（但 SEED-Bench 实验显示简单采样 4 帧即可迁移，暗示扩展空间大）
2. **图像分辨率仍有限**: 448x448 对于文档分析和细粒度 OCR 偏小（后续 Qwen2-VL 已改进为动态分辨率）
3. **仅支持 text output**: 模型不能输出图像或修改图像，不具备多模态生成能力
4. **LLM 版本未用最新**: 使用 Qwen-7B 中间 checkpoint 而非最终版，可能影响文本能力上限
5. **Grounding 粒度**: bounding box 级别定位虽实用，但缺少像素级分割能力

### 潜在改进方向
1. **引入动态分辨率**: 后续 Qwen2-VL 的 Naive Dynamic Resolution 已解决（但本文撰写时尚无）
2. **增加视频支持**: 扩展时间维度的输入处理，Qwen2-VL 已实现
3. **多模态生成**: 联合图像生成能力（text-to-image + image editing），走向真正的统一多模态模型
4. **更强的视觉 backbone**: ViT-bigG 虽然是当时大模型，但后续工作（如 DINOv2、SigLIP、InternViT）可能效果更好

### 可复现性评估
- [x] 代码开源 (GitHub: QwenLM/Qwen-VL)
- [x] 预训练模型 (Qwen-VL 和 Qwen-VL-Chat 权重均已公开)
- [x] 训练细节完整 (Table 8 超参数详尽)
- [x] 数据集可获取 (大部分使用公开数据集，OCR/纯文本数据为 in-house)

---

## 关联笔记

### 基于
- [[Qwen]]: Qwen-7B 作为 LLM backbone
- [[Vision Transformer]]: ViT-bigG 作为视觉编码器
- [[OpenCLIP]]: ViT 预训练权重来源
- [[LLaVA]]: 同期 LVLM，instruction tuning 思路参考
- [[Kosmos-2]]: grounding 数据格式参考（GRIT 数据集）
- [[Shikra]]: bounding box 文本化表示参考

### 对比
- [[BLIP-2]]: 不同 adapter 设计（Q-Former vs Cross-Attention）
- [[InstructBLIP]]: 指令微调方法对比
- [[Flamingo]]: few-shot learning 能力对比
- [[mPLUG-Owl]]: 同级别的开源 LVLM

### 方法相关
- [[LVLM]]: 大视觉语言模型框架
- [[Cross-Attention]]: VL Adapter 核心机制
- [[Instruction Tuning]]: 第三阶段训练方法
- [[Supervised Fine-Tuning]]: 对话能力对齐方法
- [[Visual Grounding]]: 核心能力之一
- [[Few-shot Learning]]: 上下文学习评估
- [[ChatML]]: 对话数据格式

### 后继工作
- [[Qwen2-VL]]: 动态分辨率 + MRoPE + 更强的视觉 backbone
- [[Qwen2.5-VL]]: 更强的推理和长视频理解
- [[Qwen3-VL]]: 原生多分辨率 + 增强推理链

---

## 速查卡片

> [!summary] Qwen-VL: A Versatile Vision-Language Model
> - **核心**: 基于 Qwen-7B + ViT-bigG + 位置感知 Cross-Attention Adapter 的统一多模态模型
> - **方法**: 三阶段训练（预训练 -> 多任务 -> SFT），bounding box 文本化实现 grounding
> - **结果**: 9.6B 参数，在 captioning/VQA/grounding/OCR 上超越同期通用模型
> - **代码**: https://github.com/QwenLM/Qwen-VL

---

*笔记创建时间: 2026-05-18T16:41:00*
