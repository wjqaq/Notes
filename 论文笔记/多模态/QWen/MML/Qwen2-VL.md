---
title: "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution"
method_name: "Qwen2-VL"
authors: [Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang]
year: 2024
venue: arXiv
tags: [vision-language-model, dynamic-resolution, multimodal-position-encoding, video-understanding, document-understanding]
zotero_collection: 多模态/QWen/MML
image_source: local
created: 2024-09-18
---

# 论文笔记：Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Alibaba Group, Qwen Team |
| 日期 | September 2024 |
| 项目主页 | https://github.com/QwenLM/Qwen2-VL |
| 对比基线 | [[GPT-4o]], [[Claude 3.5 Sonnet]], [[LLaVA|LLaVA-NeXT]], [[InternVL2]] |
| 链接 | [arXiv](https://arxiv.org/abs/2409.12191) / [GitHub](https://github.com/QwenLM/Qwen2-VL) / [HF](https://huggingface.co/Qwen) |

---

## 一句话总结

> Qwen2-VL 通过 Naive Dynamic Resolution 和 M-RoPE 两大创新，使 VLM 能够原生处理任意分辨率图像和视频，72B 版本在多模态基准上达到 GPT-4o 和 Claude 3.5 Sonnet 水平。

---

## 核心贡献

1. **Naive Dynamic Resolution**: 移除 ViT 的绝对位置编码，改用 [[2D-RoPE]]，使视觉编码器能原生处理任意分辨率的图像输入，无需像 LLaVA-NeXT 的 AnyRes 那样分割填充固定网格。
2. **M-RoPE（Multimodal Rotary Position Embedding）**: 将 [[RoPE]] 的位置编码分解为时间（T）、高度（H）、宽度（W）三个子维度，文本使用 1D、图像使用 2D、视频使用 3D 位置编码，实现多模态的统一位置建模。
3. **统一图文视频范式**: 将图像视为 2 帧"微视频"送入 ViT，利用 [[3D Convolution]] 深度可分离卷积实现图像和视频的统一处理管线，支持 20 分钟以上长视频理解。
4. **三阶段训练流水线**: 大规模图文预训练 → 多任务高质量微调 → 指令微调（SFT），逐步提升模型的多模态理解和指令遵循能力。

---

## 问题背景

### 要解决的问题

现有 [[LVLM|大规模视觉语言模型]] 在处理高分辨率、任意长宽比的图像和长视频时面临根本性限制：视觉编码器的固定输入尺寸导致信息损失，位置编码方案无法统一处理文本（1D）、图像（2D）和视频（3D）三种模态。

### 现有方法的局限

- **固定分辨率瓶颈**: 大多数 VLM（如 [[LLaVA]]）将图像统一缩放至固定尺寸（如 336x336），导致细粒度视觉信息（小文字、远端物体）丢失。
- **AnyRes 方案的缺陷**: LLaVA-NeXT 的 AnyRes 将图像切分为多个固定尺寸子图，虽然提升了分辨率，但引入大量 padding 浪费计算，且切分破坏了物体的空间连续性。
- **位置编码割裂**: 文本使用 1D 位置编码，图像使用 2D 位置编码，视频使用 3D 位置编码，三者未统一，多模态之间的空间对应关系难以学习。
- **视频处理低效**: 现有方法通常将视频帧独立编码后拼接，帧间时序建模弱，且无法处理长视频。

### 本文的动机

"任何分辨率的世界都不应被固定尺寸的窗口所限制。" Qwen2-VL 的核心理念是从底层架构上消除分辨率限制，让模型能够自由感知任意分辨率的视觉世界。通过 Naive Dynamic Resolution 和 M-RoPE 两项底层创新，Qwen2-VL 无需复杂的预处理即可原生处理各种尺寸的输入。

---

## 方法详解

### 模型架构

Qwen2-VL 采用经典的 **Vision Encoder + MLP Projector + LLM** 三组件架构，提供 2B / 7B / 72B 三个规模：

- **输入处理**: 图像/视频帧经过动态分辨率调整后送入 [[Vision Transformer|ViT]]
- **Vision Encoder**: 统一的 ViT（~675M 参数），使用 [[2D-RoPE]] 替代原始绝对位置编码，三种规模共享同一 ViT
- **Vision-Language Merger**: [[MLP]] 投影器，先将相邻 $2 \times 2$ patch 特征拼接压缩（减少 4x token 数量），再映射到 LLM 文本嵌入维度
- **LLM Backbone**: [[Qwen2]] 系列（2B / 7B / 72B），将标准 1D [[RoPE]] 替换为 [[MRoPE]]
- **输出**: 自回归生成文本回复，支持多模态交错的输入输出

| 配置项 | Qwen2-VL-2B | Qwen2-VL-7B | Qwen2-VL-72B |
|--------|------------|------------|-------------|
| ViT 参数 | ~675M | ~675M | ~675M |
| LLM 参数 | ~2B | ~7B | ~72B |
| ViT Hidden Size | 1280 | 1280 | 1280 |
| ViT Layers | 32 | 32 | 32 |
| ViT Num Heads | 16 | 16 | 16 |
| LLM Hidden Size | 2048 | 4096 | 8192 |
| LLM Layers | 24 | 28 | 80 |
| LLM Num Heads | 16 | 28 | 64 |
| Patch Size | 14 | 14 | 14 |

### 核心模块

#### 模块1: Naive Dynamic Resolution（原生动态分辨率）

**设计动机**: 绕过 LLaVA-NeXT AnyRes 的"切分-填充"范式，让 ViT 原生支持变长序列输入，实现真正的任意分辨率感知。

**具体实现**:
- **位置编码替换**: 移除 ViT 原始的可学习绝对位置编码，改用 [[2D-RoPE]]，使位置编码成为相对位置的函数，天然支持变长序列
- **动态缩放**: 输入图像缩放至高度和宽度均为 patch_size（14）的整数倍，无需 padding 即可直接输入 ViT
- **Token 压缩**: ViT 输出后，将相邻 $2 \times 2$ 的 patch 特征拼接后经 MLP 投影，将 token 数量减少 4 倍
- **推理自适应**: 推理时可根据输入图像的实际分辨率动态调整 token 数量，高分辨率产生更多 token，低分辨率产生更少 token

**与 AnyRes 的对比**:
- AnyRes: 切分成固定子图 → 独立编码 → 拼接 → 大量 padding tokens 浪费
- Naive Dynamic Resolution: 整图输入 → ViT 原生处理 → 自适应 token 数 → 无浪费

#### 模块2: M-RoPE（多模态旋转位置编码）

**设计动机**: 统一文本（1D序列）、图像（2D网格）、视频（3D帧-高-宽）三种模态的位置编码，使模型能同时理解空间位置和时序关系。

**具体实现**:
- 将 query/key 向量的隐藏维度等分为 3 个 chunk，分别对应时间（T）、高度（H）、宽度（W）三个维度
- 每个 chunk 应用对应维度的 1D [[RoPE]]：
  - **T chunk**: 文本位置索引 / 视频帧索引 → 控制"在序列的哪一步 / 第几帧"
  - **H chunk**: 垂直空间坐标 → 控制"在第几行"
  - **W chunk**: 水平空间坐标 → 控制"在第几列"
- **文本 tokens**: 仅 T chunk 有效（退化为标准 1D RoPE）；H、W chunk 的 position ID 设为相同常数
- **图像 tokens**: H + W chunk 有效（实现 [[2D-RoPE]]）；T chunk position ID 设为常数
- **视频 tokens**: T + H + W 全部有效（实现 3D RoPE）
- 应用于自注意力层的每个 attention head

#### 模块3: 统一的图像与视频处理

**设计动机**: 让图像和视频共享同一 ViT 处理管线，避免为视频额外设计复杂的编码模块。

**具体实现**:
- **视频帧采样**: 对视频进行均匀帧采样（可支持 20+ 分钟长视频）
- **3D 深度可分离卷积**: 在 ViT 的部分层中引入 [[3D Convolution|3D 深度可分离卷积]]，用 $1 \times 3 \times 3$ 卷积核处理时序信息
- **图像作为微视频**: 将单张图像复制为 2 帧相同的"微视频"输入 ViT——3D 卷积在时间维度上处理两个相同的帧，等效退化为 2D 卷积
- **统一 token 化**: 无论图像还是视频，输出均为统一格式的视觉 token 序列，供 LLM 消费

---

## 关键公式

### 公式1: [[MRoPE|M-RoPE 位置编码分解]]

$$
\begin{aligned}
q_{t,h,w} &= \text{RoPE}_T(q, t) \oplus \text{RoPE}_H(q, h) \oplus \text{RoPE}_W(q, w) \\
k_{t,h,w} &= \text{RoPE}_T(k, t) \oplus \text{RoPE}_H(k, h) \oplus \text{RoPE}_W(k, w)
\end{aligned}
$$

**含义**: 将 query 和 key 向量按维度三等分，每部分分别用对应维度（T/H/W）的位置索引进行 RoPE 旋转变换后再拼接。

**符号说明**:
- $q, k$: attention 计算中的 query 和 key 向量
- $t, h, w$: 时间帧索引、空间高度索引、空间宽度索引
- $\text{RoPE}_D(\cdot, pos)$: 在维度 D 上应用位置 pos 的 RoPE 旋转
- $\oplus$: 向量拼接

### 公式2: [[RoPE|标准 RoPE 旋转]]

$$
\text{RoPE}(x, m) = \begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \end{bmatrix} \odot \begin{bmatrix} \cos(m\theta_0) \\ \cos(m\theta_1) \\ \cos(m\theta_2) \\ \cos(m\theta_3) \end{bmatrix} + \begin{bmatrix} -x_1 \\ x_0 \\ -x_3 \\ x_2 \end{bmatrix} \odot \begin{bmatrix} \sin(m\theta_0) \\ \sin(m\theta_1) \\ \sin(m\theta_2) \\ \sin(m\theta_3) \end{bmatrix}
$$

**含义**: 对输入向量按相邻维度配对，在每对上进行位置 $m$ 的旋转变换，使 attention 分数含有相对位置信息。

**符号说明**:
- $x$: 输入向量
- $m$: 位置索引
- $\theta_i = 10000^{-2i/d}$: 旋转频率（随维度递减）
- $\odot$: 逐元素乘

### 公式3: [[Attention|自注意力机制]]

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V
$$

**含义**: 标准缩放点积自注意力，$Q、K$ 经过 M-RoPE 变换后计算注意力权重。

**符号说明**:
- $Q, K, V$: 分别由输入经线性投影得到
- $d_k$: key 的维度，用于缩放防止梯度消失
- $M$: 因果掩码（causal mask），视觉 tokens 间可全互注意

### 公式4: [[Cross-Entropy Loss|训练损失函数]]

$$
\mathcal{L} = -\sum_{i=1}^{N} \log P_{\theta}(y_i | v, x, y_{<i})
$$

**含义**: 标准自回归语言建模损失，模型在给定视觉输入 $v$ 和文本上下文 $x, y_{<i}$ 的条件下预测下一个 token。

**符号说明**:
- $\theta$: 模型参数
- $v$: 视觉 token 序列（来自 ViT + MLP 投影）
- $x$: 指令/问题文本
- $y_i$: 第 $i$ 个回答 token
- $N$: 回答 token 总数

### 公式5: Token 压缩

$$
v'_i = \text{MLP}\left([v_{2i,2j}; v_{2i+1,2j}; v_{2i,2j+1}; v_{2i+1,2j+1}]\right)
$$

**含义**: 将 ViT 输出的 $2 \times 2$ 相邻 patch 特征拼接后经 MLP 投影，将视觉 token 数量压缩至 $1/4$。

**符号说明**:
- $v_{a,b}$: 空间位置 $(a, b)$ 处 patch 的特征向量
- $[\cdot;\cdot;\cdot;\cdot]$: 4 个向量沿维度拼接
- $\text{MLP}$: 从 $4 \times d_{vit}$ 维度映射到 $d_{llm}$ 维度的两层 MLP

---

## 关键图表

### Figure 1: 模型架构总览

![[Qwen2-VL_fig1_overview.png]]

**说明**: Qwen2-VL 的整体架构。输入图像/视频帧经 Naive Dynamic Resolution 缩放后送入共享 [[Vision Transformer|ViT]]（使用 [[2D-RoPE]]），输出经 Token 压缩（$2 \times 2$ 拼接 + MLP）后与文本 tokens 一起输入 [[Qwen2]] LLM。LLM 中使用 [[MRoPE]] 替换标准 1D RoPE，统一多模态位置编码。

### Figure 2: M-RoPE 原理示意

![[Qwen2-VL_fig2_mrope.png]]

**说明**: [[MRoPE]] 将 query/key 向量按维度分为三部分：时间（T, 蓝色）控制帧索引和文本位置，高度（H, 绿色）控制垂直坐标，宽度（W, 红色）控制水平坐标。文本仅用 T，图像用 H+W，视频用 T+H+W。三部分各维度内独立执行 RoPE 旋转，最后拼接为统一的位置感知向量。

### Figure 3: Naive Dynamic Resolution 与其他方法的对比

![[Qwen2-VL_fig3_dynamic_resolution.png]]

**说明**: 对比三种分辨率处理方式：(a) 固定分辨率缩放（如 LLaVA）；(b) AnyRes 网格切分（如 LLaVA-NeXT，产生大量 padding）；(c) Qwen2-VL 的 Naive Dynamic Resolution（整图输入，自适应 token 数，无浪费）。

### Figure 4: 定性结果展示

![[Qwen2-VL_fig4_qualitative.png]]

**说明**: Qwen2-VL 在各任务上的定性示例，包括细粒度 OCR、复杂图表理解、多图像推理和长视频理解。

### Table 1: 通用多模态基准评测

| Model | MMBench (dev) | MMVet | MMMU (val) | MathVista | MME | MMBench-CN |
|-------|--------------|-------|------------|-----------|-----|------------|
| LLaVA-NeXT-34B | 79.3 | 51.1 | 48.8 | 46.5 | 2009 | - |
| InternVL2-76B | 85.2 | 62.4 | 55.2 | 63.7 | 2252 | - |
| GPT-4o | 85.3 | 69.2 | 56.9 | 56.7 | 2310 | - |
| Claude 3.5 Sonnet | 84.5 | 66.0 | 55.3 | 61.7 | - | - |
| Qwen2-VL-2B | 74.8 | 47.2 | 42.5 | 44.8 | 1872 | 72.3 |
| Qwen2-VL-7B | 83.0 | 60.2 | 52.3 | 59.8 | 2221 | 80.5 |
| **Qwen2-VL-72B** | **86.5** | **64.9** | **56.6** | **66.7** | **2331** | **83.8** |

**说明**: Qwen2-VL-72B 在 MMBench、MathVista、MME 上超越 GPT-4o，MMMU 上与 GPT-4o 持平。即使 2B 小模型也在多个指标上展现竞争力。

### Table 2: 文档与 OCR 能力

| Model | DocVQA (test) | InfoVQA (test) | ChartQA (test) | TextVQA (val) | OCRBench |
|-------|--------------|----------------|----------------|---------------|----------|
| GPT-4o | 92.8 | 81.2 | 85.7 | - | 736 |
| Claude 3.5 Sonnet | 95.2 | - | 85.3 | - | - |
| InternVL2-76B | 94.1 | 79.2 | 83.9 | 81.4 | 841 |
| Qwen2-VL-2B | 90.1 | 68.9 | 73.9 | 79.1 | 756 |
| Qwen2-VL-7B | 94.5 | 76.8 | 82.8 | 81.5 | 832 |
| **Qwen2-VL-72B** | **96.5** | **82.0** | **83.7** | **82.3** | **869** |

**说明**: Qwen2-VL-72B 在 DocVQA（96.5%）、InfoVQA（82.0%）上超越 GPT-4o 和 Claude 3.5 Sonnet，OCR 能力达到当时最强。

### Table 3: 视频理解能力

| Model | MVBench | Video-MME (w/o sub) | EgoSchema | PerceptionTest |
|-------|---------|---------------------|-----------|----------------|
| GPT-4o | 52.1 | 63.8 | 72.2 | - |
| InternVL2-76B | 67.5 | 61.5 | 69.8 | 57.0 |
| Qwen2-VL-2B | 59.3 | 52.1 | 58.6 | 48.1 |
| Qwen2-VL-7B | 67.1 | 62.4 | 67.2 | 55.8 |
| **Qwen2-VL-72B** | **69.6** | **67.2** | **71.2** | **58.9** |

**说明**: Qwen2-VL-72B 在视频理解基准上整体领先，支持最长 20+ 分钟视频输入，MVBench 和 Video-MME 上均超越 GPT-4o。

### Table 4: 消融实验 — 动态分辨率

| 配置 | MMBench | DocVQA | ChartQA | 说明 |
|------|---------|--------|---------|------|
| Fixed 448x448 | 78.2 | 87.1 | 76.4 | 固定分辨率（无动态分辨率） |
| + Dynamic Res (672) | 81.5 | 92.3 | 80.8 | 启用动态分辨率 |
| + Dynamic Res (larger) | 82.8 | 93.9 | 82.3 | 增大最大分辨率限制 |
| Full (Qwen2-VL-7B) | **83.0** | **94.5** | **82.8** | 完整模型（含 M-RoPE） |

**关键发现**: 动态分辨率对文档/OCR 任务提升最显著（DocVQA +7.4%），因为这类任务需要保留高分辨率下的文字细节。

### Table 5: 消融实验 — M-RoPE vs 标准 RoPE

| 配置 | MMBench | Video-MME | MVBench | 说明 |
|------|---------|-----------|---------|------|
| 1D RoPE | 80.1 | 60.3 | 63.2 | 文本和视觉均用 1D RoPE |
| Separate 1D + 2D | 81.7 | 62.1 | 65.5 | 文本 1D + 图像 2D（分离式） |
| M-RoPE | **83.0** | **62.4** | **67.1** | 统一的三维分解（Qwen2-VL-7B） |

**关键发现**: M-RoPE 对视频任务提升最显著（MVBench +3.9% vs 1D RoPE），证明三维位置分解有效捕获了时空信息。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 网络爬取图文对 | 数十亿 | 大规模、噪声较高 | Stage 1 预训练 |
| 高质量多任务数据 | 数千万 | OCR、定位、文档、图表、视频 | Stage 2 多任务训练 |
| 交错的图文-文本数据 | 数亿 tokens | 图文交错、多图上下文 | Stage 2 多任务训练 |
| 多模态指令数据 | 数百万 | ChatML 格式、多轮对话 | Stage 3 指令微调 |
| 纯文本指令数据 | 数百万 | 保持语言能力 | Stage 3 指令微调 |

### 实现细节

- **Vision Encoder**: 统一 ViT（~675M），patch_size=14，使用 [[2D-RoPE]]
- **Vision-Language Merger**: 两层 [[MLP]] 投影器，先做 $2 \times 2$ token 压缩
- **LLM**: [[Qwen2]] 系列（2B / 7B / 72B），替换为 [[MRoPE]]
- **优化器**: [[Adam|AdamW]]，$\beta_1=0.9, \beta_2=0.999$
- **学习率**: Stage 1: $2 \times 10^{-4}$, Stage 2: $1 \times 10^{-4}$, Stage 3: $2 \times 10^{-5}$
- **Batch Size**: Stage 1: 全局 4096, Stage 2/3: 全局 512-1024
- **序列长度**: 最大 32768 tokens
- **硬件**: A100 GPU 集群训练
- **Position Encoding**: M-RoPE 应用于所有 attention layers

### 可视化结果

Qwen2-VL-72B 展现出以下关键能力：
- **细粒度 OCR**: 能正确识别图片中非常小的文字（街道标识、菜单细节）
- **复杂图表理解**: 准确解读包含子图的复杂学术图表，提取关键数值
- **多图像推理**: 跨多张图片进行对比和逻辑推理（如不同商品的比价）
- **长视频理解**: 在 20+ 分钟视频中定位特定事件，理解时间线
- **GUI Agent**: 理解手机/电脑屏幕截图，执行多步操作指令

---

## 批判性思考

### 优点
1. **从底层架构解决问题**: Naive Dynamic Resolution 从 ViT 位置编码入手，而非在预处理层打补丁（如 AnyRes），设计简洁优雅
2. **M-RoPE 是真正的多模态统一**: 将 1D/2D/3D 位置编码统一为一个框架，而非三个独立的子系统，数学上优美且实用
3. **工程设计务实**: 图像和视频共享 ViT 加轻量 3D 卷积，没有引入重量级的视频专用编码器，成本可控
4. **小模型也有效**: 2B 模型在多个基准上有竞争力，证明核心方法不依赖大参数规模
5. **长视频能力领先**: 20+ 分钟视频理解在 2024 年发布时处于领先地位

### 局限性
1. **ViT 计算成本**: 高分辨率下视觉 token 数量大幅增加（如 $1024 \times 1024$ 产生 $73 \times 73 \approx 5300$ tokens），虽然比 AnyRes 高效，但在超高分场景下仍可能成为瓶颈
2. **M-RoPE 维度固定分配**: 将 hidden dimension 均分为 3 份的逻辑较为刚性，不同任务对 T/H/W 维度的需求可能不同
3. **无 DPO/RLHF 阶段**: 训练仅到 SFT，未引入偏好优化，可能影响安全性和对齐质量
4. **缺乏多图像位置关系**: 多图输入时，各图之间的 M-RoPE 位置关系定义不够明确

### 潜在改进方向
1. **自适应维度分配**: 让 T/H/W 三个 chunk 的维度比例可学或任务自适应
2. **视觉 token 动态压缩**: 在 LLM 推理时对不重要的视觉 token 进行剪枝（Qwen2.5-VL 已部分解决）
3. **增加 RLHF/DPO**: 偏好优化可提升对齐质量和有用性
4. **提升多图空间关系**: 明确建模多张图片之间的 M-RoPE 相对位置

### 可复现性评估
- [x] 代码开源（Apache 2.0）
- [x] 预训练模型（HuggingFace 全尺寸）
- [x] 训练细节完整（论文 + 技术博客）
- [ ] 数据集可获取（预训练数据未公开）

---

## 关联笔记

### 基于
- [[Qwen2]]: LLM 骨干网络
- [[Vision Transformer|ViT]]: 视觉编码器基础架构
- [[RoPE]]: 位置编码基础方案
- [[Qwen-VL]]: 前代模型

### 对比
- [[GPT-4o]]: 商业闭源最强多模态基准
- [[Claude 3.5 Sonnet]]: 多模态理解与文档解析的竞品
- [[LLaVA|LLaVA-NeXT]]: AnyRes 动态分辨率方案的主要对比对象
- [[InternVL2]]: 开源最强 VLM 对比基线

### 方法相关
- [[MRoPE]]: 核心位置编码方案
- [[2D-RoPE]]: 图像维度的 RoPE 扩展
- [[3D Convolution]]: 视频时序建模
- [[Naive Dynamic Resolution]]: 原生动态分辨率方案
- [[Vision-Language Merger]]: ViT 到 LLM 的特征投影

### 后续工作
- [[Qwen2.5-VL]]: 下一代模型，引入 Window Attention、Absolute Time M-RoPE、Dynamic FPS
- [[Qwen3-VL]]: 第三代模型

---

## 速查卡片

> [!summary] Qwen2-VL: Enhancing VLMs at Any Resolution
> - **核心**: Naive Dynamic Resolution + M-RoPE 让 VLM 原生处理任意分辨率图像和长视频
> - **方法**: ViT (2D-RoPE) → Token 压缩 → LLM (M-RoPE)，图像视为 2 帧微视频
> - **结果**: 72B 在 MMBench/MathVista/DocVQA 上超越 GPT-4o，长视频理解领先
> - **代码**: https://github.com/QwenLM/Qwen2-VL

---

*笔记创建时间: 2026-05-18*
