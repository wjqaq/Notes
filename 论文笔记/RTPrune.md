---
title: "RTPrune: Reading-Twice Inspired Token Pruning for Efficient DeepSeek-OCR Inference"
method_name: "RTPrune"
authors: [Ben Wan, Yan Feng, Zihan Tang, Weizhe Huang, Yuting Zeng, Jia Wang, Tongxuan Liu]
year: 2026
venue: ICML 2026
tags: [token-pruning, ocr, deepseek-ocr, optimal-transport, inference-acceleration, visual-text-compression]
zotero_collection: ""
image_source: local
arxiv_html: https://arxiv.org/html/2605.00392v2
created: 2026-05-21
---
# 论文笔记：RTPrune

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | JD.com, Shanghai Jiao Tong University, USTC, Tsinghua University |
| 日期 | May 2026 |
| 项目主页 | N/A（代码已声明 release） |
| 对比基线 | [[FastV]], [[DivPrune]], [[CDPruner]], [[SparseVLM]], [[VisionZip]], [[FitPrune]], [[DART]], [[NUWA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.00392) / Code: 待发布 |

---

## 一句话总结

> 发现 DeepSeek-OCR 的 LLM 解码存在"读两遍"注意力模式，据此设计两阶段 token 剪枝：高范数 token 保留 + 最优传输合并，配合动态剪枝率，实现 99.47% 精度下 1.23x 推理加速。

---

## 核心贡献

1. **首次揭示 DeepSeek-OCR 的 Reading-Twice 注意力模式**: 浅层关注高 [[ℓ2-Norm Feature Selection|ℓ2-norm]] token（文本/结构信息），深层重新分配注意力到剩余 token
2. **两阶段 Token 剪枝方法 (RTPrune)**: Stage 1 基于 ℓ2-norm 选择主导 token，Stage 2 通过 [[Optimal Transport]] + [[Sinkhorn Algorithm]] 合并剩余 token
3. **动态剪枝率策略**: 联合 token 间特征相似度 + Sobel 边缘检测估计文本密度，自适应调节剪枝强度
4. **跨模型泛化验证**: 不仅在 DeepSeek-OCR 全系列有效，在 DeepSeek-OCR2、LightOnOCR、GLM-OCR 上也保持 92%+ 原始性能

---

## 问题背景

### 要解决的问题

[[DeepSeek-OCR]] 通过视觉-文本压缩将文本编码为视觉 token，实现高达 20x 的上下文压缩。但视觉 token 仍存在冗余——部分 token 对应背景区域或重复结构模式，对文本重建贡献极小，存在进一步压缩的空间。

### 现有方法的局限

现有 VLM [[Token Pruning]] 方法直接应用于 DeepSeek-OCR 时**甚至不如随机剪枝**：

| 根本原因 | 具体表现 |
|----------|----------|
| **视觉-语言对齐弱化** | DeepSeek-OCR 重新训练视觉编码器，破坏了预训练的语义-视觉耦合，CLIP-ViT-B/16 的多模态对齐本就更弱 |
| **OCR 的密集性需求** | VQA 只需关注问题相关区域，OCR 要求**穷举式高保真重建**所有文本和布局细节 |
| **Prompt 无区分力** | OCR 的 prompt 固定（`Convert the document to markdown`），文本相关性方法无法区分重要区域 |
| **LLM 层间注意力分化** | 不同 LLM 层关注不同子集的视觉 token，导致基于 LLM 注意力的剪枝策略失效 |

### 本文的动机

由于 DeepSeek-OCR 中视觉编码器和 LLM 是**联合优化**的，作者假设：LLM 中注意力更高的视觉 token，其编码器输出的 ℓ2-norm 也更大。通过分析验证了这一假设，并发现了"读两遍"的注意力轨迹，直接催生了 RTPrune 的设计。

---

## 方法详解

### 模型架构

RTPrune 采用 **训练无关、即插即用** 的 token 剪枝架构，在 DeepSeek-OCR 的视觉编码器输出后、LLM prefill 前执行：

- **输入**: 视觉 token 序列 $\mathbf{T}_v = g(f_v(\mathbf{I})) \in \mathbb{R}^{N \times D}$
- **Stage 1 — [[ℓ2-Norm Feature Selection|主导 Token 选择]]**: 基于 ℓ2-norm 选择 top-M token 作为 $\mathcal{T}_{\text{kept}}$
- **Stage 2 — [[Optimal Transport|最优传输]] Token 合并**: 将 $\mathcal{T}_{\text{prop}}$ 通过 [[Sinkhorn Algorithm]] 匹配并合并到 $\mathcal{T}_{\text{kept}}$
- **动态剪枝率**: 联合 [[Inter-Token Similarity]] + [[Sobel Operator|Sobel 边缘检测]] 文本密度
- **输出**: 合并后的精简 token 序列，送入 LLM
- **总参数**: 0（训练无关，仅推理时使用）

### 核心模块

#### 模块1: 主导 Token 选择 (Dominant Token Selection)

**设计动机**: 利用浅层 LLM 关注高 ℓ2-norm token 的行为，用 ℓ2-norm 作为 token 重要性代理指标。

**具体实现**:
- 计算每个视觉 token 的 ℓ2-norm：$C_k = \|\mathbf{T}_v^k\|_2$
- 给定目标剪枝率 $r$，保留 $M = N(1-r)$ 个 token
- 选 top-M 高 norm token 构成 $\mathcal{T}_{\text{kept}}$，其余进入 $\mathcal{T}_{\text{prop}}$
- Deeper layers attend to remaining tokens including the rest high-norm ones for complementary cues

#### 模块2: 最优传输 Token 合并 (Optimal Token Merging)

**设计动机**: 模拟深层 LLM 重新分配注意力的行为，通过最优传输将待剪 token 的信息合并到保留 token，而非简单丢弃。

**具体实现**:
- 计算 $\mathcal{T}_{\text{kept}}$ 与 $\mathcal{T}_{\text{prop}}$ 间的余弦相似度矩阵 $\mathbf{S}$
- 引入 **dustbin** 机制（借鉴 [[SuperGlue]]）：增广一行一列，填充固定参数 $z=0.2$
- 通过 [[Sinkhorn Algorithm]]（T=100 迭代，log-space）求解最优分配矩阵 $\mathbf{P}$
- 落入 dustbin 的匹配意味着该 token 被直接丢弃
- 执行加权合并：$\mathbf{T}_v^i = \mathbf{T}_v^i + \alpha \sum_j \mathbf{P}_{i,j} \cdot \mathbf{T}_v^j$，其中 $\alpha=0.1$

#### 模块3: 动态剪枝率 (Dynamic Pruning Ratio)

**设计动机**: OCR 任务信息密度高，固定剪枝率无法兼顾不同图像内容。

**具体实现**:
- **Token 间相似度** $\phi$：衡量全局特征冗余度 — 相似度越高，可剪越多

$$\phi = \frac{2}{N(N-1)} \sum_{i<j} \frac{\mathbf{T}_v^i \cdot \mathbf{T}_v^j}{\|\mathbf{T}_v^i\|_2 \cdot \|\mathbf{T}_v^j\|_2}$$

- **文本密度** $\rho$：用 [[Sobel Operator]] 检测每 patch 边缘像素占比 — 密度越低（空白/背景多），可剪越多

$$\rho_k = \frac{1}{h \times w} \sum_{i,j} \mathbb{I}(G(i,j) \geq \tau)$$

- 综合动态剪枝率：$r_{\text{dyn}} = f_{\text{normalize}}(\phi)(1 - \rho)$

---

## 关键公式

### 公式1: [[ℓ2-Norm Feature Selection|Token ℓ2-Norm 重要性]]

$$
C_k = \|\mathbf{T}_v^k\|_2
$$

**含义**: 用视觉 token 的 ℓ2-norm 作为其信息重要性度量，norm 越大表示包含越多文本/结构信息。

**符号说明**:
- $\mathbf{T}_v^k \in \mathbb{R}^D$: 第 $k$ 个视觉 token 的 embedding
- $C_k$: token 重要性分数

### 公式2: [[Optimal Transport|最优传输匹配问题]]

$$
\max \sum_{i,j} \mathbf{S}_{i,j} \cdot \mathbf{P}_{i,j}
\quad \text{s.t.} \quad \mathbf{P}\mathbf{1}_{N-M} = \mathbf{1}_M,\; \mathbf{P}^{\text{T}}\mathbf{1}_M = \mathbf{1}_{N-M}
$$

**含义**: 在满足一对一匹配约束下，最大化 kept token 与 prop token 之间的总相似度得分。

**符号说明**:
- $\mathbf{P} \in \mathbb{R}^{M \times (N-M)}$: 分配矩阵
- $\mathbf{S}$: 余弦相似度得分矩阵
- $M$: 保留 token 数，$N-M$: 待剪 token 数

### 公式3: 余弦相似度得分

$$
\mathbf{S}_{i,j} = \frac{\mathbf{T}_v^i \cdot \mathbf{T}_v^j}{\|\mathbf{T}_v^i\|_2 \cdot \|\mathbf{T}_v^j\|_2}, \quad \forall (\mathbf{T}_v^i, \mathbf{T}_v^j) \in \mathcal{T}_{\text{kept}} \times \mathcal{T}_{\text{prop}}
$$

**含义**: 衡量每个 kept token 与每个 prop token 之间的特征方向相似度。

### 公式4: [[Sinkhorn Algorithm|Dustbin 增广]]

$$
\bar{\mathbf{S}}_{i, N-M+1} = \bar{\mathbf{S}}_{M+1, j} = \bar{\mathbf{S}}_{M+1, N-M+1} = z \in \mathbb{R}
$$

**含义**: 为得分矩阵增加 dustbin 行和列，允许 token 不与任何 target 匹配而被丢弃。

**符号说明**:
- $z = 0.2$: dustbin 固定相似度参数
- $\bar{\mathbf{S}} \in \mathbb{R}^{(M+1) \times (N-M+1)}$: 增广得分矩阵

### 公式5: 增广约束

$$
\bar{\mathbf{P}}\mathbf{1}_{N-M+1} = \boldsymbol{a}, \quad \bar{\mathbf{P}}^{\text{T}}\mathbf{1}_{M+1} = \boldsymbol{b}
$$

**含义**: 增广后的边际约束，dustbin 可匹配任意数量的 token。

**符号说明**:
- $\boldsymbol{a} = [\mathbf{1}_M^{\text{T}} \quad N-M]^{\text{T}}$: kept token 期望匹配数
- $\boldsymbol{b} = [\mathbf{1}_{N-M}^{\text{T}} \quad M]^{\text{T}}$: prop token 期望匹配数

### 公式6: Token 合并

$$
\mathbf{T}_v^i = \mathbf{T}_v^i + \alpha \sum_j \mathbf{P}_{i,j} \cdot \mathbf{T}_v^j, \quad \forall (\mathbf{T}_v^i, \mathbf{T}_v^j) \in \mathcal{T}_{\text{kept}} \times \mathcal{T}_{\text{prop}}
$$

**含义**: 将待剪 token 按最优分配权重合并到保留 token 中，实现信息聚合。

**符号说明**:
- $\alpha = 0.1$: 合并强度超参数
- $\mathbf{P}_{i,j}$: Sinkhorn 求解后的最优分配权重

### 公式7: 动态剪枝率

$$
r_{\text{dyn}} = f_{\text{normalize}}(\phi)(1 - \rho)
$$

**含义**: 特征冗余度越高（$\phi$ 大）+ 文本密度越低（$\rho$ 小）→ 剪枝率越高。

### 公式8: [[Sinkhorn Algorithm|Sinkhorn 对数空间迭代]]

$$
\begin{aligned}
\mathbf{u}^{(t+1)} &= \log \boldsymbol{\mu} - \text{LogSumExp}(\mathbf{Z} + \mathbf{v}^{(t)\top}) \\
\mathbf{v}^{(t+1)} &= \log \boldsymbol{\nu} - \text{LogSumExp}(\mathbf{Z}^{\top} + \mathbf{u}^{(t+1)\top})
\end{aligned}
$$

**含义**: 在 log-space 迭代更新对偶变量 $\mathbf{u}, \mathbf{v}$，经 T=100 轮收敛后 $\mathbf{P} = \exp(\mathbf{Z}_{\text{final}})_{:-1,:-1}$。

**符号说明**:
- $\mathbf{Z}$: log-增广得分矩阵
- $\boldsymbol{\mu}, \boldsymbol{\nu}$: 边际分布（含 dustbin）
- $\text{LogSumExp}$: 数值稳定的 log-sum-exp 操作

### 公式9: 文本密度 $\rho_k$

$$
\rho_k = \frac{1}{h \times w} \sum_{i,j} \mathbb{I}(G(i,j) \geq \tau)
$$

**含义**: 每个 token patch 中梯度幅值超过阈值 $\tau=0.2$ 的"活跃边缘像素"占比。

**符号说明**:
- $G(i,j) = \sqrt{G_x(i,j)^2 + G_y(i,j)^2}$: Sobel 梯度幅值
- $\tau$: 边缘检测阈值

---

## 关键图表

### Figure 1: Performance and Efficiency

![[RTPrune_fig1_left.png]]

![[RTPrune_fig1_right.png]]

**说明**: (左) RTPrune 在 olmOCR-Bench 上以 84% visual token 保留 97.88% 精度，显著优于其他剪枝方法。(右) 在 OmniDocBench 上减少 15.29% GFLOPs 和 18.90% prefill 时间，同时保持 99.47% 精度。

### Figure 2: Overview of RTPrune

![[RTPrune_fig2_overview.png]]

**说明**: RTPrune 整体架构。先通过 token 间相似度和图像文本密度动态确定剪枝率，然后分两阶段：Stage 1 基于 ℓ2-norm 选择主导 token，Stage 2 通过 [[Optimal Transport]] 合并剩余 token 信息。训练无关、模型无关。

### Figure 3: Comparison of Different Token Pruning Methods

![[RTPrune_fig3_comparison.png]]

**说明**: 不同 token 剪枝方法在 DeepSeek-OCR-Base 上的可视化对比。蓝色高亮为被剪 token，红色文本为与 ground truth 的差异。基于原图/注意力/文本相关性/相似度的方法均无法准确生成文本，只有 RTPrune 精准捕捉了包含关键文本信息的 token。

### Figure 4: Top-K Intersection Ratio (TIR)

![[RTPrune_fig4_tir.png]]

**说明**: 高 ℓ2-norm token 与 LLM 高注意力 token 的交集比。小提琴图展示逐层 TIR，折线图展示累计 TIR。逐层 TIR 先升后降，累计 TIR 单调递增——浅层快速上升（64.4% max at top-3/8），深层趋缓（最终 88.71%），揭示了"读两遍"的注意力轨迹。

### Figure 5: Performance on Ocean-OCR Benchmark

![[RTPrune_fig5_ocean.png]]

**说明**: RTPrune + 动态剪枝策略在 Ocean-OCR benchmark (Gundam 模式) 上各 OCR 能力维度的表现。RTPrune 不仅大幅超越其他剪枝方法，甚至在某些维度超越未剪枝 baseline，验证了方法的鲁棒泛化能力。

### Figure 6: Visual Token Redundancy Visualization

![[RTPrune_fig6_redundancy.png]]

**说明**: DeepSeek-OCR-Base 上单 token 剪枝实验。蓝色为被剪 token，红色为保留 token。超过 95% 的视觉 token 被单独移除时模型仍能生成相同正确输出，证明 DeepSeek-OCR 视觉 token 存在大量冗余。

### Figure 7: Textual Relevance Visualization in Different CLIP Models

![[RTPrune_fig7_relevance.png]]

**说明**: 不同 CLIP 模型中 prompt 文本与图像 embedding 的相关性可视化。CLIP-ViT-B/16（DeepSeek-OCR 所用）的多模态对齐弱于 CLIP-ViT-L/14-336px，且经过重新训练后对齐进一步衰减，解释了文本相关性剪枝方法失败的根源。

---

## 实验

### Table 1: Pilot Study — 已有方法在 DeepSeek-OCR 上的局限 (25% pruning, OmniDocBench 10% subset, Base)

| Methods | Text ↓ | Formula ↑ (CDM) | Table ↑ (TEDS / TEDS-S) | Order ↓ | Overall |
|---------|--------|-----------------|------------------------|---------|---------|
| **Baseline** | 0.18 | 83.47 | 88.82 / 92.74 | 0.16 | **84.86** |
| Random | 0.52 | 61.72 | 62.02 / 72.80 | 0.30 | 57.24 |
| FastV (ECCV'24) | 0.90 | 5.02 | -0.29 / 2.27 | 0.63 | 4.78 |
| DivPrune (CVPR'25) | 0.36 | 70.67 | 61.14 / 72.33 | 0.25 | 65.30 |
| CDPruner (NeurIPS'25) | 0.47 | 55.74 | 47.39 / 60.92 | 0.31 | 52.01 |

**表格说明**: 所有已有 VLM 剪枝方法在 DeepSeek-OCR 上均不如随机剪枝，常产生遗漏、重复或乱码输出。

### Table 2: OmniDocBench 主实验结果 (4 个 DeepSeek-OCR 变体，固定 25% + 动态剪枝)

**DeepSeek-OCR-Tiny** (64 tokens baseline → 48 tokens):

| Method | Dynamic | #Tokens | Text↓ | Formula↑ | Table↑ / TEDS-S↑ | Order↓ | Overall↑ |
|--------|---------|---------|-------|----------|-------------------|--------|----------|
| Baseline | - | 64 | 0.27 | 70.01 | 62.86 / 70.01 | 0.20 | 68.59 |
| FastV | % | 48 | 0.95 | 2.20 | 0.26 / 6.31 | 0.68 | 2.49 |
| DART | % | 48 | 0.96 | 1.39 | -0.10 / 5.07 | 0.72 | 1.73 |
| **RTPrune** | % | 48 | **0.48** | **50.38** | **40.22 / 50.37** | **0.42** | **47.60** |
| SparseVLM | " | AVG=52 | 0.89 | 9.92 | 7.50 / 6.31 | 0.64 | 7.50 |
| VisionZip | " | AVG=52 | 0.50 | 56.06 | 47.37 / 41.44 | 0.33 | 47.37 |
| **RTPrune** | " | AVG=52 | **0.43** | **57.56** | **52.09 / 53.98** | **0.38** | **52.09** |

**DeepSeek-OCR-Small** (100 → 75):

| Method | Dynamic | #Tokens | Text↓ | Formula↑ | Table↑ / TEDS-S↑ | Order↓ | Overall↑ |
|--------|---------|---------|-------|----------|-------------------|--------|----------|
| Baseline | - | 100 | 0.17 | 79.14 | 76.58 / 82.43 | 0.15 | 79.61 |
| FitPrune | % | 75 | 0.78 | 17.63 | 13.13 / 17.55 | 0.61 | 17.55 |
| NUWA | % | 75 | 0.92 | 4.13 | 2.38 / 5.00 | 0.68 | 5.00 |
| **RTPrune** | % | 75 | **0.37** | **60.74** | **53.71 / 59.08** | **0.33** | **59.08** |
| DivPrune | " | AVG=83 | 0.39 | 68.36 | 58.93 / 59.90 | 0.26 | 58.93 |
| CDPruner | " | AVG=83 | 0.45 | 70.05 | 59.75 / 55.48 | 0.30 | 59.75 |
| **RTPrune** | " | AVG=83 | **0.31** | **67.86** | **64.91 / 63.46** | **0.28** | **64.91** |

**DeepSeek-OCR-Base** (256 → 192):

| Method | Dynamic | #Tokens | Text↓ | Formula↑ | Table↑ / TEDS-S↑ | Order↓ | Overall↑ |
|--------|---------|---------|-------|----------|-------------------|--------|----------|
| Baseline | - | 256 | 0.12 | 81.90 | 84.58 / 88.43 | 0.11 | 84.83 |
| SparseVLM | % | 192 | 0.74 | 27.39 | 21.07 / 24.92 | 0.50 | 24.92 |
| VisionZip | % | 192 | 0.37 | 67.09 | 59.39 / 63.29 | 0.23 | 63.29 |
| **RTPrune** | % | 192 | **0.21** | **77.27** | **75.55 / 77.37** | **0.19** | **77.37** |
| FastV | " | AVG=214 | 0.87 | 7.42 | 3.95 / 8.12 | 0.65 | 8.12 |
| DART | " | AVG=214 | 0.58 | 33.95 | 33.37 / 36.47 | 0.46 | 36.47 |
| **RTPrune** | " | AVG=214 | **0.17** | **79.95** | **81.60 / 81.48** | **0.16** | **81.48** |

**DeepSeek-OCR-Large** (400 → 300):

| Method | Dynamic | #Tokens | Text↓ | Formula↑ | Table↑ / TEDS-S↑ | Order↓ | Overall↑ |
|--------|---------|---------|-------|----------|-------------------|--------|----------|
| Baseline | - | 400 | 0.09 | 81.70 | 83.87 / 85.55 | 0.09 | 85.55 |
| DivPrune | % | 300 | 0.26 | 71.22 | 64.83 / 70.02 | 0.17 | 70.02 |
| CDPruner | % | 300 | 0.32 | 73.60 | 72.16 / 71.19 | 0.20 | 71.19 |
| **RTPrune** | % | 300 | **0.13** | **81.71** | **81.79 / 83.60** | **0.13** | **83.60** |
| FitPrune | " | AVG=337 | 0.30 | 55.61 | 44.17 / 56.63 | 0.24 | 56.63 |
| NUWA | " | AVG=337 | 0.85 | 5.50 | 6.04 / 8.71 | 0.64 | 8.71 |
| **RTPrune** | " | AVG=337 | **0.10** | **82.70** | **82.11 / 85.10** | **0.09** | **85.10** |

**表格说明**: RTPrune 在所有设置下均最优。**DeepSeek-OCR-Large 上 15.75% token 剪枝后保持 99.47% 原始性能**。LLM 内剪枝方法（FitPrune、SparseVLM）因浅层剪枝导致深层关键信息丢失；视觉编码器端剪枝方法（VisionZip、CDPruner）无法准确捕捉完整文本信息。

### Table 3: olmOCR-Bench 实验结果 (DeepSeek-OCR-Base + Large)

**DeepSeek-OCR-Base** (256 → 192):

| Method | Dynamic | AI | Old Scans | Tables | Headers & Footers | Multi Column | Long Tiny Text | Math | Overall |
|--------|---------|-----|-----------|--------|-------------------|--------------|----------------|------|---------|
| Baseline | - | 69.1 | 65.3 | 65.6 | 76.6 | 77.7 | 95.8 | 99.3 | 72.4 |
| FitPrune | % | 12.9 | 5.1 | 4.8 | 30.0 | 25.8 | 87.6 | 70.7 | 34.1 |
| NUWA | % | 0.2 | 0.0 | 0.0 | 27.2 | 17.1 | 92.6 | 97.1 | 23.4 |
| **RTPrune** | % | **59.5** | **31.8** | **20.4** | **65.3** | **62.8** | **98.6** | **98.9** | **57.7** |
| DivPrune | " | 42.2 | 22.3 | 25.8 | 42.8 | 48.0 | 95.1 | 98.9 | 50.1 |
| CDPruner | " | 47.2 | 15.0 | 21.5 | 28.5 | 25.1 | 95.7 | 99.1 | 53.0 |
| **RTPrune** | " | **65.1** | **45.1** | **34.8** | **69.7** | **71.1** | **98.6** | **99.1** | **63.7** |

**DeepSeek-OCR-Large** (400 → 300):

| Method | Dynamic | AI | Old Scans | Tables | Headers & Footers | Multi Column | Long Tiny Text | Math | Overall |
|--------|---------|-----|-----------|--------|-------------------|--------------|----------------|------|---------|
| Baseline | - | 74.7 | 68.2 | 76.9 | 75.8 | 80.2 | 95.8 | 99.9 | 75.5 |
| FastV | % | 1.4 | 0.1 | 0.2 | 3.1 | 2.8 | 82.1 | 84.5 | 25.2 |
| DART | % | 1.2 | 0.1 | 0.1 | 3.4 | 3.0 | 93.6 | 96.4 | 25.6 |
| **RTPrune** | % | **72.8** | **59.0** | **42.1** | **68.6** | **78.2** | **99.0** | **99.5** | **68.3** |
| SparseVLM | " | 16.1 | 6.2 | 17.6 | 6.3 | 12.5 | 87.3 | 99.0 | 32.0 |
| VisionZip | " | 58.0 | 27.7 | 41.2 | 60.9 | 63.4 | 99.0 | 98.6 | 59.7 |
| **RTPrune** | " | **73.7** | **64.6** | **73.8** | **71.8** | **80.5** | **97.1** | **99.5** | **73.9** |

**表格说明**: 已有方法在 Multi Column、Long Tiny Text 等挑战性子集上严重退化。RTPrune 动态剪枝在 Base 上以 84% token 保留 97.88% 精度。

### Table 4: 效率分析 (olmOCR-Bench, DeepSeek-OCR-Base + Large)

**DeepSeek-OCR-Base** (256 → 192 / AVG=213):

| Method | Dynamic | #Visual Tokens | #Total Tokens | GFLOPs | Prefill Time (ms) | Decode Time (ms/tok) | Performance |
|--------|---------|---------------|---------------|--------|-------------------|---------------------|-------------|
| Baseline | - | 256 | 283 | 235.7 | 78.7 | 20.7 | 72.4 |
| FitPrune | % | 192 | 219 | 199.5 | 77.4 (x1.02) | 20.4 | 34.1 |
| NUWA | % | 192 | 219 | 181.5 | 75.4 (x1.04) | 20.4 | 23.4 |
| **RTPrune** | % | **192** | **219** | **181.5** | **72.6 (x1.08)** | **20.4** | **57.7** |
| DivPrune | " | AVG=213 | AVG=240 | 199.2 | 75.1 (x1.05) | 20.5 | 50.1 |
| CDPruner | " | AVG=213 | AVG=240 | 199.2 | 75.1 (x1.05) | 20.5 | 53.0 |
| **RTPrune** | " | **AVG=213** | **AVG=240** | **199.2** | **75.1 (x1.05)** | **20.5** | **63.7** |

**DeepSeek-OCR-Large** (400 → 300 / AVG=336):

| Method | Dynamic | #Visual Tokens | #Total Tokens | GFLOPs | Prefill Time (ms) | Decode Time (ms/tok) | Performance |
|--------|---------|---------------|---------------|--------|-------------------|---------------------|-------------|
| Baseline | - | 400 | 431 | 363.0 | 96.3 | 20.9 | 75.5 |
| FastV | % | 300 | 331 | 290.9 | 92.9 (x1.04) | 20.6 | 25.2 |
| DART | % | 300 | 331 | 290.9 | 92.9 (x1.04) | 20.6 | 25.6 |
| **RTPrune** | % | **300** | **331** | **276.6** | **77.1 (x1.25)** | **20.6** | **68.3** |
| SparseVLM | " | AVG=336 | AVG=363 | 333.5 | 92.4 (x1.04) | 20.7 | 32.0 |
| VisionZip | " | AVG=336 | AVG=367 | 307.5 | 78.1 (x1.23) | 20.7 | 59.7 |
| **RTPrune** | " | **AVG=336** | **AVG=367** | **307.5** | **78.1 (x1.23)** | **20.7** | **73.9** |

**表格说明**: RTPrune 在大模型上 prefill 加速最显著（x1.25 in Base, x1.23 in Large）。LLM 内剪枝和多阶段剪枝因频繁 tensor shape 变化导致硬件吞吐下降，实际加速有限。RTPrune 在 prefill 前完成剪枝，避免此问题。

### Table 5: 跨 OCR 模型泛化 (OmniDocBench)

| Method | #Tokens | Prefill (ms) | Decode (ms/tok) | Text↓ | Formula↑ | Table↑ / TEDS-S↑ | Order↓ | Overall↑ |
|--------|---------|-------------|-----------------|-------|----------|-------------------|--------|----------|
| **DeepSeek-OCR2-Gundam** | | | | | | | | |
| Baseline | 1083 | 59.1 | 16.8 | 0.05 | 90.54 | 86.94 / 91.23 | 0.06 | 90.93 |
| FitPrune | AVG=560 | 58.3 | 15.9 | 0.22 | 65.89 | 62.23 / 66.67 | 0.20 | 68.64 |
| DivPrune | AVG=560 | 57.6 | 16.6 | 0.27 | 65.77 | 57.67 / 68.91 | 0.19 | 65.35 |
| CDPruner | AVG=560 | 57.6 | 16.6 | 0.45 | 50.30 | 39.61 / 52.79 | 0.30 | 48.37 |
| **RTPrune** | **AVG=560** | **57.6** | **16.6** | **0.14** | **85.07** | **80.44 / 85.95** | **0.10** | **83.94** |
| **LightOnOCR** | | | | | | | | |
| Baseline | AVG=1238 | 30.4 | 16.4 | 0.16 | 87.00 | 83.22 / 89.49 | 0.10 | 84.91 |
| FitPrune | AVG=1238 | 27.8 | 16.1 | 0.94 | 6.33 | 4.92 / 7.14 | 0.65 | 5.85 |
| DivPrune | AVG=1238 | 23.4 | 16.0 | 0.29 | 70.17 | 62.89 / 76.76 | 0.19 | 67.95 |
| CDPruner | AVG=1238 | 23.4 | 16.0 | 0.22 | 88.45 | 73.42 / 84.58 | 0.13 | 79.96 |
| **RTPrune** | **AVG=1238** | **23.4** | **16.0** | **0.18** | **85.67** | **81.26 / 87.53** | **0.12** | **83.00** |
| **GLM-OCR** | | | | | | | | |
| Baseline | AVG=2783 | 26.3 | 11.0 | 0.10 | 86.05 | 13.92 / 15.29 | 0.11 | 63.22 |
| FitPrune | AVG=2783 | 18.3 | 10.7 | 0.49 | 57.62 | 13.65 / 15.81 | 0.37 | 40.92 |
| DivPrune | AVG=2783 | 17.1 | 10.7 | 0.57 | 41.92 | 1.82 / 2.47 | 0.46 | 28.95 |
| CDPruner | AVG=2783 | 17.1 | 10.7 | 0.16 | 77.31 | 5.15 / 5.58 | 0.17 | 55.39 |
| **RTPrune** | **AVG=2783** | **17.1** | **10.7** | **0.15** | **81.74** | **12.72 / 13.91** | **0.14** | **59.98** |

**表格说明**: 其他剪枝方法在不同 OCR 模型上性能波动大，RTPrune 在所有架构上稳定保留 92%+ 原始性能。

### Table 6: 消融实验 — 选择度量 + Token 合并

| Methods | Text↓ | Formula↑ | Table↑ / TEDS-S↑ | Order↓ | Overall↑ |
|---------|-------|----------|-------------------|--------|----------|
| Baseline | 0.12 | 81.90 | 84.58 / 88.43 | 0.11 | 84.83 |
| Variance | 0.20 | 76.40 | 78.33 / 83.46 | 0.19 | 78.21 |
| Variance + OTM | 0.19 | 76.91 | 79.51 / 84.49 | 0.18 | 79.20 |
| Entropy | 0.53 | 61.83 | 51.99 / 63.89 | 0.29 | 53.61 |
| Entropy + OTM | 0.53 | 63.40 | 54.09 / 66.43 | 0.29 | 54.96 |
| ℓ2-Norm | 0.18 | 79.00 | 80.18 / 85.32 | 0.17 | 80.69 |
| ℓ2-Norm + GTP-ViT | 0.18 | 79.33 | 80.35 / 85.11 | 0.17 | 80.90 |
| ℓ2-Norm + VisionZip merge | 0.17 | 80.14 | 80.15 / 84.77 | 0.16 | 81.03 |
| ℓ2-Norm + SparseVLM merge | 0.18 | 79.51 | 80.26 / 85.02 | 0.17 | 80.69 |
| ℓ2-Norm + NUWA merge | 0.17 | 79.32 | 80.79 / 85.67 | 0.17 | 80.90 |
| **ℓ2-Norm + OTM (RTPrune)** | **0.17** | **79.95** | **81.60 / 86.74** | **0.16** | **81.48** |

**表格说明**: ℓ2-norm 优于 variance 和 entropy；所有 token 合并方法均提升性能，验证了"读两遍"行为的存在；[[Optimal Transport]] 合并效果最优。

### Table 7: 消融实验 — 动态剪枝率 (Ocean-OCR, DeepSeek-OCR-Base)

**固定剪枝率 (Fixed r)**:

| Methods | Document EN | Document ZH | Scene Text EN | Scene Text ZH | Handwriting EN | Handwriting ZH |
|---------|-------------|-------------|---------------|---------------|----------------|----------------|
| Baseline | 90.89 | 92.63 | 53.26 | 67.55 | 81.72 | 0.22 |
| FitPrune | 80.53 | 80.59 | 45.51 | 54.40 | 35.25 | 19.84 |
| CDPruner | 62.49 | 63.63 | 55.44 | 35.81 | 64.16 | 66.24 |
| NUWA | 2.12 | 1.34 | 1.67 | 1.32 | 19.91 | 33.24 |
| **RTPrune** | **82.29** | **84.46** | **56.11** | **66.07** | **83.59** | **74.39** |

**动态剪枝率 (Dynamic r, r_dyn)**:

| Methods | Document EN | Document ZH | Scene Text EN | Scene Text ZH | Handwriting EN | Handwriting ZH | AVG r_dyn |
|---------|-------------|-------------|---------------|---------------|----------------|----------------|-----------|
| Baseline | 90.89 | 92.63 | 53.26 | 67.55 | 81.72 | 0.22 | - |
| FitPrune | 63.32 | 90.80 | 46.86 | 33.51 | 71.51 | 62.06 | 0.18 |
| CDPruner | 62.49 | 63.63 | 55.44 | 64.16 | 83.43 | 55.06 | 0.18 |
| NUWA | 2.12 | 1.34 | 1.67 | 1.32 | 34.33 | 1.67 | 0.17 |
| **RTPrune** | **82.29** | **84.46** | **56.11** | **66.07** | **83.59** | **79.62** | **0.30** |

**表格说明**: 动态剪枝策略准确捕捉任务信息密度差异——手写识别信息量较少，可更高比率剪枝（r_avg=0.30），文档提取/场景文本保留更多 token（r_avg=0.17~0.18）。同平均剪枝率下，动态策略提升最高 13.5% 精度。

### 实现细节

- **Backbone**: DeepSeek-OCR（DeepEncoder + DeepSeek3B-MoE-A570M）
- **DeepEncoder 参数**: ~380M (SAM-base 80M + Conv Compressor + CLIP-large 300M)
- **LLM Decoder**: 12 层 (1 standard + 11 MoE)，hidden dim 1280，MoE top-k=6，激活参数 570M
- **优化器**: N/A（训练无关方法）
- **Sinkhorn 迭代数**: T=100
- **Dustbin 参数**: z=0.2
- **合并强度**: α=0.1
- **Sobel 阈值**: τ=0.2
- **硬件**: 未详述

---

## 批判性思考

### 优点

1. **观察驱动的设计**: 不是凭空设计方法，而是从 LLM 解码过程的实证分析出发——Figure 4 的 TIR 分析漂亮地揭示了两阶段阅读行为，方法设计有扎实的 empirical grounding
2. **训练无关 + 即插即用**: 无需任何 retraining 或 fine-tuning，直接在推理时工作，部署成本极低，这在实际工业场景（JD.com）中非常重要
3. **最优传输合并而非简单丢弃**: Stage 2 的设计相比简单丢弃 token 更优雅，通过 Sinkhorn 算法实现全局最优匹配，保留了被剪 token 的互补信息
4. **跨模型泛化强**: Table 5 显示方法不仅适用于 DeepSeek-OCR，在 DeepSeek-OCR2、LightOnOCR、GLM-OCR 上也能保持 92%+ 性能，说明"高 norm token = 文本信息"的假设具有普适性
5. **全面的实验设计**: 覆盖 3 个 benchmark、4 个模型尺寸、12 个对比方法，消融实验覆盖选择度量、合并方式、dustbin 参数、合并强度、Sobel 阈值、剪枝率

### 局限性

1. **加速比有限**: 1.23x prefill 加速远低于 VQA 任务中 token 剪枝的加速（通常 2-3x），这是 OCR 高信息密度固有的限制，但对于实际部署仍有一定价值
2. **Code 尚未开源**: 论文声明"Code is released"但没有找到公开仓库，影响可复现性和社区采纳
3. **仅适用于 OCR 模型**: 方法设计紧密依赖于 DeepSeek-OCR 的联合优化架构，通用 VLM 的 visual encoder 通常是 frozen 的，ℓ2-norm 与文本信息的相关性可能不成立
4. **GLM-OCR 上表格指标偏低**: Table 5 中 GLM-OCR 的 TEDS/TEDS-S 分数很低（~15%），作者解释为 GLM-OCR 输出 plain text table 而非 HTML/LaTeX，但这说明方法在特定输出格式下可能存在评估适配问题
5. **缺乏极端剪枝率实验**: 动态剪枝率的平均 r 约为 0.17-0.30，缺乏更高剪枝率下方法的退化行为分析

### 潜在改进方向

1. **扩展到通用 VLM**: 探索在 frozen visual encoder 的通用 VLM 上是否也存在类似的 norm-text 相关性，或者需要设计替代的重要性度量
2. **学习式动态剪枝**: 当前动态剪枝率的两个因子（φ 和 ρ）是启发式组合，可以探索学习一个轻量预测模块来直接输出最优剪枝率
3. **与量化/蒸馏结合**: RTPrune 是训练无关的 token 级压缩，可以与权重量化、KV-cache 压缩等方法正交叠加，实现更大的推理加速
4. **端到端训练式 token 压缩**: 将 RTPrune 的思想融入到模型训练中，让模型原生学会 token 选择与合并

### 可复现性评估

- [ ] 代码开源（声明 release 但未找到）
- [ ] 预训练模型（使用公开的 DeepSeek-OCR 权重）
- [x] 训练细节完整（方法本身训练无关，超参数在附录完整给出）
- [x] 数据集可获取（OmniDocBench、olmOCR-Bench、Ocean-OCR 均有公开链接）

---

## 关联笔记

### 基于
- [[DeepSeek-OCR]]: RTPrune 的 target 模型，基于其视觉-文本压缩框架
- [[SuperGlue]]: Stage 2 的 dustbin 增强最优传输机制借鉴自 SuperGlue 的特征匹配
- [[Sinkhorn Algorithm]]: 最优传输问题的数值求解方法

### 对比
- [[FastV]]: 基于 LLM 注意力的 token 剪枝 — 在 OCR 上失效（层间注意力分化）
- [[DivPrune]]: 基于 token 多样性的剪枝 — 不适合 OCR 的密集文本需求
- [[CDPruner]]: 基于文本相关性的剪枝 — OCR 固定 prompt 无区分力
- [[SparseVLM]]: 多阶段剪枝 — 浅层剪枝导致深层关键信息丢失
- [[VisionZip]]: 视觉编码器端剪枝 — 无法准确捕捉完整文本信息

### 方法相关
- [[Token Pruning]]: 核心方法类别
- [[Optimal Transport]]: Stage 2 的数学基础
- [[ℓ2-Norm Feature Selection]]: Stage 1 的选择度量
- [[Dynamic Pruning Ratio]]: 自适应剪枝策略

### 硬件/数据相关
- [[OmniDocBench]]: 主要评测 benchmark
- [[olmOCR-Bench]]: PDF 内容提取评测
- [[Ocean-OCR]]: 通用 OCR 能力评测

---

## 速查卡片

> [!summary] RTPrune: Reading-Twice Inspired Token Pruning
> - **核心**: 利用 DeepSeek-OCR 的"读两遍"注意力模式，用 ℓ2-norm 选主导 token + 最优传输合并余量信息
> - **方法**: Stage 1 (ℓ2-norm selection) → Stage 2 (Sinkhorn optimal transport merging) + 动态剪枝率
> - **结果**: DeepSeek-OCR-Large 上 99.47% 精度 / 1.23x prefill 加速 / 84.25% token 保留
> - **代码**: 待发布

---

*笔记创建时间: 2026-05-21*
