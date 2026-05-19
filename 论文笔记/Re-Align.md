---
title: "RE-ALIGN: Aligning Vision Language Models via Retrieval-Augmented Direct Preference Optimization"
method_name: "Re-Align"
authors: [Shuo Xing, Peiran Li, Yuping Wang, Ruizheng Bai, Yueqi Wang, Chan-Wei Hu, Chengxuan Qian, Huaxiu Yao, Zhengzhong Tu]
year: 2025
venue: arXiv
tags: [hallucination, vision-language-model, direct-preference-optimization, image-retrieval, cross-modal-alignment, preference-optimization]
zotero_collection: null
image_source: local
arxiv_html: https://arxiv.org/html/2502.13146
created: 2025-05-19
---

# 论文笔记：RE-ALIGN

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Texas A&M University, University of Michigan, UIUC, UNC Chapel Hill |
| 日期 | Sep 2025 (v3) |
| 项目主页 | - |
| 对比基线 | [[LLaVA]], LLaVA-RLHF, POVID, CSR, SIMA, STIC, mDPO |
| 链接 | [arXiv](https://arxiv.org/abs/2502.13146) |

---

## 一句话总结

> 利用图像检索构建双偏好数据集（文本+视觉），提出 rDPO 扩展目标函数同时优化两种偏好信号，有效缓解 VLM 幻觉并提升通用 VQA 性能。

---

## 核心贡献

1. **检索增强偏好生成**: 通过策略性遮蔽 + 检索相似图片 + 让 VLM 补全被遮蔽部分，有控制地诱导出自然且合理的幻觉作为 rejected 响应
2. **双偏好数据集**: 同时纳入文本偏好信号（chosen vs rejected 响应）和视觉偏好信号（原始图片 vs 检索图片），构建更丰富的偏好数据
3. **rDPO 目标函数**: 在标准 [[Direct Preference Optimization|DPO]] 基础上增加视觉偏好优化项 L_vDPO，使模型在优化过程中同时利用文本和视觉的偏好信号
4. **广泛验证**: 在多个规模（1B-13B）和架构（image-to-text 和 unified model）的 VLM 上均有效，幻觉缓解与通用性能同步提升

---

## 问题背景

### 要解决的问题
[[Vision Language Model|VLM]] 在生成回答时容易出现 [[Hallucination|幻觉]]——输出包含与输入图像不符的错误或捏造细节（物体、属性、逻辑关系）。

### 现有方法的局限

现有基于 [[Direct Preference Optimization|DPO]] 对齐的方法构造偏好数据的方式存在缺陷：

1. **扰动真实响应**（如 POVID）：生成幻觉方式过于简单粗暴（加高斯噪声），无法有控制地产生自然、合理的幻觉
2. **人工修正响应**（如 LLaVA-RLHF）：质量高但成本巨大，难以规模化
3. **仅依赖文本偏好**: 标准 DPO 微调时可能过度偏向语言偏好，忽视视觉信息的重要性，导致次优性能甚至更多幻觉（mDPO 尝试加入视觉偏好但仍依赖随机裁剪）
4. **有限的可扩展性**: 多数基线方法在扩展到更大模型时性能不一致，有的甚至不如 vanilla 模型

### 本文的动机

- 利用 [[Image Retrieval|图像检索]] 从训练集中找到与原图语义相似的图片，有控制地诱导 VLM 产生自然、合理的幻觉作为 rejected 响应
- 将检索到的图片作为"视觉偏好信号"，与文本偏好信号联合优化
- 通过 rDPO 扩展目标函数，让模型学会区分视觉前后一致的响应和不一致的响应

---

## 方法详解

### 模型架构

Re-Align 采用 **检索增强的偏好优化流水线** 架构：

- **输入**: 图像 $v$ + 文本指令 $x$
- **Backbone**: 使用预训练 [[Vision Language Model|VLM]]（如 [[LLaVA]]-v1.5-7B、LLaVA-v1.6-Mistral-7B 等）
- **核心模块**: [[Image Retrieval|图像检索]] 用于偏好数据生成，[[Direct Preference Optimization|rDPO]] 用于联合文本-视觉偏好优化
- **输出**: 经过对齐微调的 VLM

### 核心模块

#### 模块1: 检索增强偏好数据生成

**设计动机**: 利用 [[Image Retrieval|图像检索]] 找到与输入图像语义相似但不同的图片作为"视觉干扰"，诱导 VLM 在被遮蔽的文本上产生自然幻觉

**具体实现**（三阶段流水线）:

1. **策略性遮蔽 (Strategic Masking)**:
   - 用高级 VLM（GPT-4o mini）生成 chosen 响应 $y_w$
   - 将 $y_w$ 中与图像中物体、属性、逻辑关系相关的词段替换为 `[MASK]`，得到 $y_m$

2. **图像检索 (Image Retrieval)**:
   - 用原 VLM 的 vision encoder（[[CLIP]]-vit-large-patch14）将所有训练图片编码为向量
   - 用 [[FAISS]] 库对输入图片 $v_i$ 做 [[Cosine Similarity|余弦相似度]] 搜索，获取得分最高的 top-k 张相似图片

3. **诱导幻觉 (Inducing Hallucinations)**:
   - 依次用检索到的图片 $v_{j_t}$ 作为视觉输入，让 VLM 补全 $y_m$，得到候选补全 $y_c$
   - 用 [[SentenceTransformer]]（all-mpnet-base-v2）计算 $y_w$ 与 $y_c$ 的余弦相似度
   - 若相似度 < 阈值 $\tau=0.95$，则将 $y_c$ 作为 rejected 响应 $y_l$；否则尝试下一张检索图片
   - 若所有 top-k 张图片都无法满足条件，则跳过该样本

#### 模块2: rDPO 目标函数

**设计动机**: 标准 [[Direct Preference Optimization|DPO]] 仅利用文本偏好信号，忽视视觉信息。Re-Align 引入 rDPO，同时优化文本偏好和视觉偏好。

**具体实现**:

- **标准 DPO 损失**:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,v,y_w,y_l)\sim\mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x,v)}{\pi_0(y_w|x,v)} - \beta \log \frac{\pi_\theta(y_l|x,v)}{\pi_0(y_l|x,v)} \right) \right]
$$

- **视觉偏好优化 (vDPO) 损失**: 鼓励模型偏好原始图片 $v$ 而非检索图片 $v_l$ 作为视觉条件

$$
\mathcal{L}_{\text{vDPO}} = -\mathbb{E}_{(x,v,v_l,y_w,y_l)\sim\mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x,v)}{\pi_0(y_w|x,v)} - \beta \log \frac{\pi_\theta(y_w|x,v_l)}{\pi_0(y_w|x,v_l)} \right) \right]
$$

- **rDPO 总损失**:

$$
\mathcal{L}_{\text{rDPO}} = \mathcal{L}_{\text{DPO}} + \mathcal{L}_{\text{vDPO}}
$$

**与 mDPO 的区别**: mDPO 通过随机裁剪原始图片（mask 20% 视觉 token）来产生视觉偏好信号；Re-Align 通过图像检索找到语义上相关但不同的自然图片，提供更强且有意义的对比信号。

---

## 关键公式

### 公式1: [[Direct Preference Optimization|标准 DPO 损失]]

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,v,y_w,y_l)\sim\mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x,v)}{\pi_0(y_w|x,v)} - \beta \log \frac{\pi_\theta(y_l|x,v)}{\pi_0(y_l|x,v)} \right) \right]
$$

**含义**: 最大化 chosen 响应 $y_w$ 相对于 rejected 响应 $y_l$ 的对数概率比，隐式学习奖励函数

**符号说明**:
- $\pi_\theta$: 当前策略（待优化的 VLM）
- $\pi_0$: 参考策略（SFT 后的初始 VLM）
- $(x, v)$: 输入（文本指令 + 图像）
- $(y_w, y_l)$: 偏好响应对（chosen + rejected）
- $\beta$: 控制偏离参考模型程度的超参数
- $\sigma(\cdot)$: sigmoid 函数

### 公式2: [[Cosine Similarity|图像检索相似度]]

$$
s = \left\langle \frac{f_p(v_1)}{\|f_p(v_1)\|}, \frac{f_p(v_2)}{\|f_p(v_2)\|} \right\rangle
$$

**含义**: 计算两幅图像在 VLM vision encoder 嵌入空间中的余弦相似度，用于检索 top-k 最相似图片

**符号说明**:
- $f_p(\cdot)$: VLM 的 vision encoder（如 [[CLIP]]），输出图像嵌入向量
- $\langle \cdot, \cdot \rangle$: $l_2$ 空间中的内积
- $s$: 相似度得分，取值 $[-1, 1]$

### 公式3: [[Direct Preference Optimization|rDPO]] 视觉偏好优化

$$
\mathcal{L}_{\text{vDPO}} = -\mathbb{E}_{(x,v,v_l,y_w,y_l)\sim\mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x,v)}{\pi_0(y_w|x,v)} - \beta \log \frac{\pi_\theta(y_w|x,v_l)}{\pi_0(y_w|x,v_l)} \right) \right]
$$

**含义**: 鼓励模型在原始图片 $v$ 条件下比在检索图片 $v_l$ 条件下更偏好 chosen 响应 $y_w$

**符号说明**:
- $v_l$: 检索到的与 $v$ 语义相似的图片（来自训练集 top-k 检索结果）
- 其余符号同 DPO 损失

### 公式4: rDPO 总损失

$$
\mathcal{L}_{\text{rDPO}} = \mathcal{L}_{\text{DPO}} + \mathcal{L}_{\text{vDPO}}
$$

**含义**: 等权组合文本偏好和视觉偏好两种优化目标，同时增强图文一致性和回答质量

### 公式5: mDPO 的条件偏好优化损失（对比参考）

$$
\mathcal{L}_{\text{CoDPO}} = -\mathbb{E}_{(x,v,y_w,y_l)\sim\mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x,v_c)}{\pi_0(y_w|x,v)} - \beta \log \frac{\pi_\theta(y_w|x,v)}{\pi_0(y_w|x,v_c)} \right) \right]
$$

**含义**: mDPO 的视觉偏好通过随机裁剪 $v_c$（mask 20% 视觉 tokens）引入，与 Re-Align 的检索增强方式形成对比

**符号说明**:
- $v_c$: 随机裁剪后的图像（mask 20% 视觉 token）

---

## 关键图表

### Figure 1: Benchmark Performance Comparison

![[Re-Align_fig1_benchmark.png]]

**说明**: Re-Align 与其他基线方法在幻觉检测和通用 VQA benchmark 上的性能对比（min-max 归一化后的雷达图），展示 Re-Align 在各项指标上的综合优势。

### Figure 2: RE-ALIGN Framework Overview

![[Re-Align_fig2_framework.png]]

**说明**: Re-Align 的整体框架。从训练数据出发，利用高级 VLM 生成 chosen 响应，经策略性遮蔽后，结合检索图片诱导幻觉得到 rejected 响应，构建偏好数据集，最后通过 $\mathcal{L}_{\text{rDPO}}$ 微调 VLM。

### Figure 3: Preference Generation Process

![[Re-Align_fig3_preference_gen.png]]

**说明**: 偏好数据生成的详细流程。训练数据图片经 vision encoder 编码后通过 [[FAISS]] 做相似度搜索得到 top-k 检索图片；文本经策略性遮蔽后，用检索图片和预训练 VLM 补全遮蔽部分；若补全结果与 chosen 响应的 [[Cosine Similarity|相似度]] < 0.95 则作为 rejected 响应。

### Figure 4: Scaling with Preference Data Size

![[Re-Align_fig4_data_scale.png]]

**说明**: 以 LLaVA-v1.6-Mistral-7B 为 backbone 在 ScienceQA 上的性能随偏好数据量增加的变化。数据量从 11k 扩至 16k 时，性能提升从 0.45 增至 1.34，证明方法可随数据规模扩展。

### Figure 5: Example Preference Pair for VQA

![[Re-Align_fig5_vqa_example.png]]

**说明**: VQA 任务的偏好数据对示例。展示原始图片、检索图片、chosen 响应和 rejected 响应。rejected 响应因检索图片的误导导致内容与原始图片不一致。

### Figure 6: Example Preference Pair for Image Captioning

![[Re-Align_fig6_caption_example.png]]

**说明**: 图像描述任务的偏好数据对示例。原始图片与检索图片语义相似但细节不同，rejected 响应因视觉条件不同而包含与原始图片不符的描述。

### Figure 7: Response Examples (LLaVABench)

![[Re-Align_fig7_responses.png]]

**说明**: LLaVA-v1.5-7B 原始模型与 Re-Align 微调后模型在 LLaVABench 上的回答对比。原始模型存在严重的物体幻觉，Re-Align 给出更清晰、更准确的图像描述。

### Table 1: Hallucination Benchmarks (LLaVA-v1.5-7B & LLaVA-v1.6-Mistral-7B)

| Methods | POPEr | POPEp | POPEa | Hallusionq | Hallusionf | HallusionEasy | HallusionHard | Hallusiona |
|---------|-------|-------|-------|------------|------------|---------------|---------------|------------|
| LLaVA-v1.5-7B | 88.14 | 87.23 | 85.10 | 10.33 | 18.21 | 41.76 | 40.23 | 46.32 |
| w. LLaVA-RLHF | 84.77 | 84.60 | 83.40 | 10.29 | 18.79 | 38.24 | 40.67 | 44.65 |
| w. POVID | 88.21 | 87.16 | 85.06 | 10.55 | 18.21 | 41.54 | 40.93 | 46.68 |
| w. CSR (3Iter) | 87.83 | 87.00 | 85.00 | 10.11 | 18.21 | 41.76 | 40.70 | 46.94 |
| w. SIMA | 88.10 | 87.10 | 85.03 | 10.99 | 17.63 | 43.05 | 40.23 | 45.27 |
| w. mDPO | 88.17 | 87.13 | 85.03 | 9.89 | 18.50 | 41.98 | 40.00 | 46.15 |
| **w. Re-Align** | **88.65** | **87.43** | **85.16** | **11.21** | **18.79** | **45.52** | **41.63** | **47.62** |
| LLaVA-v1.6-Mistral-7B | 88.83 | 87.93 | 86.43 | 13.63 | 19.08 | 47.47 | 33.49 | 46.06 |
| w. STIC | 89.03 | 88.20 | 86.56 | 12.97 | 17.34 | 47.25 | 34.19 | 46.32 |
| **w. Re-Align** | **90.55** | **89.20** | **87.03** | **13.85** | **19.08** | **48.35** | **34.88** | **46.59** |

**说明**: Re-Align 在两个 backbone 模型的所有幻觉 benchmark 上均取得最佳或接近最佳结果。在 LLaVA-v1.5-7B 上，HallusionEasy 提升最大（41.76 -> 45.52），Hallusiona 综合得分最优。

### Table 2: General VQA Benchmarks

| Methods | SQA | TextVQA | MM-Vet | VisWiz | LLaVABench | MME_P | MME_C | MMBench | Avg. Rank |
|---------|-----|---------|--------|--------|------------|-------|-------|---------|-----------|
| LLaVA-v1.5-7B | 66.02 | 58.18 | 31.6 | 50.03 | 64.1 | 1510.28 | 357.85 | 64.60 | 3.875 |
| w. LLaVA-RLHF | 63.11 | 56.89 | 31.8 | 49.57 | 60.2 | 1378.90 | 282.85 | 64.39 | 6.000 |
| w. POVID | 65.98 | 58.18 | 31.8 | 49.80 | 67.3 | 1495.91 | 356.07 | 64.34 | 4.375 |
| w. CSR (3Iter) | 65.46 | 57.86 | 31.6 | 47.02 | 68.3 | 1525.44 | 365.35 | 64.08 | 4.500 |
| w. SIMA | 65.83 | 58.48 | 32.0 | 50.04 | 66.9 | 1510.33 | 371.78 | 64.60 | 2.750 |
| w. mDPO | 67.53 | 57.90 | 31.3 | 50.04 | 59.0 | 1510.74 | 335.71 | 64.60 | 4.250 |
| **w. Re-Align** | **68.10** | **58.55** | **32.1** | **50.06** | **67.7** | **1511.79** | **367.50** | **64.69** | **1.375** |
| LLaVA-v1.6-Mistral-7B | 76.02 | 63.80 | 47.6 | 59.85 | 80.2 | 1494.22 | 323.92 | 69.33 | 2.125 |
| w. STIC | 76.42 | 63.50 | 47.3 | 54.21 | 81.0 | 1504.91 | 308.21 | 69.16 | 2.625 |
| **w. Re-Align** | **76.47** | **64.08** | **48.3** | **57.27** | **81.8** | **1512.09** | **318.93** | **69.42** | **1.250** |

**说明**: Re-Align 在两个 backbone 上均取得最优平均排名（1.375 和 1.250），在大多数子任务上超越或持平 vanilla 模型和所有基线。证明方法在缓解幻觉的同时不会牺牲通用性能。

### Table 3: Scalability Across Model Sizes and Architectures (POPE)

| Methods               | POPEr             | POPEp             | POPEa             |
| --------------------- | ----------------- | ----------------- | ----------------- |
| Janus-Pro-1B          | 85.46             | 85.03             | 84.13             |
| **w. Re-Align**       | **87.53 (+2.07)** | **87.33 (+2.30)** | **85.86 (+1.73)** |
| Janus-Pro-7B          | 88.41             | 87.30             | 85.70             |
| **w. Re-Align**       | **89.73 (+1.32)** | **88.37 (+1.07)** | **86.27 (+0.57)** |
| Qwen2.5-VL-3B-Inst.   | 88.32             | 87.60             | 86.63             |
| **w. Re-Align**       | **89.69 (+1.37)** | **88.33 (+0.73)** | **87.16 (+0.53)** |
| Qwen2.5-VL-7B-Inst.   | 88.73             | 87.90             | 86.87             |
| **w. Re-Align**       | **89.27 (+0.54)** | **88.10 (+0.20)** | **87.10 (+0.23)** |
| LLaVA-v1.5-7B         | 88.14             | 87.23             | 85.10             |
| w. LLaVA-RLHF         | 84.77 (-3.37)     | 84.60 (-2.63)     | 83.40 (-0.50)     |
| w. POVID              | 88.21 (+0.07)     | 87.16 (-0.07)     | 85.06 (-0.04)     |
| w. CSR (3Iter)        | 87.83 (-0.31)     | 87.00 (-0.23)     | 85.00 (-0.10)     |
| w. SIMA               | 88.10 (-0.04)     | 87.10 (-0.13)     | 85.03 (-0.07)     |
| w. mDPO               | 88.17 (+0.03)     | 87.13 (-0.10)     | 85.03 (-0.07)     |
| **w. Re-Align**       | **88.65 (+0.51)** | **87.43 (+0.20)** | **85.16 (+0.06)** |
| LLaVA-v1.5-13B        | 88.07             | 87.53             | 85.60             |
| w. CSR (3Iter)        | 88.38 (+0.31)     | 87.90 (+0.37)     | 85.46 (-0.14)     |
| w. SIMA               | 88.04 (-0.03)     | 87.40 (-0.13)     | 85.40 (-0.20)     |
| w. HSA-DPO            | 85.01 (-3.06)     | 85.00 (-2.53)     | 83.86 (-1.74)     |
| **w. Re-Align**       | **90.03 (+1.96)** | **89.20 (+1.30)** | **86.20 (+0.74)** |
| LLaVA-v1.6-Vicuna-7B  | 88.52             | 87.63             | 86.36             |
| **w. Re-Align**       | **88.94 (+0.42)** | **88.03 (+0.40)** | **86.63 (+0.27)** |
| LLaVA-v1.6-Vicuna-13B | 88.24             | 87.70             | 86.43             |
| **w. Re-Align**       | **88.79 (+0.55)** | **88.10 (+0.40)** | **86.60 (+0.17)** |

**说明**: Re-Align 在所有模型规模和架构（image-to-text 和 unified model）上均取得一致的性能提升。LLaVA-v1.5-13B 提升最大（POPEr +1.96），Janus-Pro-1B 提升也显著（+2.30 POPEp）。多数基线方法表现出不稳定甚至负提升，凸显 Re-Align 的优越可扩展性。

### Table 4: Similarity Threshold $\tau$ Ablation

| $\tau$ | SQA | TextVQA | POPEr | POPEp | POPEa |
|--------|-----|---------|-------|-------|-------|
| 0.85 | 67.04 | 57.31 | **88.96** | **87.83** | 85.06 |
| 0.90 | 67.75 | 57.68 | 88.83 | 87.66 | 84.93 |
| 0.95 | **68.10** | **58.55** | 88.65 | 87.43 | **85.16** |

**说明**: $\tau=0.95$ 提供幻觉缓解与通用 VQA 性能的最佳平衡。降低 $\tau$（更强的偏好信号）提升幻觉缓解但可能损害通用性能。

### Table 5: Masking Strategy Ablation

| Masking Strategy | SQA | TextVQA | POPEr | POPEp | POPEa |
|------------------|-----|---------|-------|-------|-------|
| sentence-level | 67.58 | 57.47 | 88.91 | 87.52 | 85.16 |
| **segment-level** | **68.10** | **58.55** | 88.65 | 87.43 | 85.16 |

**说明**: Segment-level 掩码在通用 VQA 任务上略优于 sentence-level，幻觉缓解方面两种策略相当。

### Table 6: rDPO Weight $w_v$ Ablation

| $w_v$ | SQA | TextVQA | POPEr | POPEp | POPEa |
|-------|-----|---------|-------|-------|-------|
| 0.0 (DPO) | 66.26 | 57.77 | 88.56 | 87.60 | 88.18 |
| 0.25 | 67.15 | 58.55 | 88.65 | 87.43 | 88.72 |
| 0.50 | 67.01 | 57.69 | 88.76 | 87.53 | 88.65 |
| 0.75 | 67.53 | 57.41 | 88.90 | 87.70 | 84.83 |
| **1.0 (rDPO)** | **68.10** | **58.55** | 88.65 | 87.43 | **85.16** |

**说明**: 加入 $w_v > 0$（视觉偏好优化）均优于纯 DPO（$w_v=0$）。$w_v=1.0$（等权）在幻觉缓解和通用性能之间取得最佳平衡。

### Table 7: Training Epochs

| Num Epoch | SQA | TextVQA | POPEr | POPEp | POPEa |
|-----------|-----|---------|-------|-------|-------|
| 1 | 68.10 | 58.55 | 88.65 | 87.43 | 85.16 |
| 2 | 68.27 | 58.47 | 88.91 | 87.52 | 85.16 |
| 3 | **68.17** | **58.60** | 88.57 | **87.60** | **85.43** |

**说明**: 延长训练（2-3 epochs）不会导致过拟合，性能保持稳定甚至略有提升，证明方法对训练时长的鲁棒性。

### Table 8: Preference Dataset Summary

| Methods | Source | Size | Preference Signal | Curation Strategy | Visual Modification |
|---------|--------|------|-------------------|-------------------|---------------------|
| LLaVA-RLHF | LLaVA-Instruct | 10k | Textual only | Human annotation | None |
| POVID | LLaVA-Instruct | 17k | Textual only | Image noising + prompting | Gaussian noise |
| CSR | LLaVA-Instruct | 13k | Textual only | Self-rewarding | None |
| SIMA | COCO | 5k | Textual only | Self-rewarding | None |
| STIC | COCO | 6k | Textual only | Corruption + prompting | Color jitter + lower resolution |
| **Re-Align** | **LLaVA-Instruct** | **11k** | **Textual & Visual** | **Image retrieval + strategic masking** | **Semantically-guided natural images** |

**说明**: Re-Align 是唯一同时利用文本和视觉偏好信号、不依赖人工标注或人工扰动的方法，使用语义引导的自然图片作为视觉偏好信号。

### Table 9: Fine-tuning Time

| Models | Required Time |
|--------|--------------|
| Janus-Pro-1B | 50 min |
| Janus-Pro-7B | 93 min |
| LLaVA-v1.5-7B | 35 min |
| LLaVA-v1.5-13B | 45 min |
| LLaVA-v1.6-Mistral-7B | 30 min |
| LLaVA-v1.6-Vicuna-7B | 46 min |
| LLaVA-v1.6-Vicuna-13B | 72 min |

**说明**: 使用 4 张 NVIDIA A6000ada GPU，训练时间在 30-93 分钟之间。

### Table 10: Hyperparameter Settings

| Hyperparameter | Setting |
|---------------|---------|
| $\beta$ | 0.1 |
| Learning rate | 1e-5 |
| weight_decay | 0.0 |
| warmup_ratio | 0.03 |
| lr_scheduler_type | cosine |
| mm_projector_lr | 2e-5 |
| mm_projector_type | mlp2x_gelu |
| gradient_accumulation_steps | 8 |
| per_device_train_batch_size | 1 |
| bf16 | True |
| Optimizer | AdamW |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LLaVA-Instruct-150K | 150k（采样 11k） | 多模态指令跟随数据 | 偏好数据构建 |
| [[POPE]] | - | 物体幻觉检测（random/popular/adversarial） | 幻觉评测 |
| [[HallusionBench]] | - | 视觉错觉+语言纠缠诊断 | 幻觉评测 |
| [[ScienceQA]] | - | 科学问答 | 通用 VQA |
| [[TextVQA]] | - | 图中文字理解 | 通用 VQA |
| MM-Vet | - | 多模态综合能力 | 通用 VQA |
| VisWiz | - | 盲人视觉问答 | 通用 VQA |
| LLaVABench | - | 对话式多模态评测 | 通用 VQA |
| [[MME]] | - | 感知+认知综合评测 | 通用 VQA |
| [[MMBench]] | - | 全方位多模态能力 | 通用 VQA |

### 实现细节

- **Backbone**: LLaVA-v1.5-7B/13B, LLaVA-v1.6-Mistral-7B, LLaVA-v1.6-Vicuna-7B/13B, Qwen2.5-VL-3B/7B, Janus-Pro-1B/7B
- **优化器**: AdamW
- **学习率**: 1e-5 (LLM), 2e-5 (mm_projector)
- **Batch Size**: 1 per device, gradient accumulation 8
- **训练轮数**: 1 epoch（主实验）/ 最多 3 epochs（消融）
- **硬件**: 4x NVIDIA A6000ada GPU
- **微调方法**: [[LoRA]] (r=128, alpha=256, target all modules)
- **偏好数据构建**: GPT-4o mini 生成 chosen + 遮蔽, all-mpnet-base-v2 计算文本相似度
- **图像检索**: [[CLIP]]-vit-large-patch14 + [[FAISS]] (k=10)

### 可视化结果

从 Figure 7 的对比回答可以看出，原始 LLaVA-v1.5-7B 模型在 LLaVABench 上存在严重的物体幻觉（如错误描述图像中的物体），而 Re-Align 微调后的模型给出更清晰、更准确的描述，直观展示了方法的幻觉缓解效果。

---

## 批判性思考

### 优点
1. **视觉偏好信号的新颖引入方式**: 通过图像检索而非随机扰动（裁剪、加噪声）引入视觉偏好，信号更有语义意义和对比性
2. **控制幻觉的巧妙设计**: 策略性遮蔽 + 语义相似图片诱导的方式，比直接扰动响应或添加噪声更能产生自然、合理的幻觉，使偏好信号质量更高
3. **广泛的可扩展性验证**: 在 1B-13B 参数范围、image-to-text 和 unified model 两种架构上都有效，证明了方法的通用性
4. **计算高效**: 偏好数据构建可完全离线完成，训练仅增加 5-10% 时间开销，无需额外的 reward model
5. **训练鲁棒性**: 对训练轮数和数据量不敏感，不易过拟合

### 局限性
1. **Alignment Tax**: 在某些通用 VQA 子任务上不如 vanilla 模型或某些基线（如 VisWiz on LLaVA-v1.6），存在 RLHF 中普遍的对齐税问题
2. **依赖外部模型**: 偏好数据的构建依赖 GPT-4o mini（约 $90 成本）和 SentenceTransformer，影响了方法的完全自包含性
3. **数据构建有跳过率**: 当所有 top-k 检索图片都无法产生满足阈值条件的 rejected 响应时，该样本被丢弃，可能漏掉某些难例
4. **仅评估英文 benchmark**: 缺乏多语言场景下的验证

### 潜在改进方向
1. **多轮迭代优化**: 像 CSR 一样采用多轮 self-rewarding 迭代，可能进一步提升已对齐 VLM 的性能
2. **在线偏好优化**: 将离线构建的偏好数据与在线 RLHF 结合（如 PPO），可能突破离线数据的上限
3. **动态阈值**: 替代固定的 $\tau=0.95$，使用与样本难度自适应的相似度阈值
4. **多模态检索增强**: 同时使用文本+图像的多模态检索（如 [[CLIP]] 跨模态检索），而非仅用图像检索

### 可复现性评估
- [x] 代码开源（算法伪代码完整，超参数全公开）
- [ ] 预训练模型（使用开源 backbone 但未提供对齐后权重）
- [x] 训练细节完整（Table 10 列出所有超参数）
- [x] 数据集可获取（LLaVA-Instruct-150K 为 CC BY 4.0，各 benchmark 均有明确许可证）

---

## 关联笔记

### 基于
- [[LLaVA]]: 使用 LLaVA-v1.5 和 v1.6 系列作为 backbone VLM
- [[Direct Preference Optimization]]: rDPO 是 DPO 的扩展
- [[CLIP]]: 用于图像编码和相似度检索
- [[RLHF]]: 对齐范式的理论基础

### 对比
- POVID: 通过高斯噪声扰动图片 + GPT-4V 生成幻觉（Re-Align 用检索替代噪声）
- CSR: 迭代 self-rewarding 构建偏好数据（Re-Align 用检索+MASK 策略）
- SIMA: 自生成+自批评选择偏好对（信号来源不同）
- mDPO: 随机裁剪引入视觉偏好（Re-Align 用检索替代裁剪）
- STIC: 损坏图片+误导提示构建偏好数据（信号质量不如检索）

### 方法相关
- [[Vision Language Model|VLM]]: 核心应用对象
- [[Hallucination]]: 核心解决问题
- [[Image Retrieval|图像检索]]: 核心方法组件
- [[FAISS]]: 向量检索工具
- [[SentenceTransformer]]: 文本相似度计算
- [[LoRA]]: 高效微调方法
- [[Cross-Modal Alignment|跨模态对齐]]: 核心目标
- [[Preference Optimization|偏好优化]]: 训练范式
- [[Preference Dataset|偏好数据集]]: 数据构建

### 硬件/数据相关
- [[POPE]]: 幻觉评测 benchmark
- [[HallusionBench]]: 幻觉评测 benchmark
- [[ScienceQA]]: 通用 VQA benchmark
- [[TextVQA]]: 通用 VQA benchmark
- [[MME]]: 通用 VQA benchmark
- [[MMBench]]: 通用 VQA benchmark

---

## 速查卡片

> [!summary] RE-ALIGN: Aligning VLMs via Retrieval-Augmented DPO
> - **核心**: 用图像检索构建双偏好数据集 + rDPO 联合优化文本和视觉偏好信号
> - **方法**: 策略性遮蔽 -> 检索相似图片 -> 诱导自然幻觉 -> rDPO (L_DPO + L_vDPO) 对齐微调
> - **结果**: 所有 backbone 上幻觉显著缓解，通用 VQA 性能同步提升，平均排名最优
> - **代码**: 算法伪代码完整（Algorithm 1），超参数全公开（Table 10）

---

*笔记创建时间: 2025-05-19*
