---
title: "PostAlign: Multimodal Grounding as a Corrective Lens for MLLMs"
method_name: "MMGrounded-PostAlign"
authors: [Yixuan Wu, Yang Zhang, Jian Wu, Philip Torr, Jindong Gu]
year: 2025
venue: arXiv
tags: [multimodal-hallucination, visual-grounding, multimodal-alignment, mllm, hallucination-mitigation, post-alignment, selective-reasoning, negative-rejection]
zotero_collection: null
image_source: mixed
arxiv_html: https://ar5iv.org/html/2506.17901
created: 2026-05-20
---

# 论文笔记：PostAlign: Multimodal Grounding as a Corrective Lens for MLLMs

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Oxford, Zhejiang University, National University of Singapore |
| 日期 | June 2025 |
| 项目主页 | - |
| 对比基线 | [[LLaVA]], [[POPE]], [[MMBench]], [[HaloQuest]] |
| 链接 | [arXiv](https://arxiv.org/abs/2506.17901) / Code: N/A |

---

## 一句话总结

> 提出后对齐框架 MMGrounded-PostAlign，用多模态定位作为"修正镜头"，通过负样本拒绝和选择性推理抑制 MLLM 幻觉、增强视觉理解。

---

## 核心贡献

1. **后多模态对齐框架 (Post-Alignment)**: 提出 MMGrounded-PostAlign，将视觉定位与文本定位整合为"修正镜头"，在不破坏 MLLM 预训练能力的前提下增强视觉理解、减少幻觉。
2. **负样本拒绝机制 (<REJ> Token)**: 在视觉定位模块中引入 `<REJ>` token，使模型能显式拒绝图像中不存在的物体，有效对抗语言先验诱导的[[多模态幻觉|幻觉]]。
3. **选择性推理机制**: 基于查询复杂度动态分类 `<SIMPLE>` / `<COMPLEX>`，避免简单查询上的过度推理，确保复杂查询有充分推理支持。
4. **幻觉溯源分析**: 实验证明移除图像输入后，幻觉 token 重叠率达 **89.2%**，揭示语言先验是多模态幻觉的主导因素。

---

## 问题背景

### 要解决的问题
[[多模态大模型]] (MLLMs) 在视觉问答、图像描述等任务中表现出色，但存在严重的**过依赖[[虚假相关|虚假相关性]]**问题：模型倾向于基于统计共现（如 "桌子" 常与 "椅子" 共现）生成回答，而非真正分析视觉内容。

### 现有方法的局限
1. **后处理方法** (如 Volcano, Woodpecker): 需要额外推理轮次，推理时间增加，通用性差
2. **训练方法** (如 RL-based): 依赖数据增强或强化学习，未能从根本上解决跨[[多模态对齐|模态对齐]]问题
3. **BTL (Bounding-box as Language Token) 方法**: 将边界框编码为文本 token 训练，会损害 MLLM 的推理能力（"overfitting to visual bbox information at the cost of abstract reasoning"）

### 本文的动机
核心假设：**多模态幻觉主要由[[语言先验]]主导**，而非视觉误判。作者通过实验证明：移除图像后，幻觉 token 重叠率达 89.2%。因此，解决方案应该是"用多模态定位作为修正镜头"——让模型的输出同时锚定在视觉证据和文本推理上。

---

## 方法详解

### 模型架构

MMGrounded-PostAlign 采用 **后多模态对齐** 架构，基于 [[LLaVA|LLaVA-1.5]] (7B/13B)：

- **MLLM Backbone**: [[LLaVA|LLaVA-1.5-7B]] / LLaVA-1.5-13B，用 [[LoRA]] 高效微调
- **视觉定位 Backbone**: [[SAM|ViT-H SAM]]，冻结参数
- **多任务解码器**: 包含掩码解码器 (Mask Decoder) + 边界框解码器 (Bbox Decoder)，全参数微调
- **输入**: 图像 $\mathcal{I}$ + 文本查询 $\mathcal{Q}$
- **输出**: 结构化输出 $\mathcal{A} = \{\mathcal{V}, \mathcal{T}, \mathcal{F}\}$
  - $\mathcal{V}$: 视觉定位 token (`<LOC>`)
  - $\mathcal{T}$: 文本定位 token (推理理由)
  - $\mathcal{F}$: 最终答案 token
- **训练**: [[DeepSpeed]] + AdamW, lr=0.0003, batch_size=2, gradient_accumulation=10, WarmupDecayLR (100 warmup steps)

### 核心模块

#### 模块1: 视觉定位模块 (Visual Grounding Module)

**设计动机**: 利用[[视觉定位]]提供像素级和边界框级的视觉证据，锚定模型输出。

**具体实现**:
- MLLM 生成 `<LOC>` token，其最后一层嵌入通过 MLP 投影层转为 prompt 嵌入
- [[SAM|ViT-H SAM]] 提取图像密集视觉特征
- 多任务解码器同时输出：
  - **分割掩码**: [[SAM]] 的 mask decoder 以 `<LOC>` 嵌入为 prompt
  - **边界框**: 轻量 MLP 回归器从 SAM 特征预测坐标

#### 模块2: 负样本拒绝机制 (Negative Rejection Mechanism)

**设计动机**: MLLMs 常因[[共现偏差]] + [[语言先验]]幻觉出不存在的物体（如看到"椅子"就预测"桌子"）。需要显式惩罚这种错误。

**具体实现**:
- 引入 `<REJ>` token：当图像中不存在被指代物体时，模型预测 `<REJ>` 替代 `<LOC>`
- 多任务解码器对 `<REJ>` token 直接分配空掩码和空边界框
- 训练数据中混入负样本（物体不存在的标注），标注为 `<REJ>`
- 使用[[负样本拒绝|负样本拒绝损失]] $\mathcal{L}_{\text{rej}}$ 监督

#### 模块3: 选择性推理机制 (Selective Reasoning Mechanism)

**设计动机**: 不是所有查询都需要显式推理链。简单查询上的"过度推理"反而降低效率甚至损害准确性。

**具体实现**:
- 训练时将查询分为 `<SIMPLE>` 和 `<COMPLEX>` 两类
- `<SIMPLE>`: 直接输出 `<LOC>` + 答案，跳过推理 (如 "车是什么颜色？")
- `<COMPLEX>`: 输出 `<LOC>` + 推理理由 + 答案 (如 "图中哪种食物蛋白质最多？")
- 推理时通过自反思提示 (self-reflection prompting) 自动判断复杂度
- 使用[[选择性推理|选择性推理损失]] $\mathcal{L}_{\text{reason}}$ 监督

---

## 关键公式

### 公式1: [[多模态大模型|结构化输出定义]]

$$
\mathcal{A} = \text{MLLM}(\mathcal{I}, \mathcal{Q})
$$

$$
\mathcal{A} = \{\mathcal{V}, \mathcal{T}, \mathcal{F}\}
$$

**含义**: MLLM 接收图像和查询，输出包含视觉定位 token、文本定位 token 和最终答案的结构化响应。

**符号说明**:
- $\mathcal{I}$: 输入图像
- $\mathcal{Q}$: 文本查询
- $\mathcal{A}$: 结构化输出
- $\mathcal{V}$: 视觉定位 token (`<LOC>`)
- $\mathcal{T}$: 文本定位 token (推理理由)
- $\mathcal{F}$: 最终答案 token

### 公式2: [[负样本拒绝|负样本拒绝损失]]

$$
\mathcal{L}_{\text{rej}} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_{i}^{\text{rej}}\log p_{i}^{\text{rej}} + (1-y_{i}^{\text{rej}})\log(1-p_{i}^{\text{rej}})\right]
$$

**含义**: 二分类交叉熵损失，监督模型判断目标物体是否应被拒绝（不存在于图像中）。

**符号说明**:
- $y_{i}^{\text{rej}} \in \{0, 1\}$: 样本 $i$ 中的指代物是否应被拒绝
- $p_{i}^{\text{rej}}$: 模型预测的拒绝概率
- $N$: 样本数

### 公式3: [[选择性推理|选择性推理损失]]

$$
\mathcal{L}_{\text{reason}} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_{i}^{\text{rea}}\log p_{i}^{\text{rea}} + (1-y_{i}^{\text{rea}})\log(1-p_{i}^{\text{rea}})\right]
$$

**含义**: 监督模型判断查询是否需要复杂推理（`<SIMPLE>` vs `<COMPLEX>`）。

**符号说明**:
- $y_{i}^{\text{rea}} \in \{0, 1\}$: 查询 $i$ 是否需要生成推理理由
- $p_{i}^{\text{rea}}$: 模型预测的推理需求概率

### 公式4: [[损失函数|总训练损失]]

$$
\mathcal{L} = \lambda_{1}\mathcal{L}_{\text{rej}} + \lambda_{2}\mathcal{L}_{\text{reason}} + \mathcal{L}_{\text{ground}} + \mathcal{L}_{\text{text}}
$$

**含义**: 联合优化拒绝损失、推理损失、视觉定位损失和文本生成损失。

**符号说明**:
- $\lambda_{1}, \lambda_{2}$: 平衡系数
- $\mathcal{L}_{\text{ground}}$: 视觉定位损失（det + seg）
- $\mathcal{L}_{\text{text}}$: 交叉熵语言建模损失

### 公式5: [[边界框检测|检测损失]]

$$
\mathcal{L}_{\text{det}} = \mathcal{L}_{\text{smooth-L1}}(\hat{y}_{\text{bbox}}, y_{\text{bbox}}) + \mathcal{L}_{\text{GIoU}}(\hat{y}_{\text{bbox}}, y_{\text{bbox}})
$$

**含义**: 边界框回归损失，结合 Smooth L1 和 GIoU（广义 IoU）。

**符号说明**:
- $\hat{y}_{\text{bbox}}$: 预测边界框
- $y_{\text{bbox}}$: 真实边界框
- $\mathcal{L}_{\text{smooth-L1}}$: Smooth L1 损失
- $\mathcal{L}_{\text{GIoU}}$: 广义 IoU 损失

### 公式6: [[分割掩码|分割损失]]

$$
\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{BCE}}(\hat{y}_{\text{mask}}, y_{\text{mask}}) + \mathcal{L}_{\text{DICE}}(\hat{y}_{\text{mask}}, y_{\text{mask}})
$$

**含义**: 分割掩码损失，结合二值交叉熵和 Dice 损失。

**符号说明**:
- $\hat{y}_{\text{mask}}$: 预测掩码
- $y_{\text{mask}}$: 真实掩码
- $\mathcal{L}_{\text{BCE}}$: 二值交叉熵
- $\mathcal{L}_{\text{DICE}}$: Dice 系数损失

### 公式7: [[语言建模|文本生成损失]]

$$
\mathcal{L}_{\text{text}} = \mathcal{L}_{\text{LM}}(\hat{y}_{\text{txt}}, y_{\text{txt}})
$$

**含义**: 标准交叉熵语言建模损失，监督文本输出。

**符号说明**:
- $\hat{y}_{\text{txt}}$: 预测文本
- $y_{\text{txt}}$: 目标文本
- $\mathcal{L}_{\text{LM}}$: 语言建模交叉熵

---

## 关键图表

### Figure 1: 问题动机与方法概览

![[MMGrounded-PostAlign_fig1.png|600]]

**说明**: (a) 当前 MLLM 受[[虚假相关|虚假相关性]]影响产生[[共现偏差|共现幻觉]]（如在沙发上错误识别出猫）。(b) MMGrounded-PostAlign 通过多模态定位模块将最终输出锚定在真实视觉和文本证据上。

### Figure 2: MMGrounded-PostAlign 整体管线

![[MMGrounded-PostAlign_fig2.png|600]]

**说明**: 给定图像和文本查询，MLLM 生成 `<LOC>` token、文本推理和最终答案。`<LOC>` 的最后一层嵌入送入多任务解码器，生成分割掩码和边界框。当目标物体不存在时，`<LOC>` 被 `<REJ>` 替换，分配空掩码/空边界框。文本定位为复杂查询提供推理理由。

### Figure 3: 幻觉 Token 概率分析

![Figure 3](https://ar5iv.org/html/2506.17901/assets/x3.png)

**说明**: (a)(b) 各 Transformer 层的 token 概率分布——非幻觉 token (绿色) 和幻觉 token (粉色) 呈现不同趋势。非幻觉 token 如 "woman" 在第 20 层即达到高概率，幻觉 token 如 "bottle" 在第 30 层才追上。(c)(d) 移除图像输入后，幻觉 token 仍有 **89.2%** 的重叠率，证实[[语言先验]]主导幻觉生成。

### Figure 4: 注意力图对比

![Figure 4](https://ar5iv.org/html/2506.17901/assets/x4.png)

**说明**: 左图为无定位引导的注意力分布（稀疏、分散），右图为有定位引导的注意力分布（集中、聚焦于相关图像区域）。说明[[视觉定位]]有效引导模型关注相关视觉特征，产生更视觉锚定的回答。

### Table 1: 视觉定位 Token 消融实验 (HaloQuest)

| Method | `<SEG>` | `<DET>` | `<REJ>` | False Premise (Human) | False Premise (Auto) | Visually Challenging (Human) | Visually Challenging (Auto) | Insufficient Context (Human) | Insufficient Context (Auto) |
|--------|---------|---------|---------|----------------------|---------------------|----------------------------|----------------------------|----------------------------|----------------------------|
| Baseline | ✗ | ✗ | ✗ | 2.0 | 2.3 | 23.5 | 23.0 | 2.5 | 1.7 |
| +Mask | ✓ | ✗ | ✗ | 6.5 | 7.2 | 30.1 | 30.6 | 7.4 | 8.2 |
| +Bbox | ✗ | ✓ | ✗ | 8.2 | 8.9 | 31.1 | 31.7 | 6.6 | 7.4 |
| +Mask+Bbox | ✓ | ✓ | ✗ | 9.9 | 10.5 | 33.9 | 35.0 | 9.9 | 11.6 |
| **+Mask+Bbox+REJ** | ✓ | ✓ | ✓ | **33.2** | **33.9** | **38.3** | **37.2** | **31.4** | **32.2** |

**说明**: 视觉定位模块的消融。组合掩码+边界框定位显著提升所有子任务，`<REJ>` token 在 False Premise 和 Insufficient Context 场景（即反概念场景，如 "图中有狗吗？" 答案应为 "没有狗"）中带来最大增益。

### Table 2: 定位策略对比 (POPE / MME / MMBench / VQAv2)

| Method | POPE Rand | POPE Pop | POPE Adv | MME | MMBench EN | MMBench CN | VQAv2 |
|--------|-----------|----------|----------|------|------------|------------|-------|
| Baseline-7B | 83.3 | 80.1 | 78.2 | 1504.6 | 62.2 | 57.7 | 77.4 |
| +BTL-Generation | 82.7 | 80.3 | 79.2 | 1489.4 | 59.2 | 54.2 | 75.9 |
| +BTL-Caption | 84.5 | 81.0 | 79.9 | 1499.6 | 59.7 | 54.5 | 76.6 |
| +BTL-Gen+Cap | 84.9 | 81.9 | 80.6 | 1505.4 | 60.3 | 57.4 | 76.2 |
| **MMGrounded-PostAlign-7B** | **86.6** | **84.2** | **82.3** | **1514.3** | **63.9** | **58.7** | **78.8** |
| Baseline-13B | 85.4 | 82.2 | 79.2 | 1520.3 | 66.8 | 62.2 | 79.1 |
| +BTL-Generation | 84.4 | 80.9 | 78.3 | 1501.7 | 65.9 | 62.4 | 78.0 |
| +BTL-Caption | 85.1 | 81.7 | 78.8 | 1509.2 | 65.2 | 61.4 | 79.4 |
| +BTL-Gen+Cap | 86.8 | 83.4 | 81.7 | 1504.9 | 65.4 | 62.2 | 79.2 |
| **MMGrounded-PostAlign-13B** | **88.9** | **87.3** | **85.6** | **1517.4** | **68.9** | **63.2** | **79.9** |

**关键发现**: 
- 显式视觉定位模块显著优于 BTL 方法
- BTL-Generation 损害 MLLM 推理能力 (MME 下降、VQAv2 下降)
- PostAlign 在保留/提升泛化推理能力的同时，大幅抑制幻觉

### Table 3: 选择性推理消融 (ReasonSeg)

| Method | Easy gIoU | Easy cIoU | Medium gIoU | Medium cIoU | Hard gIoU | Hard cIoU |
|--------|-----------|-----------|-------------|-------------|-----------|-----------|
| Baseline-7B | 67.7 | 66.4 | 51.2 | 50.2 | 47.0 | 46.3 |
| +Pre-Reasoning | 67.3 | 66.7 | 57.2 | 58.1 | 57.0 | 56.5 |
| +Inter-Reasoning | 67.0 | 65.8 | 54.1 | 54.3 | 52.5 | 51.1 |
| **+Selective-Reasoning** | **68.3** | **68.1** | **58.0** | **58.7** | **57.2** | **56.8** |

**关键发现**: 
- 预推理 (Pre-Reasoning) 在复杂查询上表现好，但需多次推理轮次
- 并发推理 (Inter-Reasoning) 效率高但性能不及
- **选择性推理在简单查询上避免过思考，在复杂查询上确保充分推理，单轮推理达到最优**

### Table 4: REC / RES 任务 (RefCOCO 系列)

| Model | RefCOCO REC | RefCOCO RES | RefCOCO+ REC | RefCOCO+ RES | RefCOCOg REC | RefCOCOg RES |
|-------|-------------|-------------|--------------|--------------|--------------|--------------|
| Kosmos-2 | 52.3 | - | 45.4 | - | 60.5 | - |
| LISA-7B (ft) | - | 74.9 | - | 65.1 | - | 67.9 |
| LLaVASeg-7B (ft) | - | 76.2 | - | 65.7 | - | 69.8 |
| MiniGPT v2-7B | 88.0 | - | 79.5 | - | 84.1 | - |
| Shikra-7B | 87.0 | - | 81.6 | - | 82.2 | - |
| Ferret-7B | 87.4 | - | 80.7 | - | 83.9 | - |
| LLaVA-Grounding-7B | 89.1 | 77.1 | 81.6 | 68.7 | 84.8 | 71.5 |
| GLaMM | - | 79.5 | - | 72.6 | - | 74.2 |
| GSVA-7B (ft) | 86.2 | 77.2 | 72.8 | 65.9 | 81.5 | 72.7 |
| **MMGrounded-PostAlign-7B** | 88.2 | **77.9** | 78.4 | **68.2** | 83.3 | **73.2** |
| **MMGrounded-PostAlign-13B** | **89.2** | **79.7** | **80.1** | **70.9** | **85.3** | **74.8** |

**关键发现**: 虽未专门优化定位任务（未增加定位训练数据），PostAlign 在 REC/RES 上仍达到有竞争力甚至最优的性能——说明 MLLM 与视觉定位模块实现双向增益。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[HaloQuest]] | 多类型幻觉评测 | 含 False Premise / Visually Challenging / Insufficient Context 子类 | 幻觉评测 |
| [[POPE]] | 9000 样本 (3 设置) | Random / Popular / Adversarial 三种采样策略 | 幻觉评测 |
| [[VQAv2]] | ~1.1M 训练 | 通用视觉问答 | 泛化能力 |
| [[MMBench]] | ~3000 样本 | 涵盖 EN/CN，多层能力维度 | 综合评测 |
| MME | 14 子任务 | 感知 + 认知能力 | 综合评测 |
| [[RefCOCO]] 系列 | 指称定位 | RefCOCO / RefCOCO+ / RefCOCOg，含 REC 和 RES 任务 | 定位评测 |
| [[ReasonSeg]] | 推理分割 | 需要推理的语义分割，分 Easy/Medium/Hard | 推理+定位 |

### 实现细节

- **Backbone**: [[LLaVA|LLaVA-1.5-7B]] / LLaVA-1.5-13B + [[SAM|ViT-H SAM]]
- **微调**: LoRA (冻结 MLLM 主参数，冻结 SAM)；多任务解码器、LLM head、投影层全参数微调
- **优化器**: AdamW, lr=0.0003, weight_decay=0
- **学习率调度**: WarmupDecayLR, 100 warmup iterations
- **Batch Size**: per-device 2, gradient accumulation 10
- **训练引擎**: [[DeepSpeed]]
- **训练数据**: 自建多模态数据集，标注 `<SIMPLE>` / `<COMPLEX>` 推理类型 + `<REJ>` 负样本

### 五大发现

1. **语言先验覆盖视觉信息**: 幻觉 token 概率在深层 Transformer 才追上非幻觉 token；移除图像后幻觉 token 重叠率 **89.2%**，证明语言先验主导幻觉
2. **显式视觉定位有效抑制幻觉**: HaloQuest 消融显示 `<SEG>` + `<DET>` + `<REJ>` 比 baseline 在 False Premise 上提升 **31.2 个百分点**
3. **后对齐方法保留推理和泛化能力**: BTL-Generation 损害 MME/VQAv2，但 PostAlign 对所有指标均有提升
4. **选择性推理在效率和准确性上最优**: 简单查询避免过思考，复杂查询保证充分推理，单轮推理
5. **MLLM + 视觉定位双向增益**: PostAlign 在 RefCOCO 和 ReasonSeg 上达到有竞争力的零样本定位性能

---

## 批判性思考

### 优点
1. **深刻的实证分析**: 89.2% 幻觉重叠率实验简洁有力地证明了语言先验的主导作用，为方法设计提供了坚实动机
2. **"后对齐"设计理念优雅**: 不改变预训练 MLLM，以 LoRA + 附加模块方式实现，即插即用，保护已有能力
3. **全面的消融实验**: 覆盖视觉定位各组件、BTL vs 显式定位、三种推理策略，实验设计严谨
4. **双向增益的洞察**: 发现定位模块不仅辅助 MLLM，MLLM 也反哺定位能力——这种协同效应在多数工作中被忽略

### 局限性
1. **无代码开源**: 论文未承诺开源代码和模型，可复现性受限
2. **仅基于 LLaVA-1.5**: 未在其他 MLLM 架构（如 Qwen-VL, InternVL）上验证，泛化性有待证明
3. **训练数据细节缺失**: 自建训练数据集的具体组成、规模、来源描述不够详细
4. **`<REJ>` 泛化边界不明确**: 负样本拒绝机制对反概念场景有效，但在更 subtle 的属性幻觉和空间关系幻觉上效果未单独分析

### 潜在改进方向
1. 在多架构 MLLM (Qwen2-VL, InternVL2, GPT-4V) 上验证通用性
2. 探索将 `<REJ>` 机制推广到属性级拒绝（"红色的苹果"→"有苹果但不是红色"→部分拒绝）
3. 研究如何自动化生成高质量负样本，减少人工标注成本
4. 将选择性推理扩展到更多粒度级别（而非仅 SIMPLE/COMPLEX 二分类）

### 可复现性评估
- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整 (架构、超参数、优化器、学习率)
- [ ] 数据集可获取 (训练数据来源不详)

---

## 关联笔记

### 基于
- [[LLaVA]]: MLLM 骨干网络
- [[SAM]]: 视觉定位编码器
- [[LoRA]]: 参数高效微调方法
- [[DeepSpeed]]: 分布式训练引擎

### 对比
- [[POPE]]: 幻觉评测基准，BTL 方法在此对比
- [[MMBench]]: 综合评测基准
- [[HaloQuest]]: 多类型幻觉评测
- BTL (Bounding-box as Language Token): 定位策略对比基线
- Shikra / Ferret / LISA / GLaMM / GSVA: MLLM 接地模型对比

### 方法相关
- [[多模态对齐]]: 核心方法论
- [[视觉定位]]: 视觉定位模块
- [[语言先验]]: 需要对抗的偏置来源
- [[虚假相关]]: 导致幻觉的根本原因
- [[共现偏差]]: 幻觉的具体表现形式
- [[负样本拒绝]]: 幻觉抑制机制
- [[选择性推理]]: 推理优化策略
- [[分割掩码]]: 像素级定位输出
- [[边界框检测]]: 目标级定位输出

### 硬件/数据相关
- [[DeepSpeed]]: 分布式训练
- [[VQAv2]]: 泛化评测
- [[RefCOCO]]: 定位评测
- [[ReasonSeg]]: 推理定位评测

---

## 速查卡片

> [!summary] PostAlign: Multimodal Grounding as a Corrective Lens for MLLMs
> - **核心**: 后对齐框架用多模态定位作"修正镜头"，负样本拒绝 + 选择性推理双管齐下抑制幻觉
> - **方法**: LLaVA-1.5 + ViT-H SAM, LoRA 微调, `<LOC>` / `<REJ>` 视觉 token + `<SIMPLE>` / `<COMPLEX>` 推理分类
> - **结果**: HaloQuest False Premise +31.2pp, POPE Adv +4.1pp (7B), 幻觉重叠率 89.2% 证实语言先验主导
> - **代码**: 未开源

---

*笔记创建时间: 2026-05-20*
