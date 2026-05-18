---
title: "Qwen3-VL Technical Report"
method_name: "Qwen3-VL"
authors: [Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, Ke Zhu]
year: 2025
venue: arXiv
tags: [multimodal, vision-language-model, mixture-of-experts, chain-of-thought, spatial-understanding, video-understanding, document-understanding]
zotero_collection: 多模态/QWen/MML
image_source: mixed
arxiv_html: https://arxiv.org/html/2511.21631
created: 2026-05-18
---

# 论文笔记：Qwen3-VL Technical Report

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Alibaba Qwen Team |
| 日期 | November 2025 |
| 项目主页 | https://huggingface.co/Qwen / https://github.com/QwenLM/Qwen3-VL |
| 对比基线 | [[Qwen2.5-VL]], [[Gemini 2.5 Pro]], GPT-5, Claude Opus 4.1 |
| 链接 | [arXiv](https://arxiv.org/abs/2511.21631) / [Code](https://github.com/QwenLM/Qwen3-VL) |

---

## 一句话总结

> 最强 Qwen 系列 VLM，支持 256K 交错上下文、Dense/MoE 双架构、Thinking/Non-thinking 双模式，在视觉问答、数学推理、文档理解、视频理解、具身空间理解等全面 SOTA。

---

## 核心贡献

1. **架构三大升级**: [[Interleaved MRoPE]] 优化位置编码频谱、[[DeepStack]] 多层 ViT 特征注入增强视觉-语言对齐、文本时间戳替代 T-RoPE 实现精确视频时间定位
2. **训练策略创新**: 四阶段预训练(67B→1T→1T→100B tokens)、平方根重加权损失平衡文本与多模态、SFT+蒸馏+RL三阶段后训练、Thinking/Non-thinking 双叉
3. **全面 SOTA 性能**: 在 MMMU、MathVista、MathVision、MMT-Bench、HallusionBench、OCRBench、MVBench 等广泛 benchmark 上达到或超越同级最强模型
4. **多尺寸全覆盖**: 4个Dense(2B/4B/8B/32B) + 2个MoE(30B-A3B/235B-A22B)，开源 Apache 2.0

---

## 问题背景

### 要解决的问题
在视觉-语言模型能力快速发展的背景下，如何在提升多模态感知与推理能力的同时，不损害底层 LLM 的纯文本能力，并有效扩展到长上下文（256K）的多模态理解场景。

### 现有方法的局限
1. [[Qwen2.5-VL]] 使用的 [[MRoPE]] 将嵌入维度按 t/h/w 分块分配旋转频率，导致频谱不平衡，长视频理解退化
2. 视频时间编码依赖 T-RoPE 绝对时间对齐，在长视频中产生过于稀疏的时间位置 ID，且需要大量跨帧率采样的训练数据
3. 视觉-语言对齐仅依赖单一的 vision-language merger，缺乏多层级的视觉特征融合

### 本文的动机
通过重新设计位置编码（交错频谱）、引入多层视觉特征注入（DeepStack）、改用显式文本时间戳，系统性提升 VL 模型的三大核心能力：纯文本理解、长上下文理解、多模态推理。

---

## 方法详解

### 模型架构

Qwen3-VL 采用 **vision encoder + MLP-based vision-language merger + LLM** 三模块架构：

- **Vision Encoder**: [[SigLIP|SigLIP-2]] SO-400M（2B/4B用SigLIP2-Large），支持动态分辨率（2D-RoPE + 位置嵌入插值）
- **LLM Backbone**: [[Qwen3]] 系列，Dense (2B/4B/8B/32B) + MoE (30B-A3B, 235B-A22B)
- **Merger**: 两层 [[MLP]]，将 2x2 视觉特征压缩为单个视觉 token，对齐到 LLM 隐藏维度
- **总参数**: 旗舰 235B-A22B（235B总参、22B激活/Token），支持 256K 上下文窗口

### 核心模块

#### 模块1: Interleaved MRoPE

**设计动机**: [[Qwen2-VL]] 引入的 [[MRoPE]] 将嵌入维度划分为时间(t)、水平(h)、垂直(w)三个子空间，各自分配不同旋转频率，导致频谱不平衡，损害长视频理解能力。

**具体实现**:
- 将 t、h、w 组件在嵌入维度上交错分布（而非连续分块），使每个时空轴均匀覆盖低频和高频带
- 结果：平衡的频率谱缓解了原始频谱偏差，显著提升长距离位置建模能力

#### 模块2: DeepStack

**设计动机**: 单一 merger 层仅能传递高层视觉语义，丢失了 [[Vision Transformer|ViT]] 中间层的丰富视觉信息（从低级纹理到高级语义）。

**具体实现**:
- 从 ViT 的三个不同层级提取视觉特征（浅层/中层/深层）
- 每个层级配备专用的 vision-language merger，将多层级特征投影为视觉 token
- 将不同层级的视觉 token 以残差连接方式注入 LLM 的前三层对应隐藏状态
- 不增加额外上下文长度

#### 模块3: Video Timestamp (文本时间戳)

**设计动机**: [[Qwen2.5-VL]] 的 T-RoPE 时间对齐有两个局限：(1) 时间位置ID直接绑定绝对时间，长视频中产生过大稀疏的ID；(2) 需要大量跨帧率采样的训练数据，成本高。

**具体实现**:
- 用显式文本时间戳标记视频片段组，格式如 `<3.0 seconds>`
- 同时生成秒和 HMS (时:分:秒) 两种格式，确保模型学习多样的时间编码表示
- 代价：略微增加上下文长度，但带来更精确的时间感知

#### 模块4: Square-Root Reweighting

**设计动机**: 传统 per-sample loss 无法平衡文本数据和视觉-语言数据的贡献差异。

**具体实现**:
- 从 per-sample loss 改为 **平方根归一化的 per-token loss**
- 提升多模态性能同时不损害文本能力

---

## 训练流程

### 预训练四阶段

| 阶段 | 目标 | 训练参数 | Token 预算 | 序列长度 |
|------|------|----------|-----------|---------|
| S0 | Vision-Language Alignment | 仅 Merger | 67B | 8,192 |
| S1 | Multimodal Pre-Training | 全部 | ~1T | 8,192 |
| S2 | Long-Context Pre-Training | 全部 | ~1T | 32,768 |
| S3 | Ultra-Long-Context Adaptation | 全部 | 100B | 262,144 |

### 后训练三阶段

1. **SFT**: 32K → 256K 两阶段，分 Non-thinking 和 Thinking (CoT) 两种格式，约120万样本
2. **Strong-to-Weak Distillation**: 仅用文本数据蒸馏 LLM backbone，含 off-policy + on-policy 两阶段
3. **Reinforcement Learning**: 推理RL（[[SAPO]]算法，~30K queries）+ 通用RL（多任务，含规则奖励+模型奖励）

---

## 关键公式

### 公式1: [[Square-Root Reweighting|平方根重加权损失]]

$$
\mathcal{L} = \sum_{i} \frac{\mathcal{L}_i}{\sqrt{N_i}}
$$

**含义**: 将每个样本的损失按其 token 数 $N_i$ 的平方根进行归一化，防止长序列样本主导训练梯度。

**符号说明**:
- $\mathcal{L}_i$: 第 i 个样本的原始损失
- $N_i$: 第 i 个样本的 token 数量

---

## 关键图表

### Figure 1: Qwen3-VL Framework Overview

![[Qwen3-VL_fig1_framework.png]]

**说明**: Qwen3-VL 整体框架。Vision Encoder 处理动态原生分辨率视觉输入，通过 [[DeepStack]] 机制将多层 ViT 特征注入 LLM 各层。[[Interleaved MRoPE]] 编码多模态位置信息，文本时间戳标记视频时间结构。

### Figure 2: Multilingual OCR Performance

![[Qwen3-VL_fig2_ocr.png]]

**说明**: 39种语言的 OCR 性能分布图。32/39 种语言超过 70% 准确率（实用可用阈值），展示了强大的多语言能力（vs [[Qwen2.5-VL]] 的 10 种语言）。

### Figure 3: Needle-in-a-Haystack Heatmap

![[Qwen3-VL_fig3_niah.png]]

**说明**: Qwen3-VL-235B-A22B-Instruct 的视频 NIAH 测试热力图。在原生 256K 上下文（约30分钟视频）内达到 100% 准确率；通过 [[YaRN]] 外推到 1M tokens（约2小时视频）仍保持 99.5% 准确率。

### Table 1: 预训练阶段配置

| 阶段 | 目标 | 训练参数 | Token预算 | 序列长度 |
|------|------|----------|-----------|---------|
| S0 | VL Alignment | 仅 Merger | 67B | 8,192 |
| S1 | Multimodal Pre-Training | All | ~1T | 8,192 |
| S2 | Long-Context Pre-Training | All | ~1T | 32,768 |
| S3 | Ultra-Long-Context Adaptation | All | 100B | 262,144 |

**说明**: 从基础对齐到超长上下文的渐近训练策略，序列长度逐级翻倍。

### Table 2: Qwen3-VL-235B-A22B vs 顶级模型 (核心指标)

| Benchmark | Qwen3-VL Thinking | Qwen3-VL Instruct | Gemini 2.5 Pro Thinking | GPT-5 High | Claude Opus 4.1 |
|-----------|-------------------|-------------------|------------------------|------------|-----------------|
| MMMU | 80.6 | 78.7 | 81.7 | 84.2 | 78.4 |
| MathVista_mini | **85.8** | 84.9 | 82.7 | 81.3 | 75.5 |
| MathVision | **74.6** | 66.5 | 73.3 | 70.9 | 57.7 |
| MMBench-EN | 88.8 | 89.3 | 90.1 | 83.8 | 83.0 |
| MMStar | 78.7 | 78.4 | 77.5 | 76.4 | 72.1 |
| HallusionBench | **66.7** | 63.2 | 63.7 | 65.7 | 60.4 |
| MIA-Bench | **92.7** | 91.3 | 92.3 | 92.4 | 91.2 |
| OCRBench | 875 | **920** | 866 | 810 | 764 |
| DocVQA_test | **96.5** | 97.1 | 92.6 | 91.5 | 92.5 |
| RefCOCO-avg | **92.1** | 91.9 | 74.6 | 66.8 | - |
| MVBench | **75.2** | 76.5 | 69.9 | 75.3 | 61.4 |
| Video-MME | 79.0 | 79.2 | 85.1 | 84.7 | 75.6 |
| EmbSpatialBench | **84.3** | 83.1 | 79.1 | 82.9 | 69.2 |
| OSWorld | **38.1** | 31.6 | - | - | 44.4 |

**说明**: Qwen3-VL 在数学视觉推理和空间理解上全面领先；在文档/OCR 及 grounding 上也基本 SOTA。

### Table 11: Vision Encoder 消融

| ViT | ImageNet-1K | ImageNet-V2 | ImageNet-A | ImageNet-R | OmniBench | OCRB | AI2D | InfoVQA |
|-----|-------------|-------------|------------|------------|-----------|------|------|---------|
| SigLIP-2 | 84.2 | 78.6 | 87.0 | 96.1 | 36.9 | 77.2 | 74.1 | 65.3 |
| **Qwen3-ViT** | **84.6** | **78.8** | **87.1** | 95.7 | **45.5** | **78.7** | **76.2** | **67.0** |

**说明**: Qwen3-ViT 是持续训练的增强版 SigLIP-2，在保持标准 benchmark 性能的同时大幅提升 OmniBench（世界知识）上的表现。

### Table 12: DeepStack 消融

| Method | AVG | AI2D | OCRB | InfoVQA | ChartQA | DocVQA | MMMU | MMStar |
|--------|-----|------|------|---------|---------|--------|------|--------|
| Baseline | 74.7 | 81.8 | 81.0 | 71.9 | 81.5 | 89.5 | 52.9 | 55.5 |
| **DeepStack** | **76.0** | **83.2** | **83.6** | **74.2** | **83.3** | **91.1** | **54.1** | **57.7** |

**说明**: DeepStack 带来 1.3 个百分点的平均提升，尤其显著提升 InfoVQA (+2.3) 和 OCRB (+2.6) 等细粒度视觉理解任务。

---

## 实验

### 评估体系

论文在超过 **50 个 benchmark** 上进行评估，覆盖以下类别：
- **通用 VQA**: MMBench, RealWorldQA, MMStar, SimpleVQA
- **多模态推理**: MMMU, MathVista, MathVision, We-Math, ZeroBench, LogicVista, VisuLogic, VisualPuzzles
- **对齐/主观**: HallusionBench, MM-MT-Bench, MIA-Bench
- **文档理解**: DocVQA, InfoVQA, OCRBench, OmniDocBench, CharXiv, MMLongBenchDoc
- **2D/3D Grounding**: RefCOCO, CountBench, ODinW-13, Omni3D
- **具身/空间**: ERQA, VSI-Bench, EmbSpatialBench, RefSpatialBench, RoboSpatialHome
- **视频**: Video-MME, MVBench, MLVU, LVBench, VideoMMMU, MMVU
- **多模态 Agent**: ScreenSpot Pro, OSWorld, AndroidWorld, WindowsAA
- **文本任务**: MMLU-Pro, AIME-25, LiveCodeBench, IFEval, Arena-Hard v2

### 预训练数据

| 类别 | 关键数据/规模 | 核心方法 |
|------|-------------|---------|
| Image Caption | 大规模中英双语 | Qwen2.5-VL-32B recaption 流水线 + 语义聚类增强 |
| Interleaved Text-Image | 真实网页 + 书籍级 | 领域分类过滤 + 超长书页拼接(256K) |
| Knowledge | 12+语义类别的实体 | 重要性采样 + LLM描述增强 |
| OCR | 3000万自建 + 29种语言合成 | Coarse-to-Fine 流水线(无人工标注) |
| Document Parsing | 300万CC PDF + 400万内部 | QwenVL-HTML / QwenVL-Markdown 双格式 |
| Grounding | Box+Point+Counting | 三阶段自动标注 + [0,1000] 归一化坐标 |
| Spatial/3D | 关系/可负担性/动作规划/3D bbox | 多传感器融合 + 虚拟相机坐标系统一 |
| Video | 多源均衡 + Dense Caption | 短到长 caption synthesis + 时空Grounding |
| STEM | 600万图表+6000万习题+1200万CoT | Divide-and-Conquer + 拒绝采样 |
| Agent | GUI + Function Calling + Search | Self-evolving trajectory + Tool-integrated RL |

### 实现细节

- **Vision Encoder**: SigLIP-2 SO-400M，动态分辨率继续训练
- **LLM Backbone**: Qwen3 全系列
- **Infrastructure**: 阿里云 PAI-Lingjun，Megatron-LM (TP+PP+CP+EP+ZeRO-1 DP)，最多 10000 GPUs
- **推理部署**: vLLM (PagedAttention) 或 SGLang

### Text-Centric 关键发现

Qwen3-VL-235B-A22B-Instruct 纯文本能力甚至能超越 DeepSeek V3 0324 和 Claude Opus 4.1（非思考模式），在其基座 [[Qwen3]] 之上还有提升，部分 benchmark 甚至达到或超越同规模的纯文本模型。

---

## 批判性思考

### 优点
1. **全面性史无前例**: 50+ benchmark 评估，涵盖几乎所有 VLM 子方向，Dense+MoE+Thinking/Non-thinking 完整覆盖
2. **纯文本不退化**: VLM 训练不仅不损害、反而增强了纯文本能力，这是业界罕见的正反馈
3. **开源诚意足**: Apache 2.0 + 全尺寸开源 + 完整训练方法公开，包括数据 curation 的详细配方
4. **架构创新务实**: 三大架构改进均源自对 Qwen2.5-VL 的"缺陷修补"，体现了工程迭代的深度

### 局限性
1. **数据不可复现**: 大规模内部数据未公开，其他团队无法完全复现训练
2. **长视频评估帧数不公**: 评估中各模型输入帧数不统一(512 vs 256 vs 100)，部分比较不公平
3. **计算成本未披露**: 总训练 FLOPs 和碳排放未报告
4. **Thinking 模式开销**: Thinking 变体虽强但推理开销大，论文未提供延迟/成本对比
5. **安全评估缺失**: 缺少红队测试、有害内容、偏见等方面的评估

### 潜在改进方向
1. 统一理解-生成架构（论文在 Conclusion 中提及正在探索）
2. 更高效的长上下文训练策略（当前 256K 需要成本高昂的 S3 阶段）
3. 交互式感知和实时多模态控制

### 可复现性评估
- [x] 代码开源 (GitHub)
- [x] 预训练模型 (HuggingFace/ModelScope)
- [x] 训练细节完整 (超参、数据配方)
- [ ] 数据集可获取 (大部分自建数据未公开)

---

## 关联笔记

### 基于
- [[Qwen2.5-VL]]: 前代 VLM，本作在架构上的三项改进均针对其缺陷
- [[Qwen3]]: LLM backbone 基座
- [[Qwen2-VL]]: 最早引入 MRoPE 的 Qwen 系列 VLM

### 对比
- [[Gemini 2.5 Pro]]: 最强竞品，Thinking 模式在少数 benchmark 上领先
- GPT-5: OpenAI 旗舰，high reasoning 模式在 MMMU 上最强
- Claude Opus 4.1: Anthropic 旗舰，多模态 Agent 有竞争力

### 方法相关
- [[Interleaved MRoPE]]: 交错式多模态旋转位置编码
- [[DeepStack]]: 多层视觉特征注入
- [[SAPO]]: 强化学习算法
- [[Chain-of-Thought|CoT]]: 思维链推理
- [[Mixture-of-Experts|MoE]]: 混合专家架构
- [[Vision Transformer|ViT]]: 视觉编码器骨干
- [[SigLIP]]: 视觉编码器架构
- [[MRoPE]]: 多模态旋转位置编码
- [[YaRN]]: 位置编码外推方法

### 硬件/数据相关
- [[PAI]]: 阿里云 PAI-Lingjun 训练平台

---

## 速查卡片

> [!summary] Qwen3-VL Technical Report
> - **核心**: 最强 Qwen 系列 VLM，256K 上下文 + Dense/MoE 双架构 + Thinking/Non-thinking 双模式
> - **方法**: Interleaved MRoPE + DeepStack + 文本时间戳 + 平方根重加权 + SFT/Distill/RL 后训练
> - **结果**: 50+ benchmark 全方位 SOTA，纯文本能力随 VL 训练甚至增强
> - **代码**: https://github.com/QwenLM/Qwen3-VL

---

*笔记创建时间: 2026-05-18*
