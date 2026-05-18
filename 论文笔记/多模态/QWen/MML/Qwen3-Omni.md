---
title: "Qwen3-Omni Technical Report"
method_name: "Qwen3-Omni"
authors: [Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, Junyang Lin]
year: 2025
venue: arXiv
tags: [omni-model, multimodal, speech-synthesis, audio-understanding, large-language-model, mixture-of-experts]
zotero_collection: 多模态/QWen/MML
image_source: online
created: 2026-05-18
---

# 论文笔记：Qwen3-Omni Technical Report

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Qwen Team (Alibaba) |
| 日期 | September 2025 |
| 项目主页 | https://github.com/QwenLM/Qwen3-Omni |
| 对比基线 | [[Qwen2.5-Omni]], [[Gemini-2.5-Pro]], [[GPT-4o]] |
| 链接 | [arXiv](https://arxiv.org/abs/2509.17765) / [GitHub](https://github.com/QwenLM/Qwen3-Omni) / [Demo](https://huggingface.co/spaces/Qwen/Qwen3-Omni-Demo) |

---

## 一句话总结

> 首个在全模态（文本、图像、音频、视频）上无性能退化的端到端多模态模型，基于 Thinker-Talker MoE 架构，支持流式语音合成和 234ms 首包延迟。

---

## 核心贡献

1. **全模态无退化**: 首次证明端到端多模态训练可以不牺牲任何单一模态的性能，30B-A3B 规模的 Omni 模型在文本和视觉上匹配同规模单模态 Qwen 模型
2. **Thinker-Talker MoE 架构**: 将 Thinker 和 Talker 均升级为 [[Mixture-of-Experts|MoE]] 设计，Talker 解耦文本表征、仅依赖多模态特征，支持独立系统提示
3. **AuT 音频编码器**: 全新 [[Audio Transformer|AuT]] 编码器，在 2000 万小时监督音频上从头训练，采用块状窗口注意力实现实时 prefill 缓存
4. **流式多码本语音生成**: 多码本自回归方案 + 轻量 [[ConvNet Vocoder|Code2Wav]]，替代 block-wise DiT，实现首帧即时合成，首包延迟降至 234ms
5. **Thinking + Captioner 变体**: Thinker 支持全模态推理（含音频-视频）、Captioner 为任意音频生成详细低幻觉描述

---

## 问题背景

### 要解决的问题

当前 LLM 中心的多模态模型普遍存在**模态权衡**（modality trade-off）：某一模态的性能提升往往伴随其他模态的退化。Qwen3-Omni 旨在实现真正的全模态集成训练，在不牺牲任何模态性能的前提下，利用跨模态协同增强整体能力。

### 现有方法的局限

- [[Qwen2.5-Omni]] 虽已采用 Thinker-Talker 架构，但使用 Whisper 编码器、单码本语音生成、block-wise DiT 声码器，延迟较高，语言覆盖有限
- 其他 Omni 模型（[[GPT-4o]]、[[Gemini-2.5-Pro]]）为闭源，且未见系统性证明各模态无退化
- 目前社区缺乏通用音频描述模型

### 本文的动机

人类学习天然依赖多模态协同——互补特化和跨模态协同提升学习效率。作者假设：**在 LLM 预训练早期阶段混合单模态和跨模态数据**，是实现无退化多模态训练的关键。

---

## 方法详解

### 模型架构

Qwen3-Omni 采用 **[[Thinker-Talker Architecture|Thinker-Talker]] [[Mixture-of-Experts|MoE]]** 架构：

- **Thinker**: 30B-A3B MoE Transformer，负责跨模态理解与文本生成
- **Talker**: 3B-A0.3B MoE Transformer，从 Thinker 接收多模态高层表征，生成流式语音
- **Audio Encoder**: [[Audio Transformer|AuT]]，650M 参数，12.5Hz  token rate
- **Vision Encoder**: [[SigLIP|SigLIP2-So400m]]，~543M 参数
- **MTP Module**: 80M Dense Transformer，多码本残差预测
- **Code2Wav**: 200M ConvNet，流式波形重建
- **总激活参数**: ~33B-A3.6B

关键设计变更（相比 [[Qwen2.5-Omni]]）：
1. Talker 不再消费 Thinker 的文本表征，仅依赖音频和视觉多模态特征。理由是：(i) 文本内容上离散 token 和嵌入信息等价；(ii) 音频-视频协调语音生成（如保留韵律/音色）需要多模态条件；(iii) 解耦使外部模块（RAG、函数调用、安全过滤）可干预 Thinker 输出再供给 Talker
2. Thinker 和 Talker 可使用不同系统提示，独立控制回复风格和语音风格
3. 音视频表征通过 TM-RoPE 的绝对时间 ID 直接对齐，不再分块的固定 2 秒 chunk

### 核心模块

#### 模块1: Audio Transformer (AuT)

**设计动机**: 替代 Whisper，训练更强的通用音频表征，同时支持实时 prefill 缓存

**具体实现**:
- Attention-encoder-decoder 自回归模型
- 训练数据：2000 万小时监督音频（80% 中英伪标签 ASR + 10% 其他语言 ASR + 10% 音频理解）
- 使用 Conv2D 将滤波器组特征 8 倍下采样至 12.5Hz
- 采用 [[Flash Attention|flash attention]] 动态窗口尺寸（1-8 秒），平衡实时 prefill 缓存与离线性能
- AuT 编码器作为 Qwen3-Omni 的音频编码器，参数约 0.6B

#### 模块2: Time-aligned Multimodal Rotary Position Embedding (TM-RoPE)

**设计动机**: 扩展 [[Multimodal Rotary Position Embedding|M-RoPE]] 以支持绝对时间对齐

**具体实现**:
- 将 [[Rotary Position Embedding|RoPE]] 分解为三个维度：temporal (24 angles)、height (20)、width (20)，交错分配
- Text: 三个维度共享相同位置 ID，等价于 1D RoPE
- Audio: 共享位置 ID + 绝对时间编码（每 80ms 一个 temporal ID）
- Image: 恒定 temporal ID，各行/列独立 height/width ID
- Video: 每帧单调递增 temporal ID，按实际时间戳动态调整至 80ms 分辨率
- 多模态流：位置编号连续，后一模态从前一模态最大位置 ID+1 开始

#### 模块3: 流式多码本语音生成

**设计动机**: 实现首码即合成，最小化首包延迟

**具体实现**:
- 使用 [[Residual Vector Quantization|RVQ]] token（多码本）
- Talker backbone 聚合当前帧的码本特征，线性头预测第 0 层码本
- [[Multi-Token Prediction|MTP]] 模块自回归生成所有残差码本
- Code2Wav: 轻量因果 ConvNet，仅关注左上下文，替代 block-wise DiT
- 输出编解码率 12.5Hz，单帧即可即时合成语音

### 流式与并发设计

| 机制 | 作用 |
|------|------|
| [[Chunked Prefilling|分块 Prefill]] | Thinker 完成当前块 prefill 后，立即异步 prefill Talker，同时 Thinker 处理下一块 |
| MoE 架构 | 减少长序列 KV cache 的 IO 消耗，提高 TPS 和并发 |
| 流式多码本 | Talker 生成首 token 后 MTP 立即预测剩余码本，Code2Wav 即时合成 |
| 轻量 MTP + ConvNet | 低 FLOPs，支持批处理推理，硬件加速广泛 |

---

## 关键公式

### TM-RoPE 位置编码

[[Multimodal Rotary Position Embedding|TM-RoPE]] 将位置嵌入分解为 temporal, height, width 三个维度：

$$
\mathbf{q}_m^{\top} \mathbf{k}_n = \sum_{d \in \{t, h, w\}} \text{RoPE}(\mathbf{q}_m^{(d)}, \mathbf{k}_n^{(d)}, m_d - n_d)
$$

**含义**: 对每个维度分别应用 RoPE，temporal 维度使用最多 24 个高频角度建模局部时序变化，同时保留长程外推能力。

**符号说明**:
- $\mathbf{q}_m, \mathbf{k}_n$: 位置 m 的 query 和位置 n 的 key
- $d \in \{t, h, w\}$: temporal, height, width 三个维度
- $m_d, n_d$: 各维度的位置 ID

### 语音生成: 多码本自回归

Talker 的层次化预测方案：

$$
P(c_t^0 | \mathbf{h}_t) \cdot \prod_{k=1}^{K-1} P(c_t^k | \mathbf{h}_t, c_t^{<k})
$$

**含义**: Talker backbone 先预测第 0 层码本 $c_t^0$，MTP 模块再依次预测残差码本 $c_t^1, \dots, c_t^{K-1}$。

**符号说明**:
- $c_t^k$: 第 t 帧的第 k 层码本 token
- $\mathbf{h}_t$: Talker backbone 输出的当前帧隐藏状态
- $K$: 码本总层数

### 实时因子 (RTF)

生成实时因子定义为：

$$
\text{RTF} = \frac{T_{\text{Thinker}} + T_{\text{Talker}} + T_{\text{MTP}} + T_{\text{Codec}}}{80\text{ms}}
$$

**含义**: 生成一个 80ms 音频帧所需的计算时间与音频时长之比。RTF < 1 表示可连续流式输出。

**符号说明**:
- $T_{\text{Thinker}}, T_{\text{Talker}}$: Thinker/Talker 生成一个 token 的耗时
- $T_{\text{MTP}}$: MTP 模块单 token 处理时间
- $T_{\text{Codec}}$: Codec 解码器单码本时间

---

## 关键图表

### Figure 1: Qwen3-Omni 能力展示

![[Qwen3-Omni_fig1_teaser.png]]

**说明**: Qwen3-Omni 的统一端到端能力展示。支持文本、音频、图像、视频多模态输入，可生成实时文本或语音回复。包含四种交互场景：数学推理（Thinking mode）、语音翻译、会议转录、语音对话。

### Figure 2: 模型架构总览

![[Qwen3-Omni_fig2_architecture.png]]

**说明**: Qwen3-Omni 的 [[Thinker-Talker Architecture|Thinker-Talker]] 架构。Thinker 负责文本生成，Talker 从 Thinker 直接接收高层多模态表征进行流式语音合成。Talker 自回归预测多码本序列，每步 MTP 模块输出当前帧的残差码本，Code2Wav 逐步合成波形。

### Figure 3: AuT 编码器

![[Qwen3-Omni_fig3_aut.png]]

**说明**: [[Audio Transformer|AuT]] 的 attention-encoder-decoder 架构。在 2000 万小时监督音频上训练，采用 block-wise 窗口注意力实现实时 prefill 缓存。Qwen3-Omni 使用其编码器以 12.5Hz 获取通用音频表征。

### Table 1: 架构设计与首包延迟

| Module | Architecture | Params | Streaming |
|--------|-------------|--------|-----------|
| Audio Encoder | AuT | 650M | Yes |
| Vision Encoder | SigLIP2-So400M | 540M | Yes |
| Thinker | MoE Transformer | 30B-A3B | Yes |
| Talker | MoE Transformer | 3B-A0.3B | Yes |
| MTP | Dense Transformer | 80M | Yes |
| Code2Wav | ConvNet | 200M | - |

**端到端首包延迟 (Audio/Video)**: 234/547ms（单并发冷启动）

### Table 2: 不同并发下的首包延迟

| 延迟项 | 1 并发 | 4 并发 | 6 并发 |
|--------|--------|--------|--------|
| Thinker-Talker 预处理 | 72/160ms | 94/180ms | 100/200ms |
| Thinker TTFT | 88/160ms | 468/866ms | 673/1330ms |
| Talker TTFT | 57/210ms | 145/450ms | 376/734ms |
| MTP 单 token | 14ms | 16ms | 18ms |
| Codec 单 code | 3ms | 5ms | 5ms |
| **总体延迟 (Audio/Video)** | **234/547ms** | **728/1517ms** | **1172/2284ms** |
| RTF | 0.47 | 0.56 | 0.66 |

**关键发现**: MoE 架构使 Thinker/Talker 的 prefill 和 TTFT 在高并发下保持稳定；RTF 始终 < 1。

### Table 3: 语言支持

| Modality | # Langs | Languages |
|----------|---------|-----------|
| Text | 119 | 见 Qwen3 完整列表 |
| Speech Input | 19 | ar, de, en, es, fr, id, it, ja, ko, ms, nl, pt, ru, th, tr, ur, vi, yue, zh |
| Speech Output | 10 | de, en, es, fr, it, ja, ko, pt, ru, zh |

### 部分关键评测表 (Text→Text)

#### Table 4: Instruct 模型文本性能 (非推理)

| Task | GPT-4o-0327 | Qwen3-235B-A22B | Qwen3-30B-A3B-Ins | **Qwen3-Omni-30B-A3B-Ins** | Qwen3-Omni-Flash-Ins |
|------|-------------|-----------------|--------------------|----------------------------|-----------------------|
| MMLU-Redux | 91.3 | 89.2 | 89.3 | 86.6 | 86.8 |
| GPQA | 66.9 | 62.9 | 70.4 | 69.6 | 69.7 |
| AIME25 | 26.7 | 24.7 | 61.3 | **65.0** | 65.9 |
| ZebraLogic | 52.6 | 37.7 | 90.0 | 76.0 | 76.1 |
| IFEval | 83.9 | 83.2 | 84.7 | 81.0 | 81.7 |

**关键发现**: Qwen3-Omni Instruct 在 AIME25 上**超越 GPT-4o** (65.0 vs 26.7)，且与同规模文本专用模型 Qwen3-30B-A3B 性能可比。

#### Table 5: Thinking 模型文本性能 (推理)

| Task | Gemini-2.5-Flash-Thinking | Qwen3-235B-A22B-Thinking | **Qwen3-Omni-30B-A3B-Thinking** | Qwen3-Omni-Flash-Thinking |
|------|---------------------------|---------------------------|--------------------------------|----------------------------|
| MMLU-Redux | 92.1 | 92.7 | 88.8 | 89.7 |
| GPQA | 82.8 | 71.1 | 73.1 | 73.1 |
| AIME25 | 72.0 | 81.5 | 73.7 | 74.0 |
| WritingBench | 83.9 | 80.3 | **85.5** | **85.9** |

**关键发现**: Thinking 变体在 WritingBench 上超越所有基线，验证多模态训练不会削弱推理能力。

### 部分关键评测表 (Audio→Text)

#### Table 6: ASR 性能 (WER)

| Benchmark | Seed-ASR | GPT-4o-Trans | Gemini-2.5-Pro | Qwen2.5-Omni | **Qwen3-Omni-Ins** |
|-----------|----------|-------------|----------------|-------------|-------------------|
| LibriSpeech clean/other | 1.58/2.84 | 1.39/3.75 | 2.89/3.56 | 1.74/3.45 | **1.22/2.48** |
| Wenetspeech net/meeting | 4.66/5.69 | 15.30/32.27 | 14.43/13.47 | 5.91/7.65 | **4.69/5.89** |
| Fleurs-en/zh | 3.40/2.69 | 10.01/9.84 | 9.89/8.00 | 7.61/5.13 | **6.05/4.31** |
| Multilingual (19 lang) | - | 15.67 | 8.09 | 5.55 | **4.48** |

**关键发现**: 32/36 音频基准上达开源 SOTA，22/36 总体 SOTA。ASR 全面超越 GPT-4o-Transcribe 和 Gemini-2.5-Pro。

#### Table 7: 语音交互与音频推理 (VoiceBench)

| Model | Overall | MMAU | MMSU |
|-------|---------|------|------|
| GPT-4o-Audio | 86.8 | 62.5 | 56.4 |
| Gemini-2.5-Flash | 83.4 | 71.8 | 70.2 |
| Gemini-2.5-Pro | **89.6** | 77.4 | 77.7 |
| Qwen2.5-Omni | 73.6 | 65.5 | 62.6 |
| **Qwen3-Omni-Thinking** | 89.5 | 77.6 | 69.1 |
| **Qwen3-Omni-Instruct** | 88.8 | 75.4 | 70.2 |

**关键发现**: Thinking 变体 VoiceBench 89.5，与 Gemini-2.5-Pro (89.6) 基本持平；MMAU 超越 Gemini-2.5-Pro (77.6 vs 77.4)。

### 部分关键评测表 (Vision→Text, AudioVisual→Text)

#### Table 9: Instruct 视觉性能

| Dataset | GPT-4o | Gemini-2.0-Flash | Qwen2.5-VL-72B | **Qwen3-Omni-Ins** |
|---------|--------|------------------|----------------|-------------------|
| MMMU-Pro overall | 69.1 | 71.3 | 85.2 | **86.4** |
| MathVista mini | 51.9 | 56.1 | 86.8 | **87.1** |
| MATH-Vision full | 63.8 | 71.4 | 70.2 | **71.4** |
| AI2D w.M. | 84.6 | 86.7 | 88.7 | 89.5 |

**关键发现**: 视觉性能与同规模 Qwen3-VL 可比，在 MMMU-Pro 和 MathVista 上超越 GPT-4o 和 Gemini-2.0-Flash。

#### Table 11: 音视频理解 (WorldSense)

| Model | Score |
|-------|-------|
| Previous Open-source SoTA | 47.1 |
| Gemini-2.5-Flash | 50.9 |
| Qwen2.5-Omni | 45.4 |
| **Qwen3-Omni-30B-A3B-Instruct** | **54.0** |
| Qwen3-Omni-Flash-Instruct | 54.1 |

### 部分语音生成评测

#### Table 13: 零样本语音生成 (Seed-TTS, WER content consistency)

| Model | test-zh | test-en |
|-------|---------|---------|
| Seed-TTS RL | 1.00 | 1.94 |
| CosyVoice 3 | 0.71 | 1.45 |
| Qwen2.5-Omni-7B | 1.42 | 2.33 |
| **Qwen3-Omni-30B-A3B** | **1.07** | **1.39** |

#### Table 15: 跨语言语音生成 (WER)

| Language Pair | Qwen3-Omni | CosyVoice3 | CosyVoice2 |
|---------------|------------|------------|------------|
| any-to-en (avg) | ~3.14 | ~3.79 | ~11.6 |
| any-to-ko (avg) | ~5.44 | ~9.40 | ~22.7 |
| any-to-ja (avg) | ~6.69 | ~6.00 | ~11.3 |

**关键发现**: any-to-en 和 any-to-ko 跨语言克隆超越 CosyVoice3，any-to-ja 即使不做文本正则化也达到可比水平。

### Table 16: 无退化验证 (30A3 Base 模型对比)

| Task | Qwen3-30B-A3B-Base | Qwen3-VL-30B-A3B-Base | **Qwen3-Omni-30B-A3B-Base** |
|------|--------------------|------------------------|----------------------------|
| MMLU | 81.24 | - | **81.69** |
| MMLU-Pro | 61.81 | - | **61.57** |
| BBH | 83.79 | - | 83.53 |
| MATH | 60.84 | - | 60.42 |
| EvalPlus | 69.70 | - | **73.96** |
| MMMU val | - | 57.22 | **59.33** |
| MMStar | - | 67.20 | **69.60** |
| OCRBench | - | 85.80 | **86.00** |
| Video-MME | - | 69.22 | **69.25** |
| LVBench | - | 48.61 | **51.07** |

**关键发现**: Omni 模型在文本基准上**不减反增**（MMLU 81.24 -> 81.69），视觉任务全面领先同规模 VL 专用模型，视频理解也有提升。音频数据的加入一致地提升了视觉性能。

---

## 实验

### 预训练数据与阶段

| 阶段 | 数据量 | 序列长度 | 内容 |
|------|--------|----------|------|
| S1: Encoder Alignment | - | 8,192 | 锁定 LLM，分别训练视觉和音频编码器及适配器 |
| S2: General | ~2T tokens | 8,192 | Text 0.57T, Audio 0.77T, Image 0.82T, Video 0.05T, Video-Audio 0.05T |
| S3: Long Context | - | 32,768 | 增加长音频和长视频数据比例 |

### 后训练

**Thinker** (3 阶段):
1. 轻量 SFT: 桥接预训练表示与下游任务
2. Strong-to-Weak Distillation: Off-policy (教师生成) + On-policy (KL 散度对齐教师 logits)
3. GSPO (RL): 规则奖励（数学/代码/指令遵循）+ 模型奖励（Qwen3/Qwen2.5-VL 做 LLM-as-a-Judge）

**Talker** (4 阶段):
1. 大规模语音数据训练，建立多模态表征到语音的单调映射
2. CPT: 高质量数据继续预训练 + 长上下文训练
3. DPO: 多语言语音偏好优化
4. Speaker Fine-tuning: 特定音色适配

**Captioner**: 在 Qwen3-Omni-30B-A3B 基础上用大规模音频描述数据微调

### 评测基准汇总

| 模态方向 | 基准 |
|----------|------|
| Text→Text | MMLU-Redux, GPQA, AIME25, ZebraLogic, MultiPL-E, IFEval, Creative Writing v3, WritingBench, BFCL-v3, MultiIF, PolyMath |
| Audio→Text (ASR) | Wenetspeech, Librispeech, CV15, Fleurs (19 lang), MIR-1K, Opencpop |
| Audio→Text (理解) | MMAU, MMSU, VoiceBench, RUL-MuchoMusic, GTZAN, MTG-Jamendo, MagnaTagATune |
| Vision→Text | MMStar, HallusionBench, MM-MT-Bench, MMMU, MMMU-Pro, MathVista, MATH-Vision, AI2D, ChartQA, CountBench, Video-MME, LVBench, MLVU |
| AudioVisual→Text | WorldSense, DailyOmni, VideoHolmes |
| X→Speech | SEED (test-zh/test-en), MiniMax multilingual (10 lang), CV3-Eval cross-lingual |

---

## 批判性思考

### 优点
1. **系统性证明无退化**: Table 16 的受控对比实验设计严谨——同规模、同算力、同数据，仅差音频/音视频数据，结果证明 Omni 训练在文本和视觉上不减反增
2. **端到端延迟工程实践扎实**: MoE + 分块 prefill + 流式多码本 + 轻量 ConvNet 的组合拳在单并发下达 234ms，且有完整的多并发分析 (Table 2)
3. **语音能力堪比特化模型**: 2000 万小时 AuT 训练 + 多码本表示使 ASR、音乐理解、语音合成均超越或匹配专用模型和 GPT-4o/Gemini
4. **Thinking 和 Captioner 变体**: 扩展了模型的推理和描述能力，填补了通用音频描述模型的空白

### 局限性
1. **长视频理解弱**: 论文明确指出当前模型在 Video-MME/LVBench/MLVU 上性能不足，源于位置外推受限和上下文长度不足
2. **模型权重大、部署成本高**: 30B-A3B + Talker + Code2Wav 总规模约 34B 参数，虽 MoE 降低推理 IO 但硬件门槛仍然较高
3. **Thinking 在感知任务上反直觉**: Table 17/18 显示 Thinking 变体在 ASR 和音乐理解上反而更差，论文归因于"高阶推理可能引入幻觉"，但这种行为缺乏深入分析
4. **语音生成评估维度有限**: 仅用 WER 和 speaker similarity，缺乏对自然度、情感表达、韵律多样性等主观维度的系统评估

### 潜在改进方向
1. 增强位置外推能力和上下文长度（解决长视频理解瓶颈）
2. 探索 Thinking 模型在感知任务上的退化机制，设计自适应推理深度
3. 扩展语音生成的主观质量评估（MOS、AB 测试）
4. 多说话人 ASR、视频 OCR、音视频主动学习、Agent/函数调用增强

### 可复现性评估
- [x] 代码开源 (Apache 2.0)
- [x] 预训练模型 (HuggingFace/ModelScope)
- [x] 训练细节完整 (预训练阶段、后训练流程、数据配比)
- [x] 基准结果全面 (18+ 表格、36 个音频基准)
- [ ] 预训练数据未公开（仅描述构成比例）
- [ ] 训练代码未完整开源

---

## 关联笔记

### 基于
- [[Qwen3]]: Thinker 的 LLM 基础权重来源
- [[Qwen3-VL]]: 视觉编码器来源
- [[Qwen2.5-Omni]]: Thinker-Talker 架构的前身
- [[SigLIP|SigLIP2-So400m]]: 视觉编码器初始化

### 对比
- [[Gemini-2.5-Pro]]: 闭源标杆，音频和全模态性能主要对比对象
- [[GPT-4o]]: 闭源标杆，文本和视觉对比基线
- [[Qwen2.5-VL|Qwen2.5-VL-72B]]: 视觉专用模型对比
- [[CosyVoice 3]]: 语音合成对比基线

### 方法相关
- [[Thinker-Talker Architecture]]: 核心架构
- [[Mixture-of-Experts]]: Thinker 和 Talker 的模型设计
- [[Audio Transformer|AuT]]: 自研音频编码器
- [[Multimodal Rotary Position Embedding|TM-RoPE]]: 多模态位置编码
- [[Multi-Token Prediction|MTP]]: 多码本残差预测
- [[Chunked Prefilling]]: 流式推理加速
- [[Group Sequence Policy Optimization|GSPO]]: 后训练 RL 方法
- [[Strong-to-Weak Distillation]]: 后训练蒸馏策略

### 硬件/数据相关
- vLLM: 推理框架
- 2000 万小时音频数据: AuT 训练规模
- ~2T tokens: 预训练第二阶段数据量

---

## 速查卡片

> [!summary] Qwen3-Omni
> - **核心**: 首个全模态无退化端到端模型，文本/视觉匹配同规模单模态模型
> - **方法**: Thinker-Talker MoE + AuT 编码器 + 多码本流式语音生成 + TM-RoPE
> - **结果**: 32/36 音频基准开源 SOTA，22/36 总体 SOTA，首包延迟 234ms
> - **代码**: https://github.com/QwenLM/Qwen3-Omni

---

*笔记创建时间: 2026-05-18*
