---
title: "Mitigating Multimodal LLMs Hallucinations via Relevance Propagation at Inference Time"
method_name: "LIME"
authors: [Itai Allouche, Joseph Keshet]
year: 2026
venue: arXiv
tags: [multimodal-hallucination, inference-time, relevance-propagation, vision-language, audio-language, hallucination-mitigation, training-free]
zotero_collection: 
image_source: local
arxiv_html: https://arxiv.org/html/2605.01766
created: 2026-05-08
---

# 论文笔记：Mitigating Multimodal LLMs Hallucinations via Relevance Propagation at Inference Time

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Technion -- Israel Institute of Technology |
| 日期 | May 2026 |
| 项目主页 | https://github.com/ItaiAllouche/lime |
| 对比基线 | [[OPERA]], [[VCD]], [[ICD]], [[MemVR]], [[V-ITI]], [[AAD]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.01766) / [Code](https://github.com/ItaiAllouche/lime) |

---

## 一句话总结

> 利用 Layer-wise Relevance Propagation 在推理时优化 KV cache，无需训练即可抑制多模态大模型幻觉。

---

## 核心贡献

1. **首次用 LRP 诊断多模态幻觉成因**: 通过 [[Layer-wise Relevance Propagation|LRP]] 定量证明幻觉源于推理时文本 token 主导、感知模态 token 未被充分利用的模态利用不平衡
2. **提出 LIME 训练无关框架**: 在每一步解码时，基于 relevance 目标对 KV cache 进行可学习扰动，增强模态 token 贡献，无需修改模型参数
3. **跨模态与跨模型泛化**: 在视觉（LLaVA、Qwen-VL）和音频（SALMONN、Qwen2-Audio）两大模态、多种模型上一致降低幻觉，同时保持生成质量

---

## 问题背景

### 要解决的问题
[[多模态幻觉|Multimodal Hallucination]]：多模态大模型（MLLM）在生成回答时，倾向于依赖语言先验而非实际的感知输入（图像/音频），导致生成内容与感知信号不一致。

### 现有方法的局限
- [[VCD|Visual Contrastive Decoding]]、[[ICD|Instruction Contrastive Decoding]] 等方法通过对比不同分布的 logit 输出来减少幻觉，但缺乏对模型内部表征机理的深入理解
- [[OPERA]] 依赖 attention map 的启发式分析，未触及 token 级别的精确归因
- 大多数方法仅针对视觉模态，缺乏对音频等其它模态的覆盖
- 现有方法无法量化"模型到底在多大程度上利用了感知信息"

### 本文的动机
作者提出核心假设：**多模态幻觉的根本原因是推理时文本 token 对输出的贡献远超感知模态 token**。通过 [[Layer-wise Relevance Propagation|LRP]] 可以精确量化每个 token 对输出的贡献度，进而设计优化目标主动增强感知 token 的 relevance，从根源上抑制幻觉。

---

## 方法详解

### 模型架构

LIME 采用 **推理时优化 + 冻结模型** 架构：

- **输入**: 多模态输入 $X$（图像/音频 + 文本提示）+ 已生成的 token 序列 $y_{<t}$
- **Backbone**: 任意冻结的 MLLM（LLaVA / Qwen-VL / Qwen2-Audio 等）
- **核心模块**: [[Layer-wise Relevance Propagation|LRP]] 用于 token 贡献归因 + [[KV Cache]] 可学习扰动 $\Delta = \{\Delta K, \Delta V\}$
- **输出**: 增强模态利用后的下一个 token 分布
- **总参数**: 无额外可训练参数（仅推理时零初始化 $\Delta$）

### 核心模块

#### 模块1: LRP-Based 模态利用分析

**设计动机**: 利用 [[Layer-wise Relevance Propagation|LRP]] 将模型输出逐层分解为各输入 token 的可加性贡献，量化"模型到底看了多少感知信息"。

**具体实现**:
- 使用 [[AttnLRP]]（Achtibat et al., 2024）适配 Transformer 架构，包含针对注意力机制的特殊传播规则
- 定义模态 relevance $\Phi_M = \sum_{i \in M} \Phi_i$ 和文本 relevance $\Phi_T = \sum_{j \in T} \Phi_j$
- 分析发现：即使问题需要视觉/音频信息，文本 token 的 relevance 仍远高于感知 token

#### 模块2: 推理时 KV 优化

**设计动机**: 利用 [[Inference-time Optimization|推理时优化]] 在每一步解码时对 [[KV Cache]] 施加微小扰动，增强感知 token 的 relevance，同时保持输出分布不偏离原始模型太远。

**具体实现**:
- 在每步解码时，向所有 Transformer 层的 key 和 value 张量添加可学习扰动 $\Delta = \{\Delta K^\ell, \Delta V^\ell\}_{\ell=1}^L$
- 扰动在每层内跨注意力头共享，每步解码后重置为零
- 使用 [[Adam]] 优化器进行少量梯度步（7 步）最小化组合损失
- 需要计算通过 LRP 的二阶导数

---

## 关键公式

### 公式1: [[Layer-wise Relevance Propagation|LRP 传播规则]]

$$
\Phi_j^\ell = \sum_i \frac{a_j^\ell W_{ji}^\ell}{\sum_k a_k^\ell W_{ki}^\ell} \cdot \Phi_i^{\ell+1}
$$

**含义**: 将第 $\ell+1$ 层的 relevance $\Phi_i^{\ell+1}$ 按前向传播中激活 $a_j^\ell$ 与权重 $W_{ji}^\ell$ 的贡献比例反向分配到第 $\ell$ 层的神经元。

**符号说明**:
- $\Phi_j^\ell$: 第 $\ell$ 层神经元 $j$ 的 relevance 值
- $a_j^\ell$: 第 $\ell$ 层神经元 $j$ 的激活值
- $W_{ji}^\ell$: 从层 $\ell$ 到层 $\ell+1$ 的连接权重
- $\Phi_i^{\ell+1}$: 第 $\ell+1$ 层神经元 $i$ 的 relevance 值

### 公式2: [[LRP-ε|LRP-ε 稳定规则]]

$$
\Phi_j^\ell = \sum_i \frac{x_j^\ell W_{ji}^\ell}{\sum_k x_k^\ell W_{ki}^\ell + \varepsilon \cdot \operatorname{sign}(\sum_m x_m^\ell W_{mi}^\ell)} \cdot \Phi_i^{\ell+1}
$$

**含义**: 在分母添加 $\varepsilon$ 项防止数值不稳定，$\operatorname{sign}$ 保证符号一致性。

**符号说明**:
- $\varepsilon$: 小的稳定常数
- $x_j^\ell$: 第 $\ell$ 层的输入特征
- $\operatorname{sign}(\cdot)$: 符号函数，保留分母原始符号

### 公式3: [[AttnLRP|注意力-值交互传播]]

$$
\Phi_{ji}^{\ell-1} = \sum_p \frac{A_{ji}^\ell V_{ip}^\ell}{2 O_{jp}^\ell + \varepsilon} \cdot \Phi_{jp}^\ell
$$

**含义**: 将 relevance 通过注意力矩阵 $A$ 和值矩阵 $V$ 的交互传播到前一层，保留注意力机制的成对贡献结构。

**符号说明**:
- $A_{ji}^\ell$: 注意力权重矩阵，query $j$ 对 key $i$ 的注意力
- $V_{ip}^\ell$: 值矩阵
- $O_{jp}^\ell$: 注意力输出
- $\Phi_{jp}^\ell$: 第 $\ell$ 层输出位置 $j$、维度 $p$ 的 relevance

### 公式4: [[LRP Conservation|Relevance 守恒律]]

$$
\Phi^{\ell-1} = \sum_i \Phi_i^{\ell-1} = \sum_j \Phi_j^\ell = \Phi^\ell
$$

**含义**: LRP 保证每层的 relevance 总和守恒，使得跨层和跨 token 的 relevance 可以直接比较。

### 公式5: [[Softmax Relevance Propagation|Softmax Relevance 重分配]]

$$
\Phi_j^{\ell-1} = x_j^\ell \left( \Phi_j^\ell - a_j^\ell \sum_i \Phi_i^\ell \right)
$$

**含义**: 处理 softmax 非线性，将 relevance 从 softmax 输出重分配到输入，减去均值以保持守恒。

**符号说明**:
- $x_j^\ell$: softmax 输入
- $a_j^\ell$: softmax 输出（注意力权重）
- $\Phi_j^\ell$: softmax 输出端的 relevance

### 公式6: [[Modality Relevance|模态 Relevance 目标]]

$$
\mathcal{L}_{\text{rel}}(\Delta) = -\frac{1}{|M|} \sum_{i \in M} \log \frac{\exp(\Phi_{i,\Delta} / \tau)}{\sum_{k \in M \cup T} \exp(\Phi_{k,\Delta} / \tau)}
$$

**含义**: 对比损失，将感知模态 token 视为正样本，最大化其相对所有 token（模态+文本）的 relevance 比例。温度 $\tau$ 控制 softmax 锐度。

**符号说明**:
- $M$: 感知模态 token 集合（图像 patch / 音频帧）
- $T$: 文本 token 集合
- $\Phi_{i,\Delta}$: 施加扰动 $\Delta$ 后 token $i$ 的 relevance
- $\tau$: 温度参数（所有模型设为 0.1）
- $\Delta$: KV cache 可学习扰动

### 公式7: [[KL Divergence Regularization|KL 正则项]]

$$
\mathcal{L}_{\text{KL}}(\Delta) = D_{\text{KL}}\left( p_{\theta,\Delta}(y_t \mid X, y_{<t}) \;\|\; p_\theta(y_t \mid X, y_{<t}) \right)
$$

**含义**: 约束扰动后的输出分布不偏离原始模型太远，避免生成质量下降。

**符号说明**:
- $p_{\theta,\Delta}$: 施加扰动后的 token 分布
- $p_\theta$: 原始冻结模型的 token 分布
- $D_{\text{KL}}$: KL 散度

### 公式8: 组合损失

$$
\operatorname{argmin}_\Delta \; \mathcal{L}_{\text{rel}}(\Delta) + \lambda \mathcal{L}_{\text{KL}}(\Delta)
$$

**含义**: 在增强模态利用和保持输出一致性之间取得平衡，$\lambda$ 控制正则强度。

**符号说明**:
- $\lambda$: KL 正则权重（多数模型设为 0.1，Qwen2-Audio 为 $7 \times 10^{-3}$）

### 公式9: [[CHAIR|CHAIR 评估指标]]

$$
\text{CHAIR}_I = \frac{|\{\text{hallucinated objects}\}|}{|\{\text{all objects mentioned}\}|}
$$

$$
\text{CHAIR}_S = \frac{|\{\text{sentences with hallucinated object}\}|}{|\{\text{all sentences}\}|}
$$

**含义**: CHAIR 从对象级别（$\text{CHAIR}_I$）和句子级别（$\text{CHAIR}_S$）衡量描述生成中的幻觉比例，越低越好。

---

## 关键图表

### Figure 1: 幻觉示例与 LIME 缓解效果

![[assets/LIME_fig1.png]]

**说明**: (a,d) 输入图像和问题；(b,e) 标准解码下模型预测错误且 [[Layer-wise Relevance Propagation|relevance]] 热力图显示模型注意力分散；(c,f) LIME 纠正预测且 relevance 集中到正确图像区域。直观展示了 LIME 通过重分配 token relevance 来纠正幻觉的机理。

### Figure 2: LIME 方法概览

![[assets/LIME_fig2.png]]

**说明**: LIME 整体框架。冻结的 MLLM 在推理时通过可学习的 [[KV Cache]] 更新（$\Delta$KV）进行优化，损失函数由 [[Modality Relevance|relevance 目标]] $\mathcal{L}_{\text{rel}}$ 和 [[KL Divergence Regularization|KL 正则]] $\mathcal{L}_{\text{KL}}$ 组成，迭代优化使模型逐步增强对感知模态的依赖。

### Figure 3: 视觉 Relevance 在推理时的演化

![[assets/LIME_fig3.png]]

**说明**: 随 LIME 优化步数增加，relevance 从初始的弥散分布逐步集中到与问题相关的正确图像区域。定量验证了优化过程有效地将模型的注意力从语言先验转移到视觉证据上。

### Figure 4: 音频域定性示例

![[assets/LIME_fig4.png]]

**说明**: 音频波形叠加 relevance 分数（绿色），虚线标记真实事件区域。标准解码下 relevance 弥散且与正确时间段对齐差；LIME 将 relevance 集中到正确的时间区域。展示了方法在 [[Audio-Language Model|音频-语言模型]] 上的泛化能力。

### Figure 5: KL 正则系数消融

![[assets/LIME_fig5.png]]

**说明**: 在 LLaVA-1.5-7B（左）和 Qwen2-Audio-7B-Instruct（右）上对不同 $\lambda$ 值和编辑策略（仅 K、仅 V、KV 联合）的消融。结果表明：(1) KV 联合编辑最优；(2) 适中的 $\lambda$ 在增强模态利用和保持输出质量之间取得最佳平衡。

### Figure 6: LLaVA-1.5-7B 幻觉减少定性展示

![[assets/LIME_fig6.png]]

**说明**: 红色斜体标记幻觉预测。LIME 在 LLaVA-1.5-7B 上有效纠正了多个典型幻觉场景，包括对象存在性判断错误和属性描述错误。

### Figure 7: Qwen-VL-Chat 幻觉减少定性展示

![[assets/LIME_fig7.png]]

**说明**: 同上风格，展示 LIME 在 Qwen-VL-Chat 上的幻觉纠正效果，证明方法跨模型泛化能力。

### Figure 8: Qwen2.5-VL-7B-Instruct 幻觉减少定性展示

![[assets/LIME_fig8.png]]

**说明**: 同上风格，展示 LIME 在更先进的 Qwen2.5-VL-7B-Instruct 上仍能进一步降低幻觉，证明方法对强基线的增益。

### Table 1: POPE Benchmark（LLaVA-1.5-7B on MSCOCO）

| Method            | Random Acc | Random F1 | Popular Acc | Popular F1 | Adversarial Acc | Adversarial F1 | Avg Acc   | Avg F1    |
| ----------------- | ---------- | --------- | ----------- | ---------- | --------------- | -------------- | --------- | --------- |
| LLaVA-1.5-7B      | 83.49      | 82.28     | 79.98       | 79.34      | 76.03           | 76.26          | 79.83     | 79.29     |
| + OPERA           | 87.53      | 86.45     | 84.21       | 83.50      | 80.88           | 80.69          | 84.21     | 83.55     |
| + VCD             | 86.84      | 86.83     | 82.65       | 83.37      | 77.31           | 79.28          | 82.27     | 83.16     |
| + ICD             | 84.87      | 83.27     | 82.93       | 81.45      | 81.07           | 79.96          | 82.96     | 81.56     |
| + MemVR           | 88.50      | 87.34     | 87.10       | 86.01      | 85.20           | 84.28          | 86.93     | 85.88     |
| + V-ITI           | 89.74      | 87.72     | 84.96       | 84.77      | 86.31           | 82.44          | 87.00     | 84.98     |
| **+ LIME (ours)** | **90.27**  | **89.75** | **87.91**   | **87.85**  | 85.51           | **84.52**      | **87.89** | **87.37** |

**说明**: LIME 在 [[POPE]] 的 MSCOCO 变体上取得最优平均 Accuracy 和 F1，在 Random 和 Popular 设置上尤其突出。相比原始 LLaVA-1.5-7B，Avg Acc 提升 **+8.06 个百分点**。

### Table 2: CHAIR 评估

| Method | CHAIR_S ↓ | CHAIR_I ↓ | Average ↓ | Recall ↑ |
|--------|----------|----------|----------|---------|
| LLaVA-1.5-7B | 52.0 | 15.8 | 32.7 | 75.2 |
| + OPERA | 47.8 | 14.6 | 31.8 | 77.3 |
| + VCD | 48.6 | 14.9 | 31.2 | 76.8 |
| + ICD | 56.2 | 16.3 | 36.3 | 16.3 |
| + MemVR | 46.6 | 13.0 | 29.8 | 80.8 |
| + V-ITI | 46.1 | 13.5 | 29.8 | 80.4 |
| **+ LIME (ours)** | **42.7** | 13.0 | **27.85** | 72.0 |
| Qwen-VL-Chat | 46.0 | 12.5 | 29.3 | 64.3 |
| + VCD | 46.8 | 12.3 | 29.6 | 67.9 |
| + ICD | 45.0 | 14.3 | 29.7 | 47.6 |
| + V-ITI | 44.2 | 12.5 | 28.4 | 66.4 |
| **+ LIME (ours)** | 44.5 | **12.0** | 28.25 | 68.7 |
| Qwen2.5-VL-7B-Instruct | 25.6 | 9.1 | 17.35 | 55.1 |
| **+ LIME (ours)** | **21.2** | **8.2** | **14.7** | 56.5 |

**说明**: LIME 在三款视觉模型上均降低 [[CHAIR]] 指标。在 LLaVA-1.5-7B 上 CHAIR_S 从 52.0 降至 42.7（降幅 **17.9%**），在 Qwen2.5-VL-7B 上更进一步从 25.6 降至 21.2。值得注意的是，LIME 在 Qwen2.5-VL 这种强基线上仍能带来显著改善。

### Table 3: 音频 Benchmark

**Audio Hallucination QA:**

| Method | Random Acc | Random F1 | Popular Acc | Popular F1 | Adversarial Acc | Adversarial F1 |
|--------|-----------|-----------|-------------|-------------|-----------------|-----------------|
| SALMONN-7B | 53.91 | 23.37 | 49.32 | 18.27 | 50.31 | 20.01 |
| + AAD | 57.22 | 36.74 | 48.71 | 18.78 | 48.04 | 17.42 |
| **+ LIME** | 56.88 | 36.85 | **53.12** | **25.76** | **54.35** | **26.00** |
| Qwen2-Audio-7B | 56.19 | 26.78 | 51.34 | 20.50 | 50.13 | 20.24 |
| + AAD | 59.50 | 31.62 | 51.31 | 12.09 | 51.29 | 11.43 |
| **+ LIME** | **63.36** | **50.27** | **57.53** | **46.43** | 53.10 | 37.08 |

**[[AIR-Bench]] (Speech / Sound / Music):**

| Method | Speech | Sound | Music |
|--------|--------|-------|-------|
| SALMONN-7B | 37.51 | 33.58 | 31.05 |
| + AAD | 42.62 | 34.56 | 30.38 |
| **+ LIME** | **45.20** | **36.90** | 31.95 |
| Qwen2-Audio-7B | 57.56 | 60.86 | 55.89 |
| + AAD | 60.00 | 61.90 | 57.43 |
| **+ LIME** | **66.10** | **66.41** | 56.25 |

**说明**: 在音频域，LIME 同样一致优于基线方法 [[AAD]]。Qwen2-Audio-7B 上 Random F1 从 26.78 跃升至 50.27（提升 **87.7%**），说明 LIME 在音频幻觉问题上尤为有效。AIR-Bench 上 Speech 和 Sound 子任务均有显著增益。

### Table 4: Relevance-Based 模态利用分析

| Metric | Decoding | LLaVA-1.5-7B | Qwen-VL-Chat | Qwen2.5-VL-7B | SALMONN-7B | Qwen2-Audio-7B |
|--------|----------|-------------|-------------|--------------|-----------|---------------|
| Spatial Grounding ↑ | Vanilla | 0.27 | 0.13 | 0.12 | 0.19 | 0.31 |
| | **LIME** | **0.36** | **0.20** | **0.21** | **0.28** | **0.57** |
| Modality Reliance ↑ | Vanilla | 0.10 | 0.41 | 0.43 | 0.10 | 0.34 |
| | **LIME** | **0.17** | **0.53** | **0.52** | **0.19** | **0.42** |

**说明**: 定量验证了 LIME 的核心假设——所有模型的 [[Spatial Grounding]] 和 [[Modality Reliance|模态依赖度]] 在使用 LIME 后均提升。尤其 Qwen2-Audio 的 Spatial Grounding 从 0.31 跃升至 0.57（提升 **83.9%**），直观证明了方法有效增强了模型对感知输入的利用。

### Table 5: 超参数配置（Appendix A.1）

| Model | Optimization Steps | Learning Rate | KL Weight ($\lambda$) | Temperature ($\tau$) |
|-------|-------------------|---------------|----------------------|---------------------|
| LLaVA-1.5-7B | 7 | $3 \times 10^{-4}$ | 0.1 | 0.1 |
| Qwen-VL-Chat | 7 | $4 \times 10^{-4}$ | 0.1 | 0.1 |
| Qwen2.5-VL-7B-Instruct | 7 | $3 \times 10^{-4}$ | 0.1 | 0.1 |
| SALMONN-7B | 7 | $3 \times 10^{-4}$ | 0.1 | 0.1 |
| Qwen2-Audio-7B-Instruct | 7 | $5 \times 10^{-4}$ | $7 \times 10^{-3}$ | 0.1 |

**说明**: 所有模型统一使用 7 步优化。Qwen2-Audio 使用更低的 KL 正则权重（$7 \times 10^{-3}$ vs 0.1），可能是因为其原始模型的 [[Modality Reliance|模态依赖]] 已经较高（0.34），过度约束反而限制优化空间。

### Table 6: 计算开销（Appendix A.2）

| Method | Tokens/sec ↑ | Slowdown ↓ | Peak Memory (GB) ↓ |
|--------|-------------|-----------|-------------------|
| LLaVA-1.5-7B | 3.02 | 1.0× | 26.24 |
| + LIME | 0.32 | 9.43× | 34.75 |
| Qwen2-Audio-7B-Instruct | 2.76 | 1.0× | 30.84 |
| + LIME | 0.30 | 9.2× | 33.84 |

**说明**: LIME 带来约 **9-10 倍推理减速**（~3 tok/s → ~0.3 tok/s），显存开销约 **8 GB**。这是方法的主要局限性，源于每步解码需进行 7 步梯度优化和 LRP 计算。

### Table 7: POPE on Qwen-VL-Chat 和 Qwen2.5-VL-7B（Appendix A.3）

**MSCOCO:**

| Method | Avg Acc | Avg F1 |
|--------|---------|--------|
| Qwen-VL-Chat | 83.70 | 81.70 |
| + VCD | 86.67 | 86.03 |
| **+ LIME** | **87.25** | **87.03** |
| Qwen2.5-VL-7B | 85.34 | 83.25 |
| **+ LIME** | **86.40** | **84.76** |

**A-OKVQA:**

| Method | Avg Acc | Avg F1 |
|--------|---------|--------|
| Qwen-VL-Chat | 83.93 | 83.24 |
| + VCD | 86.11 | 86.40 |
| **+ LIME** | **86.32** | **86.17** |
| Qwen2.5-VL-7B | 86.56 | 85.48 |
| **+ LIME** | **87.80** | **87.08** |

### Table 8: POPE LLaVA-1.5-7B on A-OKVQA（Appendix A.4）

| Method | Avg Acc | Avg F1 |
|--------|---------|--------|
| LLaVA-1.5-7B | 79.13 | 79.10 |
| + OPERA | 84.27 | 84.08 |
| + VCD | 80.99 | 82.30 |
| + ICD | 81.64 | 82.00 |
| + MemVR | 86.21 | 86.64 |
| + V-ITI | 86.44 | 86.37 |
| **+ LIME** | **87.33** | **87.15** |

**说明**: 在 A-OKVQA 上 LIME 同样取得最优，验证了方法在不同数据集上的泛化性。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[POPE]] (MSCOCO) | 3000 样本 | 三种负采样策略（random/popular/adversarial） | 视觉幻觉评测 |
| [[POPE]] (A-OKVQA) | 3000 样本 | 同上 | 视觉幻觉评测 |
| [[CHAIR]] (MSCOCO Caption) | 500 张图像 | 对象级+句子级幻觉评估 | 描述幻觉评测 |
| Audio Hallucination QA | 3000 样本 | 三策略负采样，声音事件二分类 | 音频幻觉评测 |
| [[AIR-Bench]] | 19k+ 单选题 | Speech/Sound/Music 三域 | 音频理解评测 |
| DCASE 2019 Task 4 | 100 样本 | 声音事件检测 | 音频 relevance 分析 |

### 实现细节

- **Backbone**: LLaVA-1.5-7B (CLIP + LLaMA)、Qwen-VL-Chat (CLIP + Qwen)、Qwen2.5-VL-7B-Instruct、SALMONN-7B (Whisper large v2 + Vicuna)、Qwen2-Audio-7B-Instruct (Whisper large v3 + Qwen)
- **推理时优化器**: [[Adam]]
- **优化步数**: 7 步/token（所有模型统一）
- **学习率**: $3\times 10^{-4} \sim 5\times 10^{-4}$
- **KV 扰动**: 零初始化，每步解码后重置
- **LRP 框架**: [[AttnLRP]] (Achtibat et al., 2024)
- **硬件**: 8 NVIDIA A100 GPUs

### 可视化结果

- Figure 3 清晰展示了 relevance 从弥散到集中的演化过程
- Figure 4 验证了方法在音频时序定位上的有效性
- Figure 6-8 提供了大量跨模型、跨场景的幻觉纠正案例

---

## 批判性思考

### 优点
1. **训练无关**：不需要任何微调、额外数据或模型修改，即插即用到任意 MLLM
2. **理论根基扎实**：基于 [[Layer-wise Relevance Propagation|LRP]] 提供了可解释的 token 级贡献量化，方法论上有充分的理论支撑
3. **跨模态泛化**：统一框架同时适用于视觉和音频，这一点远超仅针对视觉的现有方法
4. **实验全面**：覆盖 5 种模型、2 种模态、6 个 benchmark，消融充分

### 局限性
1. **推理速度慢 9-10 倍**：每步解码需 7 步梯度优化+LRP 计算，吞吐量从 ~3 tok/s 降至 ~0.3 tok/s，无法用于实时或延迟敏感场景
2. **超参数依赖**：不同模型需要不同的学习率和 $\lambda$（尤其是 Qwen2-Audio 的 $\lambda=7\times10^{-3}$ 远小于其他模型的 0.1），实际部署需额外调参
3. **仅测 7B 规模**：未验证在更大模型（13B/34B/72B）上的效果，无法判断 scaling 特性
4. **二阶导数计算**：优化需要计算通过 LRP 的梯度，增加了工程实现的复杂度

### 潜在改进方向
1. **减少优化步数**：探索能否用更少的步数（如 3 步）或更高效的优化器减少推理开销
2. **预测性扰动**：学习一个轻量网络直接预测 $\Delta$KV，避免每步迭代优化
3. **更大模型验证**：在 13B+ 模型上测试 scaling 效果
4. **扩展到更多模态**：视频、3D 点云等时序/空间模态的幻觉问题

### 可复现性评估
- [x] 代码开源（https://github.com/ItaiAllouche/lime）
- [ ] 预训练模型（不适用，方法无需训练）
- [x] 训练细节完整（超参数在 Appendix A.1 完整列出）
- [x] 数据集可获取（POPE、CHAIR、AIR-Bench 均为公开 benchmark）

---

## 关联笔记

### 基于
- [[Layer-wise Relevance Propagation|LRP]] (Bach et al., 2015): 核心归因技术
- [[AttnLRP]] (Achtibat et al., 2024): Transformer 适配的 LRP 实现

### 对比
- [[VCD|Visual Contrastive Decoding]]: 对比 logit 分布的幻觉抑制方法
- [[ICD|Instruction Contrastive Decoding]]: 指令对比解码
- [[OPERA]]: 基于 attention map 的幻觉缓解
- [[MemVR]]: memory-based 视觉 rectification
- [[V-ITI]]: inference-time intervention 方法
- [[AAD|Audio-Aware Decoding]]: 唯一的训练无关音频幻觉抑制 baseline

### 方法相关
- [[Layer-wise Relevance Propagation|LRP]]: 核心归因方法
- [[KV Cache]]: 优化目标
- [[Inference-time Optimization|推理时优化]]: 方法范式
- [[Modality Relevance|模态 Relevance]]: 核心优化目标

### 硬件/数据相关
- [[POPE]]: 主要评测 benchmark
- [[CHAIR]]: 描述幻觉评测指标
- [[AIR-Bench]]: 音频理解评测
- [[COCO|MSCOCO]]: 视觉评测数据集

---

## 速查卡片

> [!summary] LIME: Learning Inference-time Modality Enhancement
> - **核心**: 训练无关推理时框架，用 LRP 归因 + KV 优化抑制多模态幻觉
> - **方法**: 每步解码对 KV cache 做 7 步 Adam 优化，最大化感知 token relevance 同时 KL 约束保持在原分布附近
> - **结果**: POPE +8.06 Acc、CHAIR_S -17.9%、Audio Random F1 +87.7%，5 个模型一致改善
> - **代价**: 推理减速 ~9-10 倍，显存 +8 GB
> - **代码**: https://github.com/ItaiAllouche/lime

---

*笔记创建时间: 2026-05-08*
