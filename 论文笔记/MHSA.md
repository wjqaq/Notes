---
title: "MHSA: A Lightweight Framework for Mitigating Hallucinations via Steered Attention in LVLMs"
method_name: "MHSA"
authors: [Wei Ding, Yilin Li, Yudong Zhang, Ruobing Xie, Xingwu Sun, Jiansheng Chen, Yu Wang]
year: 2026
venue: arXiv
tags: [hallucination-mitigation, cross-modal-attention, vision-language-model, lightweight-framework, attention-steering, adversarial-training, token-level-detection, vqa]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2605.14966v1
created: 2026-05-18
---

# 论文笔记：MHSA: A Lightweight Framework for Mitigating Hallucinations via Steered Attention in LVLMs

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University, Tencent, University of Science and Technology Beijing, University of Macau |
| 日期 | May 2026 |
| 项目主页 | (未提供) |
| 对比基线 | [[DHCP]] (检测), [[OPERA]], [[PAI]], [[VCD]], [[ICD]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.14966) / [HTML](https://arxiv.org/html/2605.14966v1) |

---

## 一句话总结

> 通过可学习的样本自适应[[Cross-Modal Attention|跨模态注意力]]修正，实现无需修改 LVLM 参数的轻量级[[多模态幻觉|幻觉]]抑制。

---

## 核心贡献

1. **首个样本自适应注意力修正框架**: 首次从可学习的数据驱动角度而非手工启发式规则来修正跨模态注意力以抑制幻觉
2. **三目标联合训练**: 基于 detector-guided loss + 正则化 loss + LVLM 输出质量 loss 稳定训练轻量 MLP 生成器
3. **扩展到 Token 级别**: 将 sentence-level 检测升级为 token-level，使 MHSA 从判别式 VQA 扩展到生成式图像描述任务
4. **跨模型、跨数据集泛化**: 在 Qwen2.5-VL、InternVL2、LLaVA 上一致有效，且表现出跨数据集 OOD 泛化能力

---

## 问题背景

### 要解决的问题
[[LVLM|大型视觉语言模型]] (LVLMs) 在多模态任务中表现优异，但存在严重的[[多模态幻觉|幻觉问题]]——生成与视觉输入不一致的内容。幻觉分为三类：[[Object Hallucination|对象幻觉]]（错误识别物体）、[[Attribute Hallucination|属性幻觉]]（错误识别状态/数量/动作）、[[Relational Hallucination|关系幻觉]]（错误描述空间/上下文关系）。

### 现有方法的局限

| 方法类别 | 代表工作 | 问题 |
|----------|---------|------|
| 训练式 | [[RLHF]]、[[Instruction Tuning]] | 计算成本高、需大量标注数据、与训练分布强绑定 |
| 推理时对比解码 | [[VCD]]、[[ICD]]、[[HALC]] | 推理成本翻倍（需多次 forward）、依赖失真假设 |
| 注意力操纵 | [[OPERA]]、[[PAI]] | 固定启发式规则、不随样本自适应调整 |

### 本文的动机
[[DHCP]] 已证明可通过[[Cross-Modal Attention|跨模态注意力模式]]检测幻觉，但不能抑制。如果注意力模式可以检测幻觉，是否也能通过学习修正这些模式来抑制幻觉？

---

## 方法详解

### 模型架构

MHSA 采用**两阶段检测-修正**架构，基于 [[DHCP]] 检测器，不修改任何 LVLM 参数：

- **输入**: 原始 [[Cross-Modal Attention|跨模态注意力]] $\mathbf{A} \in \mathbb{R}^{L \times H \times N}$（从 LVLM 前向传播中提取）
- **模块1 — [[Hallucination Detector|判别器]] $D$**: 预训练的 DHCP 二层 [[MLP]]（hidden=128, output=2），判断注意力模式是否表现为幻觉
- **模块2 — [[Attention Steering Generator|生成器]] $G$**: 三层 [[MLP]]（hidden=512），学习修正 $\Delta\mathbf{A} = G(\mathbf{A})$
- **修正机制**: $\mathbf{A}' = \mathbf{A} + \Delta\mathbf{A}$（[[Residual Learning|残差学习]]）
- **总参数**: 仅 G 和 D 参与训练，LVLM 完全冻结

### 核心模块

#### 模块1: [[Hallucination Detector|判别器]] $D$

**设计动机**: 利用 [[DHCP]] 的跨模态注意力检测能力，将其作为 token 级监督信号

**具体实现**:
- 基于预训练的 DHCP 二层 [[MLP]]（hidden dim=128，output dim=2: non-hallucinatory / hallucinatory）
- 输入为 flatten 后的跨模态注意力 $\mathbf{A}$
- 在 MHSA 训练中微调学习率极小（保持稳定监督信号）
- 同时用原始注意力及其标签训练：$\mathcal{L}_d = -[y\log D_1(\mathbf{A}) + (1-y)\log D_0(\mathbf{A})]$

#### 模块2: [[Attention Steering Generator|生成器]] $G$

**设计动机**: [[Adversarial Training|对抗训练]]范式，让 G 学习将幻觉注意力模式修正为非幻觉模式

**具体实现**:
- 三层 [[MLP]]：hidden dim=512，激活函数 [[ReLU]]
- 权重初始化为 $\mathcal{U}(-10^{-5}, 10^{-5})$，确保初始 $\Delta\mathbf{A} \approx 0$
- [[Residual Learning|残差设计]]：只学习偏移量而非从头预测，保持预训练注意力结构
- 输入维度 $d = L \cdot H \cdot N$ 因模型而异（Qwen2.5-VL: 112,896; InternVL2-8B: 262,144; LLaVA-v1.5-7B: 589,824）

### 推理流程

1. LVLM 前向传播，提取第一个输出 token 对视觉 token 的[[Cross-Modal Attention|跨模态注意力]] $\mathbf{A}$
2. $D(\mathbf{A})$ 判断是否为幻觉
3. 若检测到幻觉：$\Delta\mathbf{A} = G(\mathbf{A})$，$\mathbf{A}' = \mathbf{A} + \Delta\mathbf{A}$
4. 替换 LVLM 中的原始跨模态注意力，重新生成 token
5. **87.7% 的样本无需修正**，仅在检测为幻觉时触发（12.3%），amortized overhead 仅 $+0.43\times$

---

## 关键公式

### 公式1: [[Cross-Modal Attention|跨模态注意力定义]]（判别式任务）

$$
\mathbf{A}^{(l,h)}_{n} = \mathbf{A}^{(l,h)}_{q_1 \to n}, \quad l=1,\ldots,L,\; h=1,\ldots,H,\; n=1,\ldots,N
$$

**含义**: 从第一个输出 token 位置 $q_1$ 到第 $n$ 个视觉 token 在 layer $l$、head $h$ 处的注意力权重

**符号说明**:
- $L$: LLM 层数
- $H$: 每层注意力头数
- $N$: 视觉 token 数量（统一 resize 后固定，Qwen2.5-VL: 336x336 → N=144）
- $\mathbf{A} \in \mathbb{R}^{L \times H \times N}$: 跨模态注意力张量

### 公式2: [[Cross-Modal Attention|跨模态注意力定义]]（生成式任务，sentence-level）

$$
\mathbf{A}^{(l,h)}_{n} = \frac{1}{M}\sum_{i=1}^{M}\mathbf{A}^{(l,h)}_{q_i \to n}, \quad l=1,\ldots,L,\; h=1,\ldots,H,\; n=1,\ldots,N
$$

**含义**: 对所有输出 token 的跨模态注意力取平均（仅能 sentence-level 检测，稀释了幻觉 token 信号）

### 公式3: [[DHCP]] 检测器

$$
D(\mathbf{A}) = D_{l_2}\left(D_{l_1}\left(\text{flatten}(\mathbf{A})\right)\right) \in \mathbb{R}^{2}
$$

**含义**: 二层 [[MLP]] 将扁平化的注意力映射为二分类概率（Class 0 = non-hallucinatory, Class 1 = hallucinatory）

### 公式4: [[Cross-Modal Attention Correction|注意力修正]]

$$
\mathbf{A}' = \mathbf{A} + \Delta\mathbf{A}
$$

**含义**: 残差形式的注意力修正，保留原始注意力结构的同时进行局部修正

### 公式5: [[Attention Steering Generator|生成器]] $G$

$$
\Delta\mathbf{A} = G(\mathbf{A})
$$

**含义**: 三层 MLP 从原始注意力映射到修正量，输入输出维度相同

### 公式6: [[Detector-Guided Loss|检测器引导损失]] $\mathcal{L}_{\text{dg}}$

$$
\mathcal{L}_{\text{dg}} = -\log D_{0}(\mathbf{A}') = -\log D_{0}(\mathbf{A} + G(\mathbf{A}))
$$

**含义**: [[Adversarial Training|对抗训练]]风格损失——鼓励 $G$ 产生使 $D$ 判定为非幻觉的注意力修正

**符号说明**:
- $D_0(\cdot)$: 判别器输出中 Class 0（非幻觉）的概率

### 公式7: [[Regularization Loss|正则化损失]] $\mathcal{L}_{\text{reg}}$

$$
\mathcal{L}_{\text{reg}} = \|\Delta\mathbf{A}\|_{2}^{2} = \|G(\mathbf{A})\|_{2}^{2}
$$

**含义**: L2 正则化约束修正量大小使其尽可能小，避免破坏原始注意力语义

### 公式8: [[LVLM Output Quality Loss|LVLM 输出质量损失]] $\mathcal{L}_{\text{LVLM}}$

$$
\mathcal{L}_{\text{LVLM}} = \text{CE}(f_{\text{LVLM}}(\mathbf{A}'), y_{\text{gt}})
$$

**含义**: 用修正后注意力生成的输出与 ground-truth 的[[Cross-Entropy Loss|交叉熵]]，确保修正后的注意力维持或提升输出质量

### 公式9: [[MHSA Training Objective|总训练目标]]

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{dg}} \cdot \mathcal{L}_{\text{dg}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}} + \lambda_{\text{LVLM}} \cdot \mathcal{L}_{\text{LVLM}}
$$

**含义**: 三目标加权联合优化，对应方法设计的三个 desiderata

**符号说明**:
- $\lambda_{\text{dg}}$: 检测器引导损失权重（默认 0.01）
- $\lambda_{\text{reg}}$: 正则化损失权重（模型依赖性：1e-4 到 5e-4）
- $\lambda_{\text{LVLM}}$: LVLM 质量损失权重（默认 1.0）

### 公式10: [[DHCP]] 判别器训练损失

$$
\mathcal{L}_{\text{d}} = -\left[y\log D_{1}(\mathbf{A}) + (1-y)\log D_{0}(\mathbf{A})\right]
$$

**含义**: 用原始注意力和标签训练判别器的标准二元交叉熵

**符号说明**:
- $y \in \{0,1\}$: 幻觉标签（$y=1$ 为幻觉，$y=0$ 为非幻觉）

### 公式11: [[Token-Level Hallucination Detection|Token 级跨模态注意力]]

$$
\mathbf{A}_{m} = \mathbf{A}^{(l,h)}_{q_m \to n}, \quad l=1,\ldots,L,\; h=1,\ldots,H,\; n=1,\ldots,N
$$

**含义**: 将 sentence-level 升级为 token-level：为第 $m$ 个生成的 token 单独提取注意力张量

### 公式12: [[Token-Level Hallucination Detection|Token 级注意力修正]]

$$
\Delta\mathbf{A}_{m} = G(\mathbf{A}_{m})
$$

**含义**: 在生成式任务中逐 token 进行注意力修正

---

## 关键图表

### Figure 1: MHSA 机制示意图

![Figure 1](https://arxiv.org/html/2605.14966v1/x1.png)

**说明**: LVLM 首先生成幻觉响应；DHCP 判别器提供监督信号指导 MHSA 修正器修复[[Cross-Modal Attention|跨模态注意力]]，从而实现幻觉抑制。展示了"检测-修正-再生"三阶段流程。

### Figure 2: MHSA 方法流水线

![Figure 2](https://arxiv.org/html/2605.14966v1/x2.png)

**说明**: 完整的 MHSA 训练与推理框架。判别器 $D$ 检测幻觉，生成器 $G$ 修正[[Cross-Modal Attention|跨模态注意力]]，形成两阶段"幻觉检测-抑制"框架。三 loss（$\mathcal{L}_{\text{dg}}$、$\mathcal{L}_{\text{reg}}$、$\mathcal{L}_{\text{LVLM}}$）联合优化。

### Figure 3: 注意力修正的统计分析

![Figure 3](https://arxiv.org/html/2605.14966v1/x3.png)

**说明**: 修正前（幻觉）与修正后（事实）状态的注意力对比统计。(A) 逐层修正强度——集中在中间层；(B) 修正前后空间注意力熵——修正后熵降低，注意力更集中；(C) 修正前后逐层余弦相似度——中间层相似度降低最大；(D) 逐 head 修正幅度热力图——修正是稀疏的，仅集中在少数 layer-head pair。

关键发现：MHSA 是一种**针对性修正机制**，非均匀扰动所有注意力，而是集中在幻觉相关跨模态对齐失配最严重的层和头。

### Figure 4: POPE 判别式任务注意力可视化

![Figure 4](https://arxiv.org/html/2605.14966v1/x4.png)

**说明**: 修正前注意力分散且部分错位 → 幻觉预测；修正后注意力转移到被查询物体本身 → 视觉更 grounded 的判断。

### Figure 5: COCO 生成式任务注意力可视化

![Figure 5](https://arxiv.org/html/2605.14966v1/x5.png)

**说明**: 修正前生成的 token 注意力分配给不相关区域 → 幻觉描述；修正后注意力重定向到视觉相关区域 → 更忠实的描述。

### Figure 6: 采样策略对比

![Figure 6](https://arxiv.org/html/2605.14966v1/x6.png)

**说明**: POPE-COCO 上 class-balanced vs. oversampling 的指标分布。Oversampling 显著提高 Recall 和 F1，确认其有效性。

### Figure 7: 生成器学习率 (lr_G) 灵敏度

![Figure 7](https://arxiv.org/html/2605.14966v1/x7.png)

**说明**: lr_G=1e-4 达到最佳 F1 和 Accuracy。太小 (1e-5) 修正不足；太大 (1e-3) 方差增大。

### Figure 8: 判别器学习率 (lr_D) 灵敏度

![Figure 8](https://arxiv.org/html/2605.14966v1/x8.png)

**说明**: 较小的 lr_D (1e-5 到 1e-6) 产生更好的 Recall 和 F1，符合保持判别器接近冻结为稳定监督信号的设计原则。

### Figure 9: 学习率比例 (lr_G/lr_D) 灵敏度

![Figure 9](https://arxiv.org/html/2605.14966v1/x9.png)

**说明**: 比例对 F1 影响有限，MHSA 对相对学习率尺度具有鲁棒性。

### Figure 10: λ_dg 灵敏度

![Figure 10](https://arxiv.org/html/2605.14966v1/x10.png)

**说明**: λ_dg=0.01 提供最佳 Accuracy。去除 (λ_dg=0) 显著降低性能，确认检测器引导损失的必要性。

### Figure 11: λ_reg 灵敏度

![Figure 11](https://arxiv.org/html/2605.14966v1/x11.png)

**说明**: 适中正则化 (1e-4 到 1e-3) 实现最佳 F1。无正则化时方差增大；过强 (1e-2) 抑制有效修正。

### Figure 12: λ_LVLM 灵敏度

![Figure 12](https://arxiv.org/html/2605.14966v1/x12.png)

**说明**: 包含 $\mathcal{L}_{\text{LVLM}}$ (λ_LVLM ≥ 0.1) 显著提升 Recall 和 F1，验证其在保持输出质量方面的作用。

### Figure 13: POPE 注意力可视化 (Qwen2.5-VL-7B)

![Figure 13](https://arxiv.org/html/2605.14966v1/x13.png)

**说明**: 问题 "Is there a person in the image?" (GT: Yes)。修正前注意力空间分散无法定位中心主体 → 错误回答 "No"；MHSA 修正后注意力集中在人物区域 → 纠正为 "Yes"。

### Figure 14: POPE 注意力可视化 (InternVL2-8B)

![Figure 14](https://arxiv.org/html/2605.14966v1/x14.png)

**说明**: 问题 "Is there a cup in the image?" (GT: No)。基线将噪声注意力分配给背景物品 → 误报 "Yes"；MHSA 抑制虚假激活 → 纠正为 "No"。

### Figure 15: POPE 注意力可视化 (LLaVA-v1.5-7B)

![Figure 15](https://arxiv.org/html/2605.14966v1/x15.png)

**说明**: 问题 "Is there a spoon in the image?" (GT: Yes)。基线注意力未能定位勺子 → 错误 "No"；MHSA 修正后注意力重定向到勺子位置 → 纠正为 "Yes"。

### Figure 16: Caption 注意力可视化 (Qwen2.5-VL-7B)

![Figure 16](https://arxiv.org/html/2605.14966v1/x16.png)

**说明**: 生成式任务中逐 token [[Cross-Modal Attention|跨模态注意力]]的修正前后对比。上图：修正前（幻觉）注意力图；下图：修正后（引导）注意力图；右侧：每 layer-head pair 的干预 delta 热力图。基线生成 "canned goods"、"mugs" 等幻觉物品，MHSA 修正后注意力重定向到实际视觉相关区域。

### Figure 17: MHSA 修正前后生成的 Caption 对比

![Figure 17](https://arxiv.org/html/2605.14966v1/x17.png)

**说明**: 与 Fig. 16 同图。红色文本为基线输出的幻觉内容；绿色粗体为 MHSA 修正后新增或正确 grounded 的内容。修正后移除不存在物体，正确识别 "wooden chair" 等基线遗漏的物品。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[MSCOCO|POPE-COCO]] | 3000 QA | Random/Popular/Adversarial 三类难度 | 判别式主实验 |
| [[Objects365|POPE-Objects365]] | 3000 QA | 多物体场景 | 数据集泛化 |
| [[OpenImages|POPE-OpenImagesV7]] | 3000 QA | 多样物体类别 | 泛化测试 |
| [[ImageNet-1k|POPE-ImageNet]] | 3000 QA | 通用物体识别 | 数据集泛化 |
| [[MSCOCO|COCO Caption]] | — | 图像描述 | 生成式 CHAIR 评估 |
| [[Flickr30k]] | — | 图像描述 | 生成式 CHAIR 评估 |

### 模型配置

| 模型 | L (层数) | H (头数) | N (视觉token) | d = L·H·N |
|------|---------|---------|-------------|-----------|
| Qwen2.5-VL-7B | 28 | 28 | 144 | 112,896 |
| InternVL2-8B | 32 | 32 | 256 | 262,144 |
| LLaVA-v1.5-7B | 32 | 32 | 576 | 589,824 |

### G 和 D 架构

| 组件 | 输入 | Hidden | 输出 | 结构 |
|------|------|--------|------|------|
| $G$ (生成器) | d | 512 | d | Lin-ReLU-Lin-ReLU-Lin |
| $D$ (判别器) | d | 128 | 2 | LN-Lin-ReLU-Lin |

### 训练超参数

| 任务 | 模型 | lr_G | lr_D | λ_LVLM | λ_dg | λ_reg | Epochs | Batch |
|------|------|------|------|---------|------|-------|--------|-------|
| POPE | Qwen2.5-VL | 1e-4 | 1e-5 | 1.0 | 0.01 | 1e-4 | 1 | 16 |
| POPE | LLaVA-v1.5 | 1e-4 | 1e-5 | 1.0 | 0.01 | 5e-4 | 1 | 8 |
| POPE | InternVL2 | 1e-3 | 1e-4 | 1.0 | 0.01 | 1e-4 | 1 | 8 |
| Caption | Qwen2.5-VL | 1e-3 | 1e-7 | 0.0 | 0.5 | 0.01 | 1 | 32 |

所有配置使用 weight decay=1e-4，1 epoch。训练数据 80/20 按 question ID 划分。Oversampling 策略：所有 hallucination 样本保留，non-hallucination 样本下采样至 hallucination 总数的一半。

---

## 主要结果

### Table 1: POPE results on MSCOCO (Qwen2.5-VL-7B, N=3000)

| Method | Accuracy | Precision | Recall | F1 | Yes% |
|--------|----------|-----------|--------|-----|------|
| Baseline | 86.83 | 95.27 | 77.62 | 85.55 | 40.9 |
| **MHSA** | **92.77** | 92.54 | **93.40** | **92.97** | 50.5 |
| $\Delta$ | +5.94 | -2.73 | +15.78 | +7.42 | +9.6 |

**关键发现**: F1 提升 +7.42，Recall 提升 +15.78（修正了对存在物体的漏判），Yes% 从 40.9% 回归到更平衡的 50.5%。

### Table 2: Dataset generalization on POPE (Qwen2.5-VL-7B)

| Dataset | Method | Acc | Prec | Recall | F1 | Yes% |
|---------|--------|-----|------|--------|-----|------|
| Objects365 | Baseline | 83.93 | 92.84 | 73.52 | 82.06 | 39.6 |
| | **MHSA** | **91.23** | 91.87 | **90.46** | **91.16** | 49.2 |
| ImageNet | Baseline | 82.80 | 96.33 | 68.20 | 79.86 | 35.4 |
| | **MHSA** | **86.67** | 84.25 | **90.20** | **87.12** | 53.5 |

**关键发现**: 统一超参数下跨数据集一致提升（Objects365 ΔF1=+9.10, ImageNet ΔF1=+7.26）。

### Table 3: Model generalization — InternVL2-8B

| Dataset | Method | Acc | Prec | Recall | F1 | Yes% |
|---------|--------|-----|------|--------|-----|------|
| COCO | Baseline | 87.07 | 90.86 | 82.54 | 86.50 | 45.6 |
| | **MHSA** | **93.87** | 90.16 | **98.54** | **94.16** | 54.9 |
| Obj365 | Baseline | 83.97 | 89.60 | 76.90 | 82.77 | 43.0 |
| | **MHSA** | **90.53** | 89.19 | **92.28** | **90.71** | 51.8 |
| OpenImg | Baseline | 79.53 | 74.77 | 89.77 | 81.58 | 60.6 |
| | **MHSA** | **83.73** | 75.74 | **99.74** | **86.10** | 66.5 |

### Table 3 (续): Model generalization — LLaVA-v1.5-7B

| Dataset | Method | Acc | Prec | Recall | F1 | Yes% |
|---------|--------|-----|------|--------|-----|------|
| COCO | Baseline | 85.57 | 91.49 | 78.55 | 84.53 | 43.1 |
| | **MHSA** | **92.10** | 90.70 | **93.89** | **92.27** | 52.0 |
| Obj365 | Baseline | 82.57 | 90.56 | 72.77 | 80.69 | 40.2 |
| | **MHSA** | **90.37** | 85.66 | **98.12** | **91.47** | 57.0 |
| OpenImg | Baseline | 78.50 | 72.69 | 90.09 | 80.46 | 60.9 |
| | **MHSA** | **81.07** | 74.20 | **96.05** | **83.72** | 63.3 |

**关键发现**: 6 种模型-数据集组合的 F1 全部提升（+3.26 到 +10.78）。InternVL2-8B on COCO 达到最高 F1=94.16 (ΔF1=+7.66)。

### Table 4: Cross-Dataset OOD Generalization (F1 scores)

| | | COCO (test) | Obj365 (test) |
|---|------|-------------|---------------|
| Qwen2.5-VL | Train COCO | **92.97** | 91.02 |
| | Train Obj365 | 93.48 | **91.16** |
| | Baseline | 85.55 | 82.06 |
| InternVL2-8B | Train COCO | **94.16** | 90.57 |
| | Train Obj365 | 95.77 | **90.71** |
| | Baseline | 86.50 | 82.77 |
| LLaVA-v1.5 | Train COCO | **92.27** | 89.50 |
| | Train Obj365 | 93.97 | **91.47** |
| | Baseline | 84.53 | 80.69 |

**关键发现**: 所有 OOD 条目的 F1 都超越对应 baseline。InternVL2-8B 在 Obj365 训练 → COCO 测试的 OOD F1=95.77 甚至超过 in-domain 的 94.16，表明 MHSA 学到的是**可泛化的注意力修正模式**而非数据集特定 artifact。

### Table 5: Caption Generation Results (CHAIR, Qwen2.5-VL-7B)

| Dataset | Method | CHAIR$_i$ $\downarrow$ | CHAIR$_s$ $\downarrow$ | Recall $\uparrow$ |
|---------|--------|------------------------|------------------------|-------------------|
| Flickr30k | Baseline | 16.43 | 37.50 | 86.63 |
| | **MHSA** | **9.20** | **21.00** | 83.28 |
| | $\Delta$ | **-44.0%** | **-44.0%** | -3.9% |
| COCO | Baseline | 5.68 | 21.00 | 56.42 |
| | **MHSA** | **5.19** | **18.00** | 55.14 |
| | $\Delta$ | **-8.6%** | **-14.3%** | -2.3% |

**关键发现**: Flickr30k 上 CHAIR 降低 44%，Recall 仅降 3.9%——幻觉抑制与内容保留的优良 trade-off。COCO 基线幻觉率已较低，仍实现 8.6%-14.3% 的降低。

---

## 消融实验

### Table 6: Ablation on Loss Functions (Qwen2.5-VL, POPE-COCO)

| Setting | $\mathcal{L}_{\text{LVLM}}$ | $\mathcal{L}_{\text{dg}}$ | $\mathcal{L}_{\text{reg}}$ | F1 |
|---------|:---------------------------:|:-------------------------:|:--------------------------:|-----|
| Baseline | — | — | — | 85.55 |
| w/o $\mathcal{L}_{\text{dg}}$ | ✓ | ✗ | ✓ | 88.51 |
| w/o $\mathcal{L}_{\text{LVLM}}$ | ✗ | ✓ | ✓ | 84.18 |
| w/o $\mathcal{L}_{\text{reg}}$ | ✓ | ✓ | ✗ | 89.60 |
| $\mathcal{L}_{\text{reg}}$ only | ✗ | ✗ | ✓ | 83.11 |
| **MHSA (full)** | ✓ | ✓ | ✓ | **92.97** |

**关键发现**: 移除任一损失项都会导致明显性能下降，三者互补。$\mathcal{L}_{\text{reg}}$ 单独作用几乎无益。

### Table 7 & 8: Inference Efficiency (Appendix)

| | Baseline | MHSA | $\Delta$ |
|--|----------|------|----------|
| Avg latency (ms) | 113.1 | 161.2 | +0.43× |
| Throughput | 8.84 | 6.20 | -0.30× |

| Sample Type | Ratio | Avg (ms) | Median (ms) |
|-------------|-------|----------|-------------|
| Non-Halluc. (no correction) | 87.7% | 115.1 | 114.9 |
| Hallucinated (corrected) | 12.3% | 486.4 | 205.5 |
| All | 100% | 161.2 | 115.1 |

**关键发现**: ammortized overhead 仅 +0.43×，显著优于对比解码方法的固定 +1× 开销。POPE 是 adversarial 构造的高幻觉率场景，实际部署中开销更接近 +0×。

### Table 25 (Appendix H): Caption Ablation (Flickr30k)

| Variant | CHAIR$_i$ $\downarrow$ | CHAIR$_s$ $\downarrow$ | Recall $\uparrow$ |
|---------|------------------------|------------------------|-------------------|
| Baseline | 16.43 | 37.50 | 86.63 |
| **Ours (full)** | **9.20** | **21.00** | **83.28** |
| w/o $\mathcal{L}_{\text{dg}}$ | 16.74 | 37.50 | 86.63 |
| w/o $\mathcal{L}_{\text{reg}}$ | 4.41 | 8.90 | 72.59 |
| w/ $\mathcal{L}_{\text{LVLM}}$ | 3.42 | 6.40 | 71.47 |

**关键发现**: 无 $\mathcal{L}_{\text{dg}}$ 时几乎无修正效果；无 $\mathcal{L}_{\text{reg}}$ 或加入 $\mathcal{L}_{\text{LVLM}}$ 时 CHAIR 极低但 Recall 大幅下降（过度生成抑制）。完整配置在幻觉降低(−44% CHAIR$_i$)和内容保留(−3.4% Recall)之间实现最佳平衡。

---

## 批判性思考

### 优点
1. **创新的范式转移**: 首次将跨模态注意力从幻觉"检测"扩展到"抑制"，且是第一个可学习的样本自适应注意力修正方法，超越了启发式规则
2. **轻量高效**: 仅训练小型 MLPs（G: 3层512维, D: 2层128维），不修改 LVLM 大参数，ammortized overhead 仅 +0.43×，且 87.7% 样本无需修正
3. **泛化性强**: 跨模型（3种架构）、跨数据集（4个 POPE）、跨分布（OOD）一致有效，甚至 OOD 性能 surpass in-domain
4. **分析深入**: 提供逐层、逐 head 的修正模式统计分析，解释 MHSA 作为"针对性修正"的工作机制
5. **统一判别式与生成式**: Token-level 扩展使同一框架覆盖两种任务范式

### 局限性
1. **训练需提取注意力**: 训练时需从 LVLM 前向传播中提取跨模态注意力，数据准备成本较高（Table 12 显示 80K-160K 训练样本需大量注意力提取）
2. **生成式任务未使用 $\mathcal{L}_{\text{LVLM}}$**: Caption setting 采用离线训练模式，$\mathcal{L}_{\text{LVLM}}=0$，理由是加入会导致过度生成抑制——这表明生成式场景的在线训练仍是未解决难题
3. **需要固定视觉 token 数**: 通过 resize 所有图像到固定分辨率来标准化注意力形状，可能丢失高分辨率细节
4. **未探索更丰富的生成器架构**: 仅使用三层 MLP，未测试 Transformer、GNN 等可能更有效的架构
5. **仅在 POPE/CHAIR 评测**: 未在更多样化的幻觉评测集（如 HallusionBench、M-HalDetect、MME）上验证生成式任务

### 潜在改进方向
1. **在线 token-level 训练**: 解决生成式场景中 $\mathcal{L}_{\text{LVLM}}$ 过约束的问题，可能通过稀疏奖励或 RL 实现在线学习
2. **更 expressive 的生成器**: 探索 Transformer-based attention corrector 或 graph-based 结构捕获 head 间交互
3. **动态视觉 token 数**: 支持原生可变分辨率注意力修正，避免 resize 导致的信息损失
4. **与其他策略结合**: 探索 MHSA + contrastive decoding 或 MHSA + instruction tuning 的协同效应
5. **更多评测**: 在 HallusionBench、M-HalDetect、MME 上验证生成式性能

### 可复现性评估
- [ ] 代码开源（待开源）
- [ ] 预训练模型（待发布）
- [x] 训练细节完整（超参数表非常详细）
- [x] 数据集可获取（POPE/MSCOCO/Flickr30k 均为公开数据集）

---

## 关联笔记

### 基于
- [[DHCP]]: 跨模态注意力幻觉检测，MHSA 将其判别器重用作 token-level 监督信号
- [[Adversarial Training|对抗训练]] (GAN): detector-guided loss 的设计灵感
- [[Residual Learning|残差学习]]: $\mathbf{A}' = \mathbf{A} + \Delta\mathbf{A}$ 的设计原则

### 对比
- [[OPERA]]: 基于 over-trust penalty 的启发式注意力修正，MHSA 用可学习方式替代
- [[PAI]]: 手动重新加权注意力增加图像 focus，MHSA 用数据驱动修正替代
- [[VCD]]: 对比解码需双倍前向传递，MHSA 仅在检测到幻觉时额外计算
- [[ICD]]: 指令对比解码，与 VCD 同属对比解码范式
- [[HALC]]: 自适应 focal-contrast 解码，仍属对比解码大类
- [[RLHF]]: 训练式幻觉缓解，需大量资源和标注数据

### 方法相关
- [[Cross-Modal Attention|跨模态注意力]]: 核心操作对象
- [[Hallucination Detector|幻觉检测器]]: D 的角色
- [[Attention Steering Generator|注意力引导生成器]]: G 的角色
- [[Detector-Guided Loss|检测器引导损失]]: 对抗训练风格的核心 loss
- [[Token-Level Hallucination Detection|Token 级幻觉检测]]: 从 sentence 到 token 的扩展
- [[LVLM Output Quality Loss|LVLM 输出质量损失]]: 保持输出质量的监督 loss

### 评测相关
- [[POPE]]: 判别式幻觉评测
- [[CHAIR]]: 生成式幻觉评测
- [[多模态幻觉]]: 核心问题域
- [[HallusionBench]]: 建议评测的额外基准

---

## 速查卡片

> [!summary] MHSA: Mitigating Hallucinations via Steered Attention
> - **核心**: 用可学习的三层 MLP 修正跨模态注意力模式，不修改 LVLM 参数即抑制幻觉
> - **方法**: Generator(3层MLP) + Discriminator(预训练DHCP) + 三目标联合优化(adversarial + L2 reg + LVLM quality)
> - **结果**: POPE-COCO F1 +7.42 (92.97), Flickr30k CHAIR -44%, 平均 latency +0.43×
> - **代码**: 待开源

---

*笔记创建时间: 2026-05-18*
