---
title: "Token Preference Optimization with Self-Calibrated Visual-Anchored Rewards for Hallucination Mitigation"
method_name: "TPO"
authors: [Jihao Gu, Yingyao Wang, Meng Cao, Pi Bu, Jun Song, Yancheng He, Shilong Li, Bo Zheng]
year: 2024
venue: arXiv
tags: [hallucination-mitigation, preference-optimization, vision-language-model, token-level-reward, visual-anchoring]
zotero_collection: ""
image_source: local
arxiv_html: https://arxiv.org/html/2412.14487
created: 2026-05-19
---

# 论文笔记：Token Preference Optimization with Self-Calibrated Visual-Anchored Rewards for Hallucination Mitigation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Taobao & Tmall Group of Alibaba; Mohamed bin Zayed University of Artificial Intelligence |
| 日期 | December 2024 (arXiv v4: Sep 2025) |
| 项目主页 | https://github.com/alibaba/TPO |
| 对比基线 | [[LLaVA]], [[Qwen2-VL]], [[RLHF-V]], [[POVID]], [[CSR]], [[V-DPO]], [[MDPO]], [[R1-Onevision]] |
| 链接 | [arXiv](https://arxiv.org/abs/2412.14487) / [Code](https://github.com/alibaba/TPO) |

---

## 一句话总结

> 提出 TPO，通过自校准的视觉锚定 token 级奖励实现精细化偏好优化，无需细粒度标注即可大幅缓解 LVLM 幻觉。

---

## 核心贡献

1. **自校准视觉锚定奖励**: 通过对比原始图像和加噪图像下的 logits 分布差异，自动识别视觉锚定 token（visual-anchored tokens），无需人工标注即可生成 token 级奖励信号。
2. **Token Preference Optimization 损失**: 将 token 级视觉锚定奖励融入 DPO 框架，实现更精细的 token 级偏好对齐，显著强化模型对视觉信息的关注。
3. **无需细粒度标注**: 与 RLHF-V、V-DPO 等方法不同，TPO 完全消除了对细粒度人工标注的依赖，同时保持了 token 级优化的优势。
4. **SOTA 幻觉缓解性能**: 在 LLaVA-1.5 和 Qwen2-VL 上均取得最优幻觉缓解效果，AMBER F1 提升 20.4%，MMHal score 提升 22.8%。

---

## 问题背景

### 要解决的问题
[[LVLM]] 在视觉问答和图像描述任务中频繁出现 [[Hallucination]] 问题，即生成内容与输入图像不一致。这是制约 LVLM 实际部署可靠性的核心瓶颈。

### 现有方法的局限
现有基于 [[Direct Preference Optimization|DPO]] 的幻觉缓解方法存在两个关键缺陷：

1. **缺乏可扩展的 token 级奖励**: 现有方法（如 DPO、[[POVID]]）仅提供句子级全局奖励，无法对幻觉高发的特定 token（如物体名词、颜色属性）进行精细调控。而 [[RLHF-V]] 虽然提供了 token 级标注，但依赖昂贵的人工细粒度标注。
2. **忽视视觉锚定 token**: 由于大规模文本预训练带来的语言先验偏差，[[LVLM]] 倾向于优先依赖语言信息而非视觉信息，导致"视觉锚定 token"（如物体名称、属性词）更容易发生幻觉。现有方法未对这些高风险 token 进行特殊处理。

### 本文的动机
受 [[Visual Contrastive Decoding|VCD]] 思路启发——通过加噪图像对比来识别视觉敏感 token——作者提出将这一信号转化为可训练的 token 级奖励，并融入 DPO 框架，实现**无需标注的 token 级偏好优化**。

---

## 方法详解

### 模型架构

TPO 是一种训练时优化方法，不改变模型推理架构：

- **输入**: 图像 $v$ + 文本问题 $x$ + 正/负响应对 $(y_w, y_l)$
- **Backbone**: [[LLaVA|LLaVA-1.5]] / [[Qwen2-VL]]（仅训练 LLM 部分，冻结视觉编码器）
- **核心模块**: [[Self-Calibrated Rewards|自校准视觉锚定奖励]] 用于生成 token 级奖励信号；[[Token Preference Optimization loss]] 用于 token 级偏好对齐
- **输出**: 视觉感知增强的模型参数 $\pi_\theta$
- **训练数据**: RLHF-V 的 5K 偏好对（不使用其人工标注）

### 核心模块

#### 模块1: 视觉锚定奖励计算 (Visual-Anchored Rewards)

**设计动机**: 利用加噪前后模型对同一 token 的 logits 差异，量化该 token 对视觉信息的依赖程度。差异越大，说明 token 越"视觉锚定"，越需要被重点关注。

**具体实现**:
- 对输入图像 $v$ 进行 $k$ 步扩散加噪得到损坏图像 $v_c$
- 分别计算条件于原图和损坏图下的 token logits 分布
- 两者的差值 $s_{y_i}$ 即为视觉锚定分数（式 4）
- 通过 [[Self-Calibrated Rewards|自校准过程]]（式 5）将分数转换为实际奖励 $c_{y_i}$

#### 模块2: Token Preference Optimization 损失

**设计动机**: 将 token 级视觉锚定奖励融入标准 DPO 损失，使正样本中视觉锚定 token 的概率被放大，负样本中被抑制。

**具体实现**:
- 将每个 token 的条件概率 $p(y_i|x,v,y_{<i})$ 乘以奖励 $c_{y_i}$ 得到视觉感知概率 $\pi_\theta^v$
- 构建 KL 约束的奖励最大化目标（式 7）
- 推导出最终的 TPO 损失（式 12），等价于 DPO 损失 + token 级奖励修正项

---

## 关键公式

### 公式1: [[Direct Preference Optimization|DPO 奖励函数]]

$$
r(x, v, y) = \beta \log \frac{\pi_\theta(y|x, v)}{\pi_{\text{ref}}(y|x, v)}
$$

**含义**: DPO 将策略模型与参考模型的概率比作为隐式奖励，最大化正负样本间的奖励差异。

**符号说明**:
- $\pi_\theta$: 当前策略模型
- $\pi_{\text{ref}}$: 参考模型（初始冻结模型）
- $\beta$: KL 散度惩罚系数
- $x$: 文本输入
- $v$: 视觉输入

### 公式2: [[Direct Preference Optimization|DPO 最大似然目标]]

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x,v,y_w,y_l)\sim\mathcal{D}} \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x, v)}{\pi_{\text{ref}}(y_w|x, v)} - \beta \log \frac{\pi_\theta(y_l|x, v)}{\pi_{\text{ref}}(y_l|x, v)} \right)
$$

**含义**: 最大化正样本相对于负样本的隐式奖励差异。

**符号说明**:
- $\sigma(\cdot)$: sigmoid 函数
- $y_w$: 正样本（偏好响应）
- $y_l$: 负样本（非偏好响应）
- $\mathcal{D}$: 偏好数据集

### 公式3: [[Visual-Anchored Token|加噪图像生成]]

$$
v_c(k) = \sqrt{\bar{\xi}_k} \cdot v + \sqrt{1 - \bar{\xi}_k} \cdot \epsilon
$$

**含义**: 通过 $k$ 步扩散过程逐步向图像添加噪声，$\bar{\xi}_k = \prod_{i=0}^{k} \xi_i$ 为累积噪声参数。

**符号说明**:
- $v_c(k)$: $k$ 步加噪后的损坏图像
- $\xi$: 从 1,000 个等距点中取样的噪声参数，$\xi = \text{Sigmoid}(l_t) \times (0.5 \times 10^{-2} - 10^{-5}) + 10^{-5}$
- $l_t$: $[-6, 6]$ 区间等距取样的 1,000 个值
- $\epsilon \sim \mathcal{N}(0, 1)$: 标准高斯噪声

### 公式4: [[Visual-Anchored Rewards|视觉锚定分数]]

$$
s_{y_i} = p_{\log}(y_i | x, v, y_{<i}) - p_{\log}(y_i | x, v_c, y_{<i})
$$

**含义**: 计算同一 token $y_i$ 在原始图像和加噪图像条件下的 raw logits 差值。差值越大，说明该 token 越依赖视觉信息，即为视觉锚定 token。

**符号说明**:
- $p_{\log}$: 模型 raw logits（softmax 之前）
- $s_{y_i}$: token $y_i$ 的视觉锚定分数
- $y_{<i}$: 前 $i-1$ 个已生成 token
- $v_c$: 加噪后的损坏图像

### 公式5: [[Self-Calibrated Rewards|自校准视觉锚定奖励]]

$$
c_{y_i} = \begin{cases}
a + \sigma(s_{y_i}) & \text{if } y_i \in y_w \\
a + 1 - \sigma(s_{y_i}) & \text{if } y_i \in y_l
\end{cases}
$$

**含义**: 将视觉锚定分数映射为 token 级奖励。对正样本，分数越高奖励越大（鼓励关注视觉信息）；对负样本相反（抑制忽视视觉的 token）。$a=0.5$ 确保 $s=0$ 时 $c=1$（奖励无效）。

**符号说明**:
- $a$: 边际值，默认 0.5
- $\sigma(\cdot)$: sigmoid 归一化
- $c_{y_i} \in (0.5, 1.5)$: token 级奖励

### 公式6: [[Token-Level Rewards|视觉感知 token 概率]]

$$
\pi_\theta^v(y|x, v) = \prod_{y_i \in Y} c_{y_i}
$$

**含义**: 当所有 $c_{y_i}=1$ 时，视觉感知概率不额外累积。奖励 $c_{y_i} \neq 1$ 时对序列概率产生缩放效应。

### 公式7: [[Token Preference Optimization loss|TPO KL 约束奖励最大化目标]]

$$
\max_{\pi} \mathbb{E}_{(x,v,y)} \left[ r'(x, v, y) - \beta D_{\text{KL}}\left( \pi_\theta(y|x, v) \cdot \pi_\theta^v(y|x, v), \; \pi_{\text{ref}}(y|x, v) \cdot \pi_{\text{ref}}^v(y|x, v) \right) \right]
$$

**含义**: 在视觉感知概率空间中进行 KL 约束的奖励最大化。

**符号说明**:
- $D_{\text{KL}}$: [[KL Divergence]]
- $r'(x,v,y)$: 扩展的奖励函数

### 公式8: TPO 最优解

$$
\pi_\theta(y|x, v) \cdot \pi_\theta^v(y|x, v) = \frac{1}{Z(x, v)} \pi_{\text{ref}}(y|x, v) \cdot \pi_{\text{ref}}^v(y|x, v) \exp\left( \frac{1}{\beta} r'(x, v, y) \right)
$$

**含义**: KL 约束奖励最大化目标的闭式最优解，与标准 DPO 形式一致但作用于视觉感知概率空间。

**符号说明**:
- $Z(x, v) = \sum_y \pi_{\text{ref}}(y|v, x) \cdot \pi_{\text{ref}}^v(y|x, v) \exp(\frac{1}{\beta} r'(x, v, y))$: 配分函数

### 公式9: TPO 奖励函数展开

$$
\begin{aligned}
r'(x, v, y) &= \beta \log \frac{\pi_\theta(y|x, v) \cdot \pi_\theta^v(y|x, v)}{\pi_{\text{ref}}(y|x, v) \cdot \pi_{\text{ref}}^v(y|x, v)} + \beta Z(x, v) \\
&= \beta \sum_{y_i \in y} \left[ \log p_\theta(y_i|x, v, y_{<i}) - \log p_{\text{ref}}(y_i|x, v, y_{<i}) + \log \frac{c_{y_i}^\theta}{c_{y_i}^{\text{ref}}} \right] + \beta Z(x, v)
\end{aligned}
$$

**含义**: 相比于原始 DPO（式 1），TPO 在每个 token 上增加了一个自校准项 $\log(c_{y_i}^\theta / c_{y_i}^{\text{ref}}) \in (-\log 3, \log 3)$。正样本中此项增大，负样本中减小，推动模型生成时更关注视觉信息。

**符号说明**:
- $c_{y_i}^\theta$: 策略模型计算的 token 奖励
- $c_{y_i}^{\text{ref}}$: 参考模型计算的 token 奖励

### 公式10: [[Token Preference Optimization loss|TPO 损失（BT 形式）]]

$$
\mathcal{L}_{\text{TPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x,v,y_w,y_l)\sim\mathcal{D}} \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x, v) \cdot \pi_\theta^v(y_w|x, v)}{\pi_{\text{ref}}(y_w|x, v) \cdot \pi_{\text{ref}}^v(y_w|x, v)} - \beta \log \frac{\pi_\theta(y_l|x, v) \cdot \pi_\theta^v(y_l|x, v)}{\pi_{\text{ref}}(y_l|x, v) \cdot \pi_{\text{ref}}^v(y_l|x, v)} \right)
$$

**含义**: 基于 [[Bradley-Terry Model]] 的 TPO 最大似然目标，等价于 DPO 损失加 token 级视觉锚定修正：

$$
\mathcal{L}_{\text{TPO}} = \mathcal{L}_{\text{DPO}} + \mathbb{E} \log \sigma\left( \beta \log \frac{\pi_\theta^v(y_w|x, v)}{\pi_{\text{ref}}^v(y_w|x, v)} - \beta \log \frac{\pi_\theta^v(y_l|x, v)}{\pi_{\text{ref}}^v(y_l|x, v)} \right)
$$

### 公式11: TPO 损失（token 级展开）

$$
\begin{aligned}
\mathcal{L}_{\text{TPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x,v,y_w,y_l)\sim\mathcal{D}} \log \sigma\Bigg( &\beta \sum_{y_{w_i} \in y_w} \Big[ \log p_\theta(y_{w_i}|x, v, y_{w_{<i}}) - \log p_{\text{ref}}(y_{w_i}|x, v, y_{w_{<i}}) + \log \frac{c_{y_{w_i}}^\theta}{c_{y_{w_i}}^{\text{ref}}} \Big] \\
+ &\beta \sum_{y_{l_i} \in y_l} \Big[ \log p_\theta(y_{l_i}|x, v, y_{l_{<i}}) - \log p_{\text{ref}}(y_{l_i}|x, v, y_{l_{<i}}) + \log \frac{c_{y_{l_i}}^\theta}{c_{y_{l_i}}^{\text{ref}}} \Big] \Bigg)
\end{aligned}
$$

**含义**: 完整的 token 级训练损失，对 $y_w$ 和 $y_l$ 中每个 token 分别作用视觉锚定奖励修正项。

---

## 关键图表

### Figure 1: Visual Q&A Example / 视觉问答示例

<!-- Figure 1: 见原论文 PDF Figure 1 — 可视化问答案例，token 级视觉锚定奖励热力图 -->

**说明**: 一个视觉问答案例。上半部分为 ground truth 答案，下半部分为 LVLM 生成响应在 TPO 训练前后的对比。在每个框中，用颜色深浅可视化每个 token 的视觉锚定奖励。训练前模型错误声称"戴黑色眼镜"，训练后正确输出"未戴眼镜"。奖励分数有效反映了视觉锚定程度——颜色越深，token 对视觉信息依赖越强。

### Figure 2: TPO Pipeline Overview / TPO 流水线概览

<!-- Figure 2: 见原论文 PDF Figure 2 — TPO 三步骤流水线概览 -->

**说明**: TPO 的三大步骤：(1) 对图像添加噪声；(2) 计算自校准视觉锚定奖励（对比原图和加噪图下的 token logits 差异，通过 sigmoid 归一化和 margin $a$ 得到 $c_{y_i}$）；(3) Token Preference Optimization（将 token 级奖励融入 DPO 损失）。每个训练步结束后重新校准模型，为下一步生成新奖励。

### Figure 3: Attention Weights Comparison / 注意力权重对比

<!-- Figure 3: 见原论文 PDF Figure 3 — TPO 训练前后注意力权重对比 -->

**说明**: LLaVA 在 TPO 训练前后对图像 token 的注意力权重变化。蓝色部分为回答错误的 case（如 USB 被幻觉为其他物体），红色部分为回答正确的 case。TPO 训练后，响应 token 对图像的注意力权重显著增加，特别是对视觉锚定 token（如 "table"、"cord"）。

### Figure 4: Self-Calibration Convergence / 自校准收敛曲线

<!-- Figure 4: 见原论文 PDF Figure 4 — 自校准奖励收敛曲线 -->

**说明**: 正负样本的自校准奖励分数随训练步数的变化（每 10 步采样一个点）。正样本奖励 $(c_{y_i} \to 1.5)$ 逐步趋向最大值，负样本奖励 $(c_{y_i} \to 0.5)$ 逐步趋向最小值，展示了 TPO 的自校准效果——模型在训练中不断增强对视觉信息的关注。

### Figure 5: Ablation on Noise Steps and Parameter $\mathbf{a}$ / 消融实验曲线

<!-- Figure 5: 见原论文 PDF Figure 5 — 噪声步数与参数 a 消融实验曲线 -->

**说明**: (a) 噪声步数消融：最优步数为 500 步。0 步时仍优于 DPO（因图像编解码引入一定损失），250-999 步保持良好性能。(b) 参数 $a$ 消融：$a = [0, 0.5, 1]$ 均有效，$a = 0.5$ 综合最优，验证了 $s=0$ 时 $c=1$ 不引入额外奖励信号的设置合理。

### Figure 6: CHAIR Performance Comparison / CHAIR 指标对比

<!-- Figure 5: 见原论文 PDF Figure 5 — 噪声步数与参数 a 消融实验曲线 -->

**说明**: (a) 噪声步数消融；(b) 参数 $a$ 消融。500 步和 $a=0.5$ 综合最优。

### Figure 6: CHAIR Performance Comparison / CHAIR 指标对比

<!-- 见原论文 PDF Figure 6 — CHAIR 性能对比柱状图 -->

**说明**: AMBER 基准中物体幻觉评估（CHAIR 指标）的结果对比。TPO 不仅在视觉问答中缓解幻觉，还能有效减少图像描述中的物体幻觉。图中用 $10 - \text{CHAIR}$ 表示以便直观比较（越高越好）。

---

## 实验

### 数据集与基准

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RLHF-V 偏好数据 | 5K 偏好对 | 细粒度人工标注的幻觉纠正数据（不使用标注部分） | 训练 |
| [[AMBER]] | 15K+ 样本 | 多维度幻觉评测，含判别任务和生成任务 | 测试 |
| [[MMHal-Bench]] | - | GPT-4 评分，评估幻觉率和信息量 | 测试 |
| [[HallusionBench]] | - | 评估视觉错觉和知识幻觉，含原图/编辑图 | 测试 |
| [[SEED-Bench]] | - | 多模态生成式理解 | 通用测试 |
| [[MMBench]] | - | 多维度多模态能力评估 | 通用测试 |
| [[LLaVA-Bench]] | - | 多模态对话、描述、推理 | 通用测试 |
| [[MM-Vet]] | - | 综合多模态能力评估 | 通用测试 |

### 实现细节

- **Backbone**: LLaVA-1.5-7B / LLaVA-1.5-13B / Qwen2-VL-7B
- **训练策略**: 冻结 Vision Encoder，仅训练 LLM
- **优化器**: AdamW, 学习率 7B: 5e-8, 13B: 2e-7, Qwen2-VL-7B: 5e-9
- **训练轮数**: 4 epochs
- **硬件**: 8x NVIDIA A100 (7B), 32x A100 (13B)
- **推理**: Greedy decoding, seed=42
- **评价**: GPT-4-0613 8K 版本
- **框架**: PyTorch

---

### Table 1: 幻觉缓解方法对比

| Method | Visual-Anchored | Token-level | Non Fine-grained Annotations |
|--------|:---:|:---:|:---:|
| [[Direct Preference Optimization|DPO]] | X | X | V |
| [[POVID]] | X | X | V |
| [[CSR]] | V | X | V |
| [[MDPO]] | V | X | V |
| [[V-DPO]] | V | V | X |
| [[RLHF-V]] | X | V | X |
| **TPO (Ours)** | **V** | **V** | **V** |

**说明**: TPO 是首个同时满足视觉锚定关注、token 级奖励、且无需细粒度标注的方法。

### Table 2: 主要实验结果

| Method | AMBER Acc/F1 | MMHal Score/Hal | HallusionBench Easy/Hard/aAcc | SEED/MMB/LLaVA/MMVet |
|--------|-------------|------------------|-------------------------------|-----------------------|
| R1-Onevision | 80.2/85.7 | 3.85/36.46 | 63.74/50.47/62.80 | 35.2/-/83.7/67.8 |
| LLaVA-1.5-7B | 71.7/74.3 | 2.01/61.46 | 42.64/41.16/47.21 | 66.1/73.3/65.6/31.6 |
| + DPO | 77.5/82.1 | 2.14/58.33 | 37.36/37.21/43.84 | 66.4/73.3/69.1/31.6 |
| + CSR | 73.2/76.1 | 2.05/60.42 | 43.08/41.16/47.48 | 65.9/73.0/68.9/31.0 |
| + POVID | 71.9/74.7 | 2.26/55.21 | 42.86/41.63/47.56 | 66.1/73.2/68.2/31.7 |
| + RLHF-V | 74.8/78.5 | 2.02/60.42 | 42.20/43.72/48.27 | 66.1/73.1/68.0/32.3 |
| + MDPO | -/- | 2.39/54.00 | -/-/- | -/-/-/- |
| + V-DPO | -/81.6 | 2.16/56.00 | -/-/51.63 | -/-/-/- |
| **+ TPO (Ours)** | **79.3/85.0** | **2.47/51.04** | 41.76/**48.37**/**50.22** | **66.6**/**73.6**/**70.2**/**33.0** |
| LLaVA-1.5-13B | 71.3/73.1 | 2.38/53.13 | 44.40/36.51/46.94 | 68.2/76.7/73.1/36.1 |
| + DPO | 83.2/86.9 | 2.47/51.04 | 45.49/43.49/50.22 | 68.6/76.6/72.8/37.5 |
| + RLHF-V | 79.2/82.3 | 2.50/52.08 | 43.96/40.00/48.27 | 68.2/76.7/76.7/38.5 |
| **+ TPO (Ours)** | **83.9/88.0** | **2.72/45.83** | 44.40/**46.05**/**50.93** | **68.7**/**76.8**/72.8/36.2 |
| Qwen2-VL-7B | 86.5/90.0 | 3.5/29.0 | 67.0/48.8/64.0 | 45.0/79.0/82.4/61.4 |
| + DPO | 86.5/90.0 | 3.7/28.1 | 67.3/49.3/64.5 | 45.0/79.0/81.9/60.2 |
| **+ TPO (Ours)** | 86.4/89.9 | **4.2**/**18.8** | **67.9**/**50.0**/**65.2** | 45.0/79.0/82.9/61.4 |

**关键发现**:
- TPO 在 7B 上最高提升：AMBER F1 +20.4%, MMHal score +22.8%, HallusionBench aAcc +8.5%
- HallusionBench "Hard"（编辑图像题）上 TPO 提升最显著，说明方法有效增强了对视觉信息的关注而非语言先验
- Qwen2-VL 上 DPO 几乎无提升（5K 数据对强模型不够），但 TPO 仍有显著增益，体现更高数据利用效率
- 通用 benchmark 上 TPO 保持稳定或有提升，无 alignment tax

### Table 3: 消融实验 — 奖励分配方式

| Method | AMBER Acc/F1 | MMHal Score/Hal | HallusionBench Easy/Hard/aAcc | SEED/MMB/LLaVA/MMVet |
|--------|-------------|------------------|-------------------------------|-----------------------|
| LLaVA-1.5-7B | 71.70/74.3 | 2.01/61.46 | 42.64/41.16/47.21 | 66.1/73.3/65.6/31.6 |
| Only Win | 79.10/84.5 | 2.24/56.25 | 44.62/46.05/50.40 | 66.6/73.6/69.8/31.7 |
| Only Loss | 79.20/84.8 | 2.33/53.13 | 42.20/47.91/49.87 | 66.6/73.5/70.7/32.0 |
| Opposite | 75.30/80.7 | 1.91/64.58 | 42.42/45.58/48.63 | 65.6/73.1/68.9/32.1 |
| **TPO (Ours)** | **79.30/85.0** | **2.47/51.04** | 41.76/**48.37**/**50.22** | 66.6/**73.6**/**70.2**/**33.0** |

**说明**: "Only Win/Loss" 仅对正/负样本施加奖励。"Opposite" 对正样本施加与视觉相关性负相关的奖励。结果表明正负样本同时奖励效果最佳，反向奖励（Opposite）甚至比原始模型更差，验证了视觉锚定奖励设计的合理性。

### Table 4: 词性与视觉锚定分数分析

| | 平均分数 | Noun/Adj | Others |
|---|---------|----------|--------|
| Ground Truth | 1.83 | — | 0.90 |
| Ground Truth (TPO) | 5.72 | — | 4.87 |
| Response of LLaVA | 1.48 | — | 0.83 |
| Response of LLaVA+TPO | 5.67 | — | 4.59 |

**发现**: (1) 名词/形容词的视觉锚定分数显著高于其他词性，印证了视觉锚定奖励的有效性；(2) TPO 训练后所有 token 的视觉锚定分数均大幅提升，说明模型整体增强了对视觉信息的关注。

---

### Table 5: 噪声步数消融（详细）

| Method | AMBER Acc/F1 | MMHal Score/Hal | HallusionBench Easy/Hard/aAcc | SEED/MMB/LLaVA/MMVet |
|--------|-------------|------------------|-------------------------------|-----------------------|
| LLaVA-1.5-7B | 71.7/74.3 | 2.01/61.46 | 42.64/41.16/47.21 | 66.1/73.3/65.6/31.6 |
| 0 step | 77.6/82.6 | 2.10/58.33 | 44.40/45.35/49.42 | 66.2/73.2/69.9/32.1 |
| 250 steps | 79.0/84.5 | 2.33/53.13 | 43.52/46.05/49.51 | 66.6/73.4/68.5/31.3 |
| 750 steps | 79.30/85.0 | 2.40/52.08 | 41.76/48.14/50.04 | 66.7/73.5/69.2/32.8 |
| 999 steps | 79.20/85.0 | 2.41/52.08 | 41.76/47.67/49.69 | 66.7/73.5/69.2/33.3 |
| **500 steps (Ours)** | **79.30/85.0** | **2.47/51.04** | 41.76/**48.37**/**50.22** | 66.6/**73.6**/**70.2**/**33.0** |

### Table 6: 参数 $a$ 消融（详细）

| Method | AMBER Acc/F1 | MMHal Score/Hal | HallusionBench Easy/Hard/aAcc | SEED/MMB/LLaVA/MMVet |
|--------|-------------|------------------|-------------------------------|-----------------------|
| LLaVA-1.5-7B | 71.7/74.3 | 2.01/61.46 | 42.64/41.16/47.21 | 66.1/73.3/65.6/31.6 |
| a=0 | 79.2/83.0 | 2.24/56.25 | 42.20/43.72/48.27 | 66.6/73.5/68.4/32.8 |
| a=1 | 79.2/84.9 | 2.44/48.96 | 41.54/47.44/49.60 | 66.7/73.6/70.8/33.1 |
| **a=0.5 (Ours)** | **79.3/85.0** | **2.47/51.04** | 41.76/**48.37**/**50.22** | 66.6/**73.6**/70.2/**33.0** |

### Table 7: 与其他幻觉缓解方法的全面对比

| Method | AMBER Acc/F1 | MMHal Score/Hal |
|--------|-------------|------------------|
| LLaVA-1.5-7B | 71.7/74.3 | 2.01/61.46 |
| VCD | 71.8/74.9 | 2.12/54.20 |
| LURE | 73.5/77.7 | 1.64/60.40 |
| OPERA | 75.2/78.3 | 2.15/54.20 |
| HACL | -/- | 2.13/50 |
| EOS | -/- | 2.03/59 |
| HA-DPO | 75.2/79.9 | 1.97/60 |
| HALVA | 83.4/- | 2.25/54 |
| DPO | 77.5/82.1 | 2.14/58.33 |
| **TPO** | **79.3/85** | **2.47/51.04** |
| LLaVA-1.5-13B | 71.3/73.1 | 2.38/53 |
| HSA-DPO | -/- | 2.61/48 |
| HALVA | 86.5/- | 2.58/45 |
| DPO | 83.2/86.9 | 2.47/51 |
| **TPO** | **83.9/88** | **2.72/46** |

**说明**: TPO 在偏好对齐方法和解码策略方法中均取得最好结果。偏好对齐方法相比解码策略的优势：(1) 直接优化输出偏好；(2) 推理时不增加额外复杂度。两种方法可以互补。

### Table 8: 不同噪声添加方式对比

| Method | AMBER Acc/F1 | MMHal Score/Hal | HallusionBench Easy/Hard/aAcc |
|--------|-------------|------------------|-------------------------------|
| LLaVA-1.5-7B | 71.7/74.3 | 2.01/61.5 | 42.6/41.2/47.2 |
| +TPO (white) | 78.0/82.7 | 2.26/55.2 | 44.2/45.4/49.3 |
| **+TPO (noise)** | **79.3/85.0** | **2.5/51.0** | **41.8/48.4/50.2** |

**说明**: 用纯白图像替代加噪图像时性能下降。扩散加噪可以控制噪声水平产生更易诱发幻觉的图像，从而更好地识别视觉锚定 token。

### Table 9: 不同训练数据量对比

| Method | AMBER Acc/F1 |
|--------|-------------|
| LLaVA-1.5-7B | 71.7/74.3 |
| + DPO (5K) | 77.5/82.1 |
| + CSR (13K) | 73.2/76.1 |
| + POVID (17K) | 71.9/74.7 |
| + RLHF-V (5K) | 74.8/78.5 |
| + TPO (1K) | 72.5/75.3 |
| + TPO (3K) | 78.9/83.4 |
| **+ TPO (5K)** | **79.3/85.0** |

**发现**: TPO 用 1K 数据即可超越多数 baseline，3K 数据达到次优，展示了极高的数据效率。TPO 5K 数据胜过 CSR 13K 和 POVID 17K。

### 可视化结果

- **训练效率**: DPO 训练 1h24min，TPO 训练 1h57min（约 +40%），相比于需要细粒度标注的方法（CSR 需 13K、POVID 需 17K 数据），TPO 以 5K 数据和中等时间成本实现了更优性能。
- **自校准效果**: 训练过程中 $c_{y_i}$ 逐渐收敛，正样本趋于 1.5（最大奖励），负样本趋于 0.5（最小奖励），展示了持续的自校准能力。
- **注意力提升**: TPO 训练后 LLaVA-1.5-7B 的图像注意力权重从 0.14 提升至 0.17，尤其对视觉锚定 token 提升显著。
- **真实世界评估**: 对 10 张真实世界图像的评估显示，TPO 训练模型在 40% 的 case 中表现更好，原始模型仅 20%，40% 持平。

---

## 批判性思考

### 优点
1. **无需标注的 token 级优化**: 完全消除了对细粒度人工标注的依赖（与 RLHF-V、V-DPO 形成鲜明对比），同时实现了 token 级偏好对齐粒度，是以往方法无法兼得的。
2. **自适应奖励设计**: 自校准机制使奖励随模型变化动态更新，形成正向反馈循环（模型越关注视觉，奖励越精确，进而越促进视觉关注）。
3. **理论完备性**: 从 DPO 的 KL 约束奖励最大化框架严格推导出 TPO 损失，保持了理论的一致性，训练稳定。
4. **强数据效率**: 仅需 5K 数据超越 13K-17K 数据量下的其他方法，1K 数据即可超越多个 baseline。
5. **多 backbone 泛化**: 在 LLaVA-1.5 (7B/13B) 和 Qwen2-VL 上均有效，且对更强的 backbone 仍能带来增益。

### 局限性
1. **+40% 训练时间**: TPO 需要额外前传（加噪图像）计算 logits，训练时间比 DPO 多约 40%。虽然相比需要人工标注的方法已属高效，但对于超大规模模型仍需优化。
2. **仅对比单模态噪声**: 目前仅对图像添加噪声而未考虑对文本的扰动，可能无法识别文本主导的幻觉模式。
3. **限于 DPO 框架**: 方法基于 DPO 推导，尚未扩展到 [[PPO]]、[[GRPO]] 等 on-policy RL 方法（虽然作者讨论中提及可扩展性）。
4. **噪声步数敏感**: 需要调参噪声步数（最优 500 步），不同 backbone 可能需要不同设置。

### 潜在改进方向
1. **扩展到其他 RLHF 方法**: 将视觉锚定奖励作为 token 级权重融入 PPO/GRPO 等 on-policy 方法。
2. **对象级扰动**: 将全图加噪改为对特定关键物体的加噪，实现跨域视觉关注的定向增强（作者在 Limitation 节中提及）。
3. **多模态扰动**: 同时扰动图像和文本，识别跨模态纠缠导致的幻觉。
4. **减少前传开销**: 通过共享特征或一次性计算加噪 token logits 来减少 40% 的训练时间增量。

### 可复现性评估
- [x] 代码开源 (https://github.com/alibaba/TPO)
- [x] 预训练模型 (LLaVA-1.5, Qwen2-VL 均为开源)
- [x] 训练细节完整 (学习率、epochs、batch、硬件、超参数)
- [x] 数据集可获取 (RLHF-V 数据为开源)

---

## 关联笔记

### 基于
- [[Direct Preference Optimization]]: DPO 框架是 TPO 的理论基础，TPO 损失可分解为 DPO 损失 + token 级修正项
- [[RLHF-V]]: 提供了 5K 偏好对训练数据，TPO 在不使用其人工标注的情况下超越了它
- [[Visual Contrastive Decoding|VCD]]: TPO 借鉴了 VCD 通过加噪识别视觉敏感 token 的思想，但将其从推理时策略升级为训练时优化

### 对比
- [[POVID]]: 同样使用图像加噪构建负样本，但仅用于 DPO 数据构建而非 token 级奖励
- [[CSR]]: 使用 CLIP 计算图文相关性作为额外奖励，但需引入额外模型
- [[RLHF-V]]: token 级矫正但依赖人工标注
- [[V-DPO]]: 关注视觉锚定 token 但依赖合成数据集构建
- [[MDPO]]: 同时对图像和文本进行偏好优化，但仅为响应级奖励
- [[R1-Onevision]]: 基于 GRPO 的推理增强方法，非 DPO 框架

### 方法相关
- [[Token Preference Optimization loss|TPO Loss]]: 核心训练目标
- [[Visual-Anchored Rewards]]: 核心奖励机制
- [[Self-Calibrated Rewards]]: 自校准奖励映射过程
- [[Visual-Anchored Token]]: 视觉锚定 token 的定义与识别
- [[Token-Level Rewards]]: token 级奖励的概念与优势

### 硬件/数据相关
- [[AMBER]]: 幻觉评测 benchmark
- [[MMHal-Bench]]: 幻觉评测 benchmark
- [[HallusionBench]]: 幻觉评测 benchmark
- [[MMBench]]: 通用能力评测
- [[MM-Vet]]: 综合多模态能力评测
- [[CHAIR]]: 物体幻觉评测指标

---

## 速查卡片

> [!summary] Token Preference Optimization (TPO)
> - **核心**: 通过自校准视觉锚定奖励实现无需标注的 token 级 DPO，大幅缓解 LVLM 幻觉
> - **方法**: 对比原图/加噪图 logits 差异识别视觉锚定 token，将 token 级奖励融入 DPO 损失
> - **结果**: AMBER F1 +20.4%, MMHal +22.8%, HallusionBench aAcc +8.5%, 且通用能力无损
> - **代码**: https://github.com/alibaba/TPO

---

*笔记创建时间: 2026-05-19*
