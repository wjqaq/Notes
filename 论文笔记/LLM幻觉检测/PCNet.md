---
title: "Hallucination as an Anomaly: Dynamic Intervention via Probabilistic Circuits"
method_name: PCNet
authors:
  - Erik Nielsen
  - Elia Cunegatti
  - Marcus Vukojevic
  - Giovanni Iacca
venue: arXiv
tags:
  - hallucination-detection
  - anomaly-detection
  - probabilistic-circuits
  - contrastive-decoding
  - density-estimation
  - representation-engineering
zotero_collection: LLM幻觉检测
year: 2026
image_source: online
arxiv_html: https://arxiv.org/html/2605.05953
created: 2026-05-12
---

# 论文笔记：Hallucination as an Anomaly: Dynamic Intervention via Probabilistic Circuits

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Trento, Italy |
| 日期 | May 2026 |
| 项目主页 | https://anonymous.4open.science/r/PC-LDCD-63D5 |
| 对比基线 | [[ITI]], [[DoLa]], [[ICD]], AdaSteer, SADI |
| 链接 | [arXiv](https://arxiv.org/abs/2605.05953) / [Code](https://anonymous.4open.science/r/PC-LDCD-63D5) |

---

## 一句话总结

> 用 [[Probabilistic Circuit]] 做精确密度估计来检测 [[LLM]] [[Hallucination]]，并通过门控的密度惩罚 [[Contrastive Decoding]] 实现只在异常时才触发的流形保持纠正。

---

## 核心贡献

1. **Detection-Correction Asymmetry 的系统分析**: 首次系统量化无差别 [[Representation Engineering]] 对正确生成的破坏（26%-90% 的正确生成被腐化）
2. **PCNet: 可处理的隐空间异常检测**: 基于 [[Probabilistic Circuit]] 的精确 [[Density Estimation]]，通过单次前向传播计算 [[Negative Log-Likelihood|NLL]]，无需采样、外部验证器或权重修改，AUROC 高达 99%
3. **PC-LDCD: 流形保持的幻觉纠正**: 动态门控 + 密度惩罚的 token 空间前瞻搜索，实现最低腐化率（53.7%）和最高保持率（79.3%），在 TruthfulQA 上 3/4 模型领先

---

## 问题背景

### 要解决的问题

[[LLM]] 生成流畅但事实错误的输出（[[Hallucination]]），现有检测和纠正方法存在根本矛盾：隐空间检测效果优秀，但直接编辑隐状态会破坏流利性和事实一致性。

### 现有方法的局限

- [[ITI]] 等方法无差别地对每个 token 施加纠正，导致原本正确的生成也被破坏
- Token-NLL、SEP 等基于输出概率的检测器因 LLM 过度自信而表现接近随机
- 现有方法未将检测信号与纠正机制解耦，造成语义崩溃

### 本文的动机

将检测信号与纠正机制解耦：隐空间几何仅用于诊断，纠正路由到安全的离散 token 空间。利用 [[Probabilistic Circuit]] 的可处理性，实现单次前向传播的精确密度评估，无需采样或外部依赖。

---

## 方法详解

### 模型架构

PCNet + PC-LDCD 采用 **两阶段门控架构**：

- **输入**: Prompt $x$， LLM 最后一层 [[Residual Stream]] 隐状态 $h \in \mathbb{R}^{D_{LLM}}$
- **信息瓶颈**: 2 层 MLP（ReLU）投影 $f_{\phi}: \mathbb{R}^{4096} \to \mathbb{R}^{128}$
- **核心模块**: [[Probabilistic Circuit|PCNet]] 用于精确 [[Negative Log-Likelihood|NLL]] 计算，[[Contrastive Decoding|PC-LDCD]] 用于门控纠正
- **输出**: 门控选择标准解码或密度惩罚前瞻搜索的 token
- **总参数**: PCNet 约 10K 节点，LLM 参数冻结

### 核心模块

#### 模块1: PCNet — 可处理的密度估计器

**设计动机**: 利用 [[Probabilistic Circuit]] 的结构保证实现精确边缘推断，捕捉 LLM 隐空间的重尾几何结构

**具体实现**:

**信息瓶颈投影**：将高维 [[Residual Stream]] $h$ 通过 $f_{\phi}$ 压缩到低维 $z \in \mathbb{R}^{D_{PC}}$（$D_{PC}=128 \ll 4096$），过滤语法噪声，保留语义和事实几何。

**DAG 结构**（深度 $L_{PC}=4$，分支因子 3）：
- **输入节点（异构混合叶）**: 每维 $z_i$ 用 [[Gaussian]] + [[Laplace]] + [[Student-T]] 混合分布建模，解决重尾几何问题
- **乘积节点**: 编码不相交特征子集的上下文特定独立性
- **求和节点**: 建模不同隐子群体（不同语义/事实流形）的凸组合
- **根节点 $\mathcal{C}_{\text{root}}$**: 计算联合非归一化对数概率，直接用作异常分数

#### 模块2: PC-LDCD — 流形保持纠正

**设计动机**: 解决 [[Detection-Correction Asymmetry]]，仅在隐状态偏离 [[Contrastive Manifold|事实流形]] 时才触发纠正

**具体实现**:
- 动态门控 $\beta_t = \sigma(\mathcal{S}_{\text{NLL}}(z_t) - \tau)$，$\tau$ 通过验证集 F1 最大化校准
- $\beta_t < 0.05$ 时跳过干预，使用标准 $O(1)$ 解码（理论保证：[[Detection-Correction Asymmetry|Proposition 2]]）
- 异常时：$k$-candidate 前瞻搜索，通过密度惩罚 score 选择 token
- 惩罚项 $\beta_t$ 动态缩放，安全时生成流畅，异常时约束偏离 token

---

## 关键公式

### 公式1: [[Probabilistic Circuit|概率电路节点定义]]

$$
\begin{aligned}
\text{input (leaf)}: \quad \mathcal{C}_{n}(z_{\mathrm{sc}(n)}) &= q_{n}(z_{\mathrm{sc}(n)};\eta_{n}) \\[4pt]
\text{sum node}: \quad \mathcal{C}_{n}(z_{\mathrm{sc}(n)}) &= \sum_{c\in\mathrm{ch}(n)} w_{n,c}\,\mathcal{C}_{c}(z_{\mathrm{sc}(c)}),\quad w_{n,c}\geq 0,\;\sum_{c}w_{n,c}=1 \\[4pt]
\text{product node}: \quad \mathcal{C}_{n}(z_{\mathrm{sc}(n)}) &= \prod_{c\in\mathrm{ch}(n)}\mathcal{C}_{c}(z_{\mathrm{sc}(c)})
\end{aligned}
$$

**含义**: 定义 PC 的三种节点运算：叶节点为参数化密度，求和节点为子节点凸组合，乘积节点为子节点乘积

**符号说明**:
- $\mathcal{C}_{n}$: 节点 $n$ 的密度函数
- $\mathrm{sc}(n)$: 节点 $n$ 的作用域（变量子集）
- $\mathrm{ch}(n)$: 节点 $n$ 的子节点集合
- $q_{n}(\cdot;\eta_{n})$: 叶节点的参数化密度（参数 $\eta_{n}$）
- $w_{n,c}$: 求和节点到子节点 $c$ 的权重

### 公式2: [[Density Estimation|异构混合叶节点对数似然]]

$$
\log P(z_i) = \sigma(g) \cdot \log \sum_{k\in\{G,L,T\}} w_k \exp(\log P_k(z_i \mid \mu, s, \nu))
$$

**含义**: 每个特征维度的叶节点使用 Gauss/Laplace/Student-T 三种分布的加权混合来建模 LLM 隐空间的复杂重尾分布

**符号说明**:
- $g$: 维度特定的可学习门控参数，动态缩放该维度的对数似然贡献
- $\mu$: 三种分布共享的位置参数
- $s$: 共享尺度参数
- $\nu$: Student-T 的自由度参数
- $w_k$: 各分布组件的混合权重

### 公式3: [[Negative Log-Likelihood|异常分数（NLL）]]

$$
\mathcal{S}_{\text{NLL}}(z) = -\log \mathcal{C}_{\text{root}}(z)
$$

**含义**: PCNet 根节点输出的负对数似然作为幻觉异常分数，高 NLL 表示隐状态偏离事实流形

**符号说明**:
- $\mathcal{S}_{\text{NLL}}(z)$: 投影向量 $z$ 的异常分数
- $\mathcal{C}_{\text{root}}(z)$: PCNet 根节点的密度值

### 公式4: [[Contrastive Manifold|对比流形优化损失]]

$$
\mathcal{L}(\theta,\phi) = \alpha \underbrace{\mathbb{E}_{h^{+}}[-\log\mathcal{C}_{\text{root}}(z^{+})]}_{\text{Generative NLL}} + (1-\alpha) \underbrace{\mathbb{E}_{h^{+},h^{-}}[\max(0, \gamma + \log\mathcal{C}_{\text{root}}(z^{-}) - \log\mathcal{C}_{\text{root}}(z^{+}))]}_{\text{Contrastive Margin}}
$$

**含义**: 联合优化生成密度估计（拟合事实状态）和对比边界惩罚（推离幻觉状态）

**符号说明**:
- $\theta$: PCNet 参数
- $\phi$: MLP 投影 bottleneck 参数
- $\alpha$: 损失权重（论文中 $\alpha=0.8$）
- $\gamma$: 几何边界（论文中 $\gamma=5.0$）
- $h^{+}, h^{-}$: 成对的事实性/幻觉性隐状态
- $z^{+}=f_{\phi}(h^{+}), z^{-}=f_{\phi}(h^{-})$: 投影后向量

### 公式5: [[Contrastive Decoding|PC-LDCD Token 选择分数]]

$$
\text{Score}_{\text{LDCD}}(c_i) = \log P_{\text{LM}}(c_i \mid x_{<t}) - \beta_{t} \cdot \mathcal{S}_{\text{NLL}}(f_{\phi}(h_{t+1}^{(c_i)}))
$$

**含义**: 从 top-k 候选中选择 token：平衡 LM 的生成置信度与候选 token 未来隐状态的密度惩罚

**符号说明**:
- $c_i$: 第 $i$ 个候选 token（从 top-k 原始 logits 中提取）
- $P_{\text{LM}}$: 冻结 LLM 的输出概率
- $h_{t+1}^{(c_i)}$: 选择 $c_i$ 后的假设未来隐状态
- $\beta_{t}$: 动态干预强度，$\beta_t = \sigma(\mathcal{S}_{\text{NLL}}(z_t) - \tau)$
- $\tau$: 基于验证集 F1 最大化校准的 NLL 阈值

### 公式6: [[Detection-Correction Asymmetry|动态门控强度]]

$$
\beta_t = \sigma(\mathcal{S}_{\text{NLL}}(z_t) - \tau)
$$

**含义**: Sigmoid 门控，隐状态越偏离事实流形，干预强度越接近 1

---

## 关键图表

### Figure 1: 系统概览 / Teaser

![Figure 1](https://arxiv.org/html/2605.05953v1/x1.png)

**说明**: PCNet 检测幻觉隐状态（通过精确 NLL），PC-LDCD 在离散 token 空间纠正，保留正确生成。示例展示 Qwen3-4B 上两个 prompt 的幻觉检测与纠正过程。

### Figure 2: 架构总览

![Figure 2](https://arxiv.org/html/2605.05953v1/x2.png)

**说明**: Phase 1（上）：$h_{\text{last}}$ 经 MLP bottleneck（4096→128）投影到 PCNet 做 NLL 计算。Phase 2（下）：NLL 门控判断是否触发 PC-LDCD 纠正。

### Figure 3: PCNet 密度模型示意

![[PCNet_fig3_x1.png]]

**说明**: (a) 事实性隐状态投影聚集在高密度区域，幻觉投影落入低密度异常区。(b) 逐 token NLL 轨迹：事实生成稳定，幻觉触发急剧上升并跨越检测阈值。

### Figure 4: 腐化率与保持率

![[PCNet_fig4_x2.png]]

**说明**: (a) 所有方法的腐化率（红）和保持率（绿）。（b）效用-真实性权衡：无门控触发语义崩溃，PCNet 门控恢复效用并移向最优前沿。

### Figure 5: 额外基准与消融

![[PCNet_fig5_x3.png]]

**说明**: (a)-(d) TruthfulQA MC1/MC2/MC3 + TriviaQA EM 对比 Vanilla/Un-Gated RAG/Gated RAG/PC-LDCD。(e)-(h) 训练数据量消融（$n\in\{50,\dots,1000\}$）。(i)-(l) MLP 投影维度消融（$d\in\{32,\dots,512\}$）。

### Figure 6: 训练数据量消融

![[PCNet_fig6_x4.png]]

**说明**: PCNet 在 Llama-3.2-1B 和 Mistral-7B 上的 AUROC vs 训练样本数。CoQA 上 $n=100$ 即达近峰值；TruthfulQA 在 $n=750$ 前逐步提升。

### Figure 7: MLP 投影维度消融

![[PCNet_fig7_x5.png]]

**说明**: 投影维度 $d\in\{32,64,128,256,512\}$ 的 AUROC 与隐对齐分析。$d=128$ 实现最高隐对齐（平均余弦相似度 0.19）。

### Table 1: 幻觉检测性能

| Model        | Method    | CoQA AUROC | SQuAD v2.0 AUROC | TriviaQA AUROC | TruthfulQA AUROC | Avg AUROC |
| ------------ | --------- | ---------- | ---------------- | -------------- | ---------------- | --------- |
| Llama-3.2-1B | Token NLL | 0.55       | 0.61             | 0.76           | 0.47             | 0.59      |
| Llama-3.2-1B | SEP       | 0.77       | 0.62             | 0.56           | 0.53             | 0.62      |
| Llama-3.2-1B | HaloScope | 0.85       | 0.95             | 0.86           | 0.61             | 0.82      |
| Llama-3.2-1B | **PCNet** | **0.95**   | **0.98**         | **0.91**       | **0.66**         | **0.88**  |
| Mistral-7B   | Token NLL | 0.56       | 0.62             | 0.82           | 0.52             | 0.63      |
| Mistral-7B   | HaloScope | 0.88       | 0.90             | 0.88           | 0.55             | 0.79      |
| Mistral-7B   | **PCNet** | **0.97**   | **0.98**         | **0.98**       | **0.79**         | **0.92**  |
| Qwen3-4B     | **PCNet** | **0.96**   | **0.97**         | **0.95**       | **0.81**         | **0.92**  |
| Llama-3.1-8B | **PCNet** | **0.98**   | **0.98**         | **0.97**       | **0.75**         | **0.92**  |

**说明**: PCNet 在所有模型和数据集上显著超越所有基线。Token NLL 和 SEP 接近随机猜测。HaloScope 在 TruthfulQA 上崩溃。TruthfulQA 对所有方法都是最难的数据集。

### Table 2: 门控优势总结

| Method      | SQuAD EM (G) | T+I (G)  | TriviaQA EM (G) | Corr. (%) | Pres. (%) |
| ----------- | ------------ | -------- | --------------- | --------- | --------- |
| DoLa        | 0.86         | 0.63     | 0.72            | 55.3      | 77.8      |
| ITI         | 0.56         | 0.42     | 0.50            | 63.5      | 76.9      |
| AdaSteer    | 0.87         | 0.66     | 0.75            | 55.8      | 78.1      |
| SADI        | 0.87         | 0.66     | 0.75            | 53.7      | 78.6      |
| ICD         | 0.81         | 0.64     | 0.70            | 57.0      | 77.2      |
| **PC-LDCD** | **0.87**     | **0.67** | **0.72**        | **53.7**  | **79.3**  |

**说明**: PC-LDCD 实现最低腐化率和最高保持率，同时在基准性能上匹配最佳基线。

### Table 3: TruthfulQA 纠正性能（门控模式）

| Model | Method | T+I $\uparrow$ | MC1 $\uparrow$ | MC2 $\uparrow$ | MC3 $\uparrow$ | IGR (%) |
|-------|--------|---------------|----------------|----------------|----------------|---------|
| Qwen3-4B | Vanilla | 0.50 | 0.33 | 0.50 | 0.11 | 0 |
| Qwen3-4B | **PC-LDCD** | **0.78** (+0.28) | **0.51** (+0.18) | **0.68** (+0.19) | **0.23** (+0.13) | 45.4 |
| Mistral-7B | Vanilla | 0.50 | 0.34 | 0.50 | 0.13 | 0 |
| Mistral-7B | **PC-LDCD** | **0.72** (+0.22) | **0.54** (+0.20) | **0.72** (+0.22) | **0.29** (+0.16) | 58.3 |
| Llama-3.1-8B | Vanilla | 0.50 | 0.29 | 0.44 | 0.08 | 0 |
| Llama-3.1-8B | **PC-LDCD** | **0.69** (+0.19) | **0.51** (+0.22) | **0.65** (+0.21) | **0.25** (+0.17) | 68.9 |

**说明**: PC-LDCD 在 4 个模型中 3 个取得最高 True+Info/MC2/MC3。MC1 不领先因为 MC1 奖励激进重排名，而 PC-LDCD 做分布级纠正。IGR 在 41%-77% 间变化，证实选择性干预。

### Table 4: 超参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| MLP 投影维度 $d$ | 128 | 信息瓶颈维度 |
| PC 深度 $L_{PC}$ | 4 | DAG 最大深度 |
| 分支因子 | 3 | 每层分支数 |
| 训练样本数 $n$ | 500 | 250 事实 + 250 幻觉 |
| Epochs | 50 | 训练轮数 |
| Batch size | 8 | 批次大小 |
| 学习率 | $10^{-3}$ | Adam 学习率 |
| 权重衰减 | $10^{-5}$ | L2 正则化 |
| $\alpha$ | 0.8 | 生成/对比损失权重 |
| $\gamma$ | 5.0 | 对比边界 |
| 梯度裁剪 | $\|\nabla\|_2 \leq 1.0$ | 数值稳定性 |

### Table 5: 标准 QA 基准性能

| Model | Method | CoQA F1 | SQuAD EM | TriviaQA EM | IGR (%) |
|-------|--------|---------|----------|-------------|---------|
| Llama-3.2-1B | Vanilla | 0.75 | 0.75 | 0.29 | 0 |
| Llama-3.2-1B | **PC-LDCD** | **0.74** | **0.85** | **0.30** | 77 |
| Mistral-7B | Vanilla | 0.74 | 0.85 | 0.62 | 0 |
| Mistral-7B | **PC-LDCD** | **0.74** | **0.87** | **0.64** | 41 |
| Llama-3.1-8B | Vanilla | 0.68 | 0.72 | 0.52 | 0 |
| Llama-3.1-8B | **PC-LDCD** | **0.66** | **0.80** | **0.55** | 67 |

**说明**: PC-LDCD 在标准 QA 上保持竞争力，未因选择性介入而损害基准性能。IGR 因模型而异。

---

## 实验

### 数据集

| 数据集        | 规模               | 特点       | 用途        |
| ---------- | ---------------- | -------- | --------- |
| CoQA       | ~8K 对话 QA        | 对话式推理    | 检测 + 纠正评估 |
| SQuAD v2.0 | ~12K QA + 不可回答问题 | 阅读理解和拒答  | 检测 + 纠正评估 |
| TriviaQA   | ~14K QA          | 知识密集型 QA | 检测 + 纠正评估 |
| TruthfulQA | 817 问题           | 针对预训练误解  | 检测 + 纠正评估 |

### 实现细节

- **Backbone**: Llama-3.2-1B, Qwen3-4B, Mistral-7B-v0.3, Llama-3.1-8B（全部冻结）
- **优化器**: Adam, 学习率 $10^{-3}$, 权重衰减 $10^{-5}$
- **Batch Size**: 8
- **训练轮数**: 50 epochs
- **校准数据**: 500 样本（250 事实 + 250 幻觉）
- **梯度裁剪**: $\|\nabla\|_2 \leq 1.0$
- **硬件**: 3 个随机种子，均值和标准差报告
- **PCNet 配置**: $D_{PC}=128$, 深度 4, 分支因子 3

### 额外基准：RAG 对比

| Method | TruthfulQA MC1 | TruthfulQA MC2 | TruthfulQA MC3 | TriviaQA EM |
|--------|---------------|---------------|---------------|-------------|
| Vanilla | 0.365 | 0.447 | 0.492 | 0.290 |
| Un-Gated RAG | 0.323 | 0.441 | 0.470 | **0.465** |
| Gated RAG | 0.357 | 0.455 | 0.485 | 0.448 |
| **PC-LDCD** | **0.570** | **0.616** | **0.669** | 0.290 |

**关键发现**: PC-LDCD 在所有 TruthfulQA 指标上远超 RAG；RAG 仅在 TriviaQA EM 上领先（因检索段落直接包含答案）。

### 可视化结果

- 消融确认：$n=100$ 训练样本即达 CoQA 近峰值 AUROC（0.99）
- 投影维度 $d=128$ 实现最高隐对齐
- 门控机制成功将无门控方法的 56.5% 平均腐化率降至 53.7%，保持率升至 79.3%

---

## 批判性思考

### 优点
1. **理论优雅**: 用 PC 的可处理性将幻觉检测形式化为精确密度估计，提供统计一致性保证（Appendix A）
2. **门控解耦检测与纠正**: 在隐空间检测（高精度）+ token 空间纠正（流形安全），避免激活空间编辑的语义崩溃
3. **极强数据效率**: 仅需 100 标注样本即可训练有效检测器，实际部署门槛低
4. **无侵入集成**: 不修改 LLM 权重，PCN 评分开销 <2% transformer 层前向传播

### 局限性
1. **校准数据需求**: 仍需少量标注的事实/幻觉隐状态对，新领域可能收集成本高
2. **模型规模有限**: 仅评估到 8B 参数，更大模型上的隐空间几何可能不同
3. **TruthfulQA 检测较弱**: AUROC 仅 0.66-0.81，因为 TruthfulQA 的误解深植于预训练权重
4. **精确匹配劣势**: PC-LDCD 在 TriviaQA EM 上不如 RAG，不擅长需要检索精确知识的任务
5. **匿名代码**: 当前仅有匿名仓库链接，GitHub 公开版本尚未找到

### 潜在改进方向
1. **早期检测**: 探索第一个 token 的隐状态是否已携带足够的事实信号
2. **CoT 步骤级纠正**: 对思维链的每个推理步骤独立评分，精确定位幻觉出现步骤
3. **PC-LDCD + RAG 混合**: 结合密度引导的真实性分布与检索增强的知识召回
4. **多模态扩展**: 将 PCNet 应用于 VLM 的多模态 residual stream

### 可复现性评估
- [ ] 代码开源（当前仅为匿名链接 https://anonymous.4open.science/r/PC-LDCD-63D5）
- [ ] 预训练模型
- [x] 训练细节完整（Appendix B 有完整超参数表）
- [x] 数据集可获取（CoQA, SQuAD v2.0, TriviaQA, TruthfulQA 均为公开数据集）

---

## 关联笔记

### 基于
- [[ITI]]: 推理时激活干预，本文证明其无差别纠正会破坏正确生成
- [[DoLa]]: 层间对比解码，本文在其 token 空间操作的安全性基础上增加了精确密度门控
- [[ICD]]: 诱导幻觉对比解码，本文改进为无需代理模型的精确密度对比
- [[Probabilistic Circuit]]: 概率电路的理论基础（Peharz et al. 2020; Choi et al. 2020）

### 对比
- [[ITI]]: 隐空间向量加法 vs 密度门控 token 空间前瞻搜索
- [[DoLa]]: 无门控层间对比 vs 有门控密度对比
- [[ICD]]: 代理模型对比 vs 精确 NLL 对比
- HaloScope: 特征提取检测 vs 精确密度估计检测

### 方法相关
- [[Probabilistic Circuit]]: 核心建模工具
- [[Contrastive Decoding]]: 纠正策略基础
- [[Density Estimation]]: 检测框架
- [[Representation Engineering]]: 对比的隐空间编辑范式
- [[Contrastive Manifold]]: 对比流形学习

### 硬件/数据相关
- CoQA: 对话推理基准
- TruthfulQA: 真理性评估基准

---

## 速查卡片

> [!summary] PCNet: Hallucination as an Anomaly
> - **核心**: 用 Probabilistic Circuit 做精确密度估计，将幻觉检测形式化为异常检测
> - **方法**: PCNet（密度估计器）+ PC-LDCD（门控密度惩罚解码）
> - **结果**: AUROC 高达 99%，腐化率 53.7%，保持率 79.3%，TruthfulQA 3/4 模型领先
> - **代码**: https://anonymous.4open.science/r/PC-LDCD-63D5

---

*笔记创建时间: 2026-05-12*
