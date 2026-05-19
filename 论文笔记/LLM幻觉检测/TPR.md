---
title: "Systematic Reward Gap Optimization for Mitigating VLM Hallucinations"
method_name: "TPR"
authors: [Lehan He, Zeren Chen, Zhelun Shi, Tianyu Yu, Jing Shao, Lu Sheng]
year: 2025
venue: NeurIPS
tags: [hallucination-mitigation, vlm-alignment, preference-optimization, curriculum-learning, data-curation, reward-gap]
zotero_collection: LLM幻觉检测
image_source: online  <!-- arXiv HTML: https://arxiv.org/html/2411.17265 -->
arxiv_html: https://arxiv.org/html/2411.17265
created: 2026-05-19
---

# 论文笔记：Systematic Reward Gap Optimization for Mitigating VLM Hallucinations

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Beihang University, Shanghai Innovation Institute, Shanghai AI Laboratory, Tsinghua University |
| 日期 | November 2024 (v4: November 2025) |
| 项目主页 | https://tpr-dpo.github.io |
| 对比基线 | [[RLAIF-V]], [[RLHF-V]], [[LLaVA-RLHF]], [[HSA-DPO]], [[POVID]], [[Silkie]] |
| 链接 | [arXiv](https://arxiv.org/abs/2411.17265) / [Code](https://tpr-dpo.github.io) |

---

## 一句话总结

> 通过 Topic 级别的选择性替换实现 Reward Gap 的系统性优化，用模型自身的重采样候选构造高质量偏好对，将 VLM 幻觉降低最高 93%。

---

## 核心贡献

1. **Reward Gap Configuration 的系统性优化**: 首次强调并形式化了数据策划中对 reward gap 配置进行精心设计的重要性，超越简单的 ranking 或 rewriting 范式。
2. **Topic-level Preference Rewriting (TPR)**: 提出 topic 级别的偏好重写框架，将响应分解为语义单元、按 topic 聚类后，利用模型自身进行 intra-topic 自重采样，实现精细粒度的语义控制。
3. **课程学习策略 (TPR-CL)**: 利用 TPR 的精细控制能力，设计从贪心 (greedy) 到难负样本挖掘 (hard negative mining) 的课程学习策略，逐步提升模型对细微幻觉的辨别能力。
4. **State-of-the-art 性能**: 在多个幻觉基准上取得最佳结果，超越先前方法平均约 20%，在 [[Object Hal-Bench|ObjectHal-Bench]] 上将幻觉降低 93%。

---

## 问题背景

### 要解决的问题
[[VLM|Vision Language Models]] 普遍存在 [[Hallucination|视觉幻觉]] 问题——模型会自信地描述不存在的物体、错误属性或错误空间关系。[[DPO|Direct Preference Optimization]] 已被广泛用于幻觉缓解，但其效果关键取决于偏好对 (preference pair) 中 **真实 reward gap** 的质量。

### 现有方法的局限
- **Ranking-based 方法** ([[RLAIF-V]], [[AMP]], [[FGAIF]]): 直接从可能有缺陷的模型输出中选择 $(y_w, y_l)$，不纠正潜在幻觉，可能导致低信息量和不足的 reward gap。
- **Rewriting-based 方法** ([[POVID]], [[HA-DPO]], [[HSA-DPO]]): 使用外部 "black-box" 模型 (如 GPT-4V) 改写响应，难以精确控制改写类型和幅度，且可能引入偏离模型内在 failure mode 的幻觉。
- 两者都缺乏对 **reward gap configuration** 的系统性优化——即刻意设计每个偏好对中 reward gap 的大小和难度分布。

### 本文的动机
通过 **topic 级别** 的精细控制（分解、聚类、重采样、选择性替换），实现对每个偏好对 reward gap 的精确塑造，并通过课程学习策略系统性地优化整体 reward gap 配置，引导模型克服从易到难的各类幻觉。

---

## 方法详解

### 模型架构

TPR 是一个 **数据策划框架**，不修改模型架构，而是通过 topic 级别的语义控制生成高质量偏好数据用于 [[DPO]] 对齐：

- **输入**: 图像 $I$ + 指令提示 $x$ + 参考模型 $\pi_{ref}$ + 标注模型 $\pi_{label}$
- **参考模型**: [[LLaVA|LLaVA-1.5-7B]] (生成候选响应、分解、重采样、改写)
- **标注模型**: [[LLaVA-NeXT|LLaVA-NeXT-34B]] (topic 内评分排序)
- **输出**: 高质量偏好数据集 $\mathcal{D}_{pref} = \{(I, x, y_w, y_l)\}$
- **训练**: 使用 [[DPO]] 在 8 张 [[A100]] 上微调 1 epoch

### 核心模块

#### 模块1: Topic-level Alternatives Generation (Topic 级替代方案生成)

**设计动机**: VLM 响应由多个语义 topic 组成（物体、属性、空间关系等），topic 之间相关性弱，可以相对独立地操控。

**具体实现**:
- **响应分解 (Decomposition)**: 对每张图像采样 $M=10$ 个候选响应，使用 $\pi_{ref}$ 将每个响应分解为 fine-grained 语义单元 (semantic units) $\{u_{m,1}, ..., u_{m,N_m}\}$
- **Topic 聚类 (Topic Clustering)**: 基于两重标准判断两个语义单元是否属于同一 topic:
  - [[文本一致性|Textual Consistency]]: 查询 $\pi_{ref}$ "Are $u_{m,n}$ and $u_{p,q}$ describing the same topic?"
  - [[Visual Correlation]]: 使用 [[CLIP]] 视觉编码器提取文本和图像嵌入，计算相似度向量之间的 [[Pearson Correlation|Pearson 相关系数]]，超过阈值 $\tau_{vis}=0.9$ 视为视觉相关
  - 采用 [[Louvain 方法|Louvain 算法]] 进行贪心聚类，最大化 modularity
- **Intra-Topic Self-Resampling (Topic 内自重采样)**: 对每个语义单元，先用 $\pi_{ref}$ 转为 wh-question (如 "The time on the Big Ben is 3:30." -> "What time is on the Big Ben?")，再多次查询 $\pi_{ref}$ 获取该 topic 的候选语义单元。相比重采样整个响应，topic 级重采样避免了同时要求所有单元正确的高难度，同时提供了选择性替换所需的细粒度候选池。

#### 模块2: Selective Topic Rewriting (选择性 Topic 重写)

**设计动机**: 通过选择性替换来控制 $y_w$ 和 $y_l$ 之间的语义差异，从而精确塑造 reward gap。

**具体实现**:
- **Intra-Topic Ranking (Topic 内排序)**: 将每个语义单元转为 yes-no 问题，使用 $\pi_{label}$ 计算 "Yes"/"No" 概率，得分 $S(u^c_{m,n}) = p_Y - p_N$，高分为 factual accurate
- **Selective Replacement (选择性替换)**: 随机选一个候选响应作为模板 $y_k$，通过策略 $\omega$ 从 topic 候选池中选择替代单元，分别替换模板中对应的语义单元构造 $y_w$ 和 $y_l$
- **In-Context Rewriting (上下文重写)**: 不直接替换文本（可能破坏流畅性），而是使用 $\pi_{ref}$ 在上下文中将选定的替代语义单元自然融入模板响应，保持逻辑结构和语言风格

**Greedy 策略**: 用最高分替代单元构造 $y_w$，最低分构造 $y_l$，构建高区分度的 reward gap。
**Curriculum 策略**: Warm-Up 阶段用 greedy 策略，Hard-Mining 阶段逐步提高 $y_l$ 中替代单元的得分（更难与 $y_w$ 区分），形成课程学习。

---

## 关键公式

### 公式1: [[DPO|DPO 损失函数]]

$$
\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]
$$

**含义**: 优化策略 $\pi_\theta$ 使其在偏好数据上满足偏好关系，同时通过 [[KL Divergence|KL 散度]] 惩罚约束不偏离 $\pi_{ref}$ 太远。

**符号说明**:
- $\pi_\theta$: 待优化的策略模型
- $\pi_{ref}$: 参考模型（初始策略）
- $\beta$: 控制偏好建模强度与 KL 约束的超参数
- $\sigma$: Sigmoid 函数
- $y_w, y_l$: 偏好对中的 preferred 和 rejected 响应

### 公式2: [[Reward Gap|估计的 Reward Gap]]

$$
M_{\pi_\theta} = \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
$$

**含义**: 策略模型估计的 log-probability 比率差，隐式反映 reward gap 的大小。

**符号说明**:
- $M_{\pi_\theta}$: 策略模型估计的 log-probability 比率差
- $\frac{\pi_\theta}{\pi_{ref}}$: 策略与参考的概率比率

### 公式3: [[DPO|DPO 梯度]]

$$
\nabla_\theta \mathcal{L}_{DPO} = -\beta \cdot \sigma(-\beta M_{\pi_\theta}) \left[\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)\right]
$$

**含义**: 梯度更新的方向由 preferred/rejected 响应的对数似然差值决定，**幅度**由 $\sigma(-\beta M_{\pi_\theta})$ 控制。当 $M_{\pi_\theta}$ 很大（模型已擅长区分），梯度趋近于 0；当 $M_{\pi_\theta}$ 接近 0（难以区分），梯度最大 (~0.5)。

**符号说明**:
- $\sigma(-\beta M_{\pi_\theta})$: 梯度幅度项，控制更新强度
- $\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)$: 梯度方向项

### 公式4: [[Reward Gap|真实 Reward Gap]]

$$
\Delta r^* = r^*(y_w, x) - r^*(y_l, x)
$$

**含义**: 由 oracle reward function 定义的偏好对真实难度。$\Delta r^* \gg 0$ 为"简单对"（明显正确 vs 明显错误），$\Delta r^* \to 0$ 为"困难对"（正确 vs 微妙错误）。

**符号说明**:
- $r^*(y, x)$: Oracle/真实奖励函数，完美捕获期望行为（如事实准确性）
- $\Delta r^*$: 真实 reward gap，是数据对的固有属性

### 公式5: [[文本一致性|Textual Consistency]]

$$
p_{text}(u_{m,n}, u_{p,q}) = \pi_{ref}(u_{m,n}, u_{p,q} \mid \text{"Are \{u_{m,n}\} and \{u_{p,q}\} describing the same topic? Please answer Yes or No."})
$$

**含义**: 通过查询参考模型判断两个语义单元是否描述同一核心主题。

### 公式6: [[Visual Correlation]]

$$
p_{vis}(u_{m,n}, u_{p,q}) = \text{Correlation}\left(\text{Sim}(u_{m,n}, v), \text{Sim}(u_{p,q}, v)\right) > \tau_{vis}
$$

**含义**: 利用 CLIP 视觉编码器，计算两个语义单元与图像 patch 嵌入的相似度向量之间的 [[Pearson Correlation|Pearson 相关系数]]，超过阈值即视为视觉相关。这使得即使文本相似但指向不同实体的单元能被正确区分。

**符号说明**:
- $v$: CLIP 视觉编码器提取的图像 patch 嵌入
- $\text{Sim}(u, v)$: 语义单元文本嵌入与视觉嵌入的相似度向量
- $\tau_{vis} = 0.9$: 视觉相关性的阈值

---

## 关键图表

### Figure 1: Topic-level Preference Rewriting 概览 + 数据效率

<!-- 对应 PDF 第 2 页，图片见 arXiv HTML https://arxiv.org/html/2411.17265 -->

**说明**: (a) TPR 的核心流程：基于不同策略（Greedy/Curriculum）从 topic 候选池中选择性替换语义单元，构造 $y_w$ 和 $y_l$。(b) 数据效率对比：除人工标注 ([[RLHF-V]]) 外，TPR 在幻觉降低上取得最佳数据效率。

### Figure 2: Topic 级替代方案生成流程

<!-- 对应 PDF 第 5 页，图片见 arXiv HTML https://arxiv.org/html/2411.17265 -->

**说明**: 初始候选响应被分解为语义单元后，基于 [[文本一致性]] 和 [[Visual Correlation]] 聚类为不同 topic（天气、时间、风格等），再通过 intra-topic self-resampling 为每个 topic 生成多样化的替代候选。

### Figure 3: 选择性替换构造偏好对

<!-- 对应 PDF 第 6 页，图片见 arXiv HTML https://arxiv.org/html/2411.17265 -->

**说明**: 从排序后的 topic 候选池中，根据策略（Greedy/Curriculum）选择替代单元，通过 in-context rewriting 融入模板响应，生成 $(y_w, y_l)$ 偏好对。Greedy 用最高/最低分单元，Curriculum 逐步提高 $y_l$ 中单元得分。

### Figure 4: 质量对比与幻觉类型分析

<!-- 对应 PDF 第 9 页，图片见 arXiv HTML https://arxiv.org/html/2411.17265 -->

**说明**: (a) 不同数据策划策略生成的响应与 GPT-4V "GroundTruth" 响应的 win-rate 比较，TPR 和 TPR-CL 均超过 50%。(b) GPT-4V 对 topic 替代方案的 informative/trustworthy 评分。(c) 外部 rewriter 引入的幻觉类型分布与模型自身 failure mode 分布存在显著差异。

### Figure 5: Visual Correlation 机制

<!-- 对应 PDF 第 24 页，图片见 arXiv HTML https://arxiv.org/html/2411.17265 -->

**说明**: CLIP 编码器提取文本和图像嵌入，计算相似度向量之间的 Pearson 相关系数。即使 "The time on the Big Ben is 10:30" 和 "The clock shows it is about 6:20" 文本语义相似，通过视觉相关性能将它们正确关联到同一 tower 的不同时间描述。

### Figure 6: Hard Mining 有效性分析

<!-- 对应 PDF 第 25 页，图片见 arXiv HTML https://arxiv.org/html/2411.17265 -->

**说明**: 在 [[RefoMB]] 基准上按幻觉难度（Existence、Attributes、Quantities、Spatial Relations）分析 TPR-CL 的改进。对于 baseline [[LLaVA|LLaVA-1.5]] 表现最差的 "Quantities" (16.7%) 和 "Spatial Relations" (14.3%)，TPR-CL 额外带来 +20.8/+35.7 个百分点的提升。

### Figure 7: 成本效益分析

<!-- 对应 PDF 第 25 页，图片见 arXiv HTML https://arxiv.org/html/2411.17265 -->

**说明**: TPR 在最低成本下实现最陡峭的幻觉降低曲线。GPT-4V rewriting 方法成本约 $30+，TPR 仅需约 $10；[[RLAIF-V]] 的多轮迭代再生和重训练带来额外计算开销。

### Figure 8: 更大基座模型的扩展性

<!-- 对应 PDF 第 25 页，图片见 arXiv HTML https://arxiv.org/html/2411.17265 -->

**说明**: TPR-CL 应用于 7B/13B 模型在 2k~20k 数据规模上的表现，13B 模型在每个数据点都优于 7B，验证了 TPR 与模型规模的互补性。

### Figure 9: 定性结果 — Greedy vs Curriculum

<!-- 对应 PDF 第 31 页，图片见 arXiv HTML https://arxiv.org/html/2411.17265 -->

**说明**: 比较 TPR (Greedy) 和 TPR-CL (Curriculum) 在相同图像上生成的 preferred/rejected 响应。Greedy 的 rejected 响应包含明显幻觉（如 "paper plates"、"13 people"），Curriculum 的 rejected 响应包含更细微的幻觉（如 "bowl"、"toaster"、"several other skiers and snowboarders"）。

### Figure 10: 定性结果 — TPR vs LLaVA-1.5 vs LLaVA-NeXT-34B

<!-- 对应 PDF 第 33-34 页，图片见 arXiv HTML https://arxiv.org/html/2411.17265 -->

**说明**: TPR-7B 对齐后的模型在多个场景下生成比 LLaVA-1.5-7B 和更大 [[LLaVA-NeXT|LLaVA-NeXT-34B]] 更准确的描述，尤其在避免"无中生有"（如 traffic lights、handbag）方面表现突出。

### Table 1: 主要实验结果

| Model | ObjHal CH$_s$$\downarrow$ | ObjHal CH$_i$$\downarrow$ | MMHal Score$\uparrow$ | AMBER Hall.$\downarrow$ | POPE Acc.$\uparrow$ | POPE F1$\uparrow$ | RefoMB Trust.$\uparrow$ | RefoMB Win.$\uparrow$ | LLaVA-B Acc.$\uparrow$ | MMstar Acc.$\uparrow$ |
|------|------|------|------|------|------|------|------|------|------|------|
| LLaVA-RLHF-13B | 38.1 | 18.9 | 2.02 | 62.5 | 79.7 | 83.9 | 26.3 | 17.2 | 61.5 | 34.2 |
| RLHF-V-13B | 12.2 | 7.5 | 2.45 | 51.0 | 72.6 | 75.0 | 41.4 | 17.7 | 51.4 | 33.2 |
| Silkie-10B | 27.1 | 13.4 | 3.19 | 32.3 | 82.2 | 87.6 | 38.9 | 21.2 | 73.2 | 33.6 |
| POVID-7B | 48.1 | 24.4 | 2.08 | 56.2 | 82.9 | 87.4 | 44.4 | 13.6 | 62.2 | 34.3 |
| HA-DPO-7B | 39.9 | 19.9 | 1.98 | 60.4 | 75.2 | 79.9 | 39.9 | 17.2 | 67.2 | 32.9 |
| OPA-DPO-7B | 13.0 | 4.3 | 2.83 | 45.8 | 81.3 | 85.6 | 39.4 | 18.2 | 62.2 | 32.2 |
| AMP-MEG-7B | 37.8 | 22.5 | 3.17 | 35.0 | 78.3 | 83.6 | 42.9 | 18.7 | 54.6 | 27.5 |
| RLAIF-V-7B | 8.5 | 4.3 | 3.06 | 29.2 | 76.8 | 84.5 | 47.5 | 20.7 | 64.9 | 31.8 |
| FGAIF-7B | 6.2 | 3.9 | 3.09 | 36.0 | – | – | – | – | – | – |
| HSA-DPO-13B | 5.3 | 3.2 | 2.61 | 48.0 | – | – | – | – | – | – |
| LLaVA-1.5-7B | 53.6 | 25.2 | 2.36 | 51.0 | 73.5 | 77.6 | 30.8 | 12.1 | 59.7 | 30.3 |
| **+TPR-7B** | **4.0** | **2.2** | **3.01** | **31.2** | **82.3** | **87.6** | **58.1** | **31.3** | **69.2** | **33.2** |
| **+TPR-CL-7B** | **3.4** | **1.8** | **3.06** | **30.2** | **82.7** | **87.8** | **61.1** | **32.3** | **71.1** | **33.3** |

**说明**: TPR 和 TPR-CL 在几乎所有幻觉基准上取得最优。TPR-CL 在 [[Object Hal-Bench|ObjectHal-Bench]] 上将幻觉降低约 93%（CH$_s$: 53.6 -> 3.4），同时在通用能力基准上保持或提升性能（无 [[Alignment Tax]] 现象）。

### Table 2: 消融实验 — 各组件贡献

| Exp | Multi-Res. | Decompose | Intra-Topic Rsp. | Strategy | In-Ctx. | ObjHal CH$_s$$\downarrow$ | ObjHal CH$_i$$\downarrow$ | AMBER Acc.$\uparrow$ | AMBER F1$\uparrow$ |
|-----|-----------|-----------|-----------------|----------|--------|------|------|------|------|
| (2a) | Y | Y | Y | SR+Both+Greedy | Y | **5.9** | **3.1** | **82.1** | **87.0** |
| (2b) | Y | Y | Y | SR+Both+Random | Y | 29.7 | 13.1 | 78.9 | 84.1 |
| (2c) | Y | Y | Y | SR+Pref+Greedy | Y | 6.4 | 3.2 | 76.8 | 84.7 |
| (2d) | Y | Y | Y | SR+Both+Greedy | N | 35.5 | 20.1 | 80.1 | 84.4 |
| (2e) | Y | Y | N | SR+Both+Greedy | Y | 7.2 | 4.0 | 80.2 | 86.4 |
| (2f) | Y | N | N | SR+Both+Greedy | Y | 12.6 | 6.6 | 77.9 | 84.9 |
| (2g) | Y | Y | Y | Ranking | – | 9.7 | 4.8 | 80.9 | 85.9 |
| (2h) | Y | N | – | Ranking | – | 25.5 | 12.0 | 73.5 | 82.8 |
| (2i) | – | – | – | Rewrite+Pref | – | 11.3 | 9.8 | 79.3 | 84.5 |
| (2j) | – | – | – | Rewrite+Both | – | 15.6 | 12.4 | 78.4 | 83.6 |

**关键发现**:
- 随机选择 (2b vs 2a): 性能大幅下降，证实了刻意设计 reward gap 的重要性
- 无 In-Context Rewriting (2d): 严重退化，直接文本替换破坏流畅性
- 无 Intra-Topic Resampling (2e): 候选多样性不足，影响 reward gap 的塑造空间
- 无 Decomposition (2f): 粗粒度控制，无法进行 topic 级精细操控
- Ranking/Rewriting baseline (2g-2j): 均不如 TPR 的 selective replacement 机制

### Table 3: Topic Clustering 消融

| Condition | $\tau_{vis}$ | ObjHal CH$_s$$\downarrow$ | ObjHal CH$_i$$\downarrow$ | AMBER Acc.$\uparrow$ | AMBER F1$\uparrow$ |
|-----------|-------------|------|------|------|------|
| $p_{text}$ only | – | 13.4 | 6.8 | 80.1 | 85.9 |
| $p_{vis}$ only | 0.9 | 13.0 | 7.3 | 80.1 | 86.1 |
| | 0.6 | 8.3 | 4.1 | 80.8 | 86.1 |
| $p_{text} + p_{vis}$ | 0.8 | 7.2 | 3.5 | 82.5 | 86.9 |
| | **0.9** | **5.9** | **3.1** | **82.1** | **87.0** |
| | 0.95 | 6.7 | 3.5 | 81.8 | 86.8 |

**关键发现**: 文本一致性 + 视觉相关性双重标准显著优于单一标准；$\tau_{vis}=0.9$ 为最佳阈值。

### Table 4: 计算成本分解

| Stage | TPR | TPR w/ vLLM |
|-------|-----|------------|
| Response Generation | 10.27h | 1.36h |
| Decomposition | 8.91h | 3.57h |
| Wh-question Conversion | 7.95h | 3.17h |
| Self-resampling | 7.48h | 2.96h |
| Topic Cluster | 10.84h | 6.71h |
| Scoring & Ranking | 23.43h | 7.42h |
| In-context Rewriting | 2.17h | 0.85h |
| **Total** | **71.05h** | **26.04h** |

**说明**: 生成 20k 偏好数据在 8 张 A100 上需 71.05h，使用 [[vLLM]] 推理加速后降至 26.04h（4.7 GPU-seconds/pair）。

### Table 7: Self-Labeling 和跨架构泛化

| Model | ObjHal CH$_s$$\downarrow$ | ObjHal CH$_i$$\downarrow$ | MMHal Score$\uparrow$ | AMBER Hall.$\downarrow$ | POPE F1$\uparrow$ |
|-------|------|------|------|------|------|
| LLaVA-1.5-7B | 53.6 | 25.2 | 2.36 | 51.0 | 85.9 |
| +TPR-SL | 5.8 | 3.0 | 2.67 | 44.8 | 86.1 |
| Qwen-VL-2B | 42.4 | 36.3 | 2.85 | 47.9 | 86.5 |
| +NaiveRLAIF | 36.5 | 31.7 | 2.81 | 49.0 | 87.0 |
| +TPR-SL-2B | 19.5 | 13.7 | 2.98 | 43.8 | 87.1 |

**关键发现**: TPR-SL (使用参考模型自身作为 labeler) 仍能取得显著改进，验证了自我改进能力；TPR 范式在 [[Qwen-VL]] 架构上同样有效。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| VQAv2 | – | 视觉问答 | 偏好数据源 |
| MSCOCO | – | 图像描述 | 偏好数据源 |
| ShareGPT-4V | – | 多模态对话 | 偏好数据源 |
| TextVQA | – | 文本视觉问答 | 偏好数据源 |
| MovieNet | – | 视频理解 | 偏好数据源 |
| OKVQA | – | 外部知识 VQA | 偏好数据源 |
| Google Landmark v2 | – | 地标识别 | 偏好数据源 |

### 评估基准

| 基准 | 评估内容 | 指标 |
|------|---------|------|
| [[Object Hal-Bench|ObjectHal-Bench]] | 物体幻觉（描述性） | CH$_s$, CH$_i$ |
| [[MMHal-Bench]] | 幻觉 + 信息量 | Score (GPT-4) |
| [[AMBER]] | 物体存在/属性/关系 | Accuracy, F1 |
| [[RefoMB]] | 8 项基础能力（幻觉+推理） | Trust., Win. |
| [[POPE]] | 物体存在（yes-no 问答） | Acc., F1 |
| [[LLaVA-Bench]] | 多模态对话/描述/推理 | Overall |
| [[MMStar]] | 6 项核心能力 + 18 项具体 | Overall |

### 实现细节

- **Backbone**: LLaVA-1.5-7B（作为 $\pi_{ref}$ 和 $\pi_\theta$）
- **Labeler**: LLaVA-NeXT-34B（作为 $\pi_{label}$）
- **优化器**: [[AdamW]], lr=$5 \times 10^{-7}$, cosine decay
- **Batch Size**: 8
- **训练轮数**: 1 epoch
- **硬件**: 8 NVIDIA [[A100]] GPUs
- **偏好数据量**: 20,000 对
- **TPR-CL 划分**: 12,000 (60%) Warm-Up + 8,000 (40%) Hard-Mining
- **生成参数**: temperature=0.7, top_p=0.95

### 数据效率分析

TPR 在自动化方法中展示最佳数据效率：
- [[RLHF-V]] (1.4k 人工标注): 初始效率高但成本不可扩展
- [[RLAIF-V]] (22k): 表现看似有竞争力，但实际需要多轮再生+重训练（等效 88k 处理量）
- TPR (20k): 在 4k 到 20k 数据量范围内快速降低幻觉率，最终超越人工标注方法的性能

---

## 批判性思考

### 优点
1. **核心洞察力深刻**: 将 DPO 对齐问题从"如何生成偏好对"提升到"如何系统性优化 reward gap 配置"，提供了新的理论视角
2. **精细化控制**: Topic 级别的选择性替换提供前所未有的细粒度语义控制，是有原则的数据策划范式的优秀示范
3. **实验全面扎实**: 涵盖 7 个评估基准、10+ baselines、丰富的消融实验（组件、策略、聚类机制、数据效率、成本分析、自我标注、跨架构泛化）
4. **实用性强**: 不需要外部 proprietary API (如 GPT-4V)，可完全开源部署；支持 self-labeling 模式大幅降低门槛；成本远低于 rewriting-based 方法

### 局限性
1. **Pipeline 复杂度**: TPR 涉及 7 个阶段（响应生成->分解->wh-问题转换->自重采样->聚类->评分排序->上下文重写），流程较长，增加了工程实现难度和出错概率
2. **Topic 独立性假设**: 假设语义 topic 之间相关性弱可独立操控，但在复杂推理场景中 topic 之间可能有强依赖关系
3. **仅限于感知型幻觉**: 当前主要解决 perceptual hallucinations（物体存在、属性、空间关系），对逻辑/因果推理中的幻觉不直接适用
4. **聚类超参数敏感**: $\tau_{vis}$ 等超参数对性能有明显影响，不同模型/场景可能需要重新调参

### 潜在改进方向
1. **简化 Pipeline**: 探索是否可以省略聚类步骤，直接用更简单的方法关联语义单元
2. **扩展到推理型幻觉**: 将 topic 级控制适配到 multimodal reasoning tasks 中的逻辑链错误纠正
3. **自动化课程调度**: 探索更智能的课程调度策略（如基于模型当前表现的 adaptive curriculum），替代固定的 60%/40% 划分
4. **多维度优化**: 除难度维度外，探索其他数据策划维度（如多样性、覆盖率等）的系统性优化

### 可复现性评估
- [x] 代码开源 (https://tpr-dpo.github.io)
- [x] 预训练模型 (使用公开 LLaVA-1.5)
- [x] 训练细节完整 (超参数、prompt 模板、数据源均在论文和附录中提供)
- [x] 数据集可获取 (基于 7 个公开数据集)
- [x] 全部 prompts 在附录 Table 5/6 中完整列出

---

## 关联笔记

### 基于
- [[DPO|Direct Preference Optimization]]: TPR 的核心对齐算法，TPR 为 DPO 提供系统性优化的偏好数据
- [[RLHF-V]]: 细粒度人工反馈的偏好学习先驱，TPR 沿用了其分解和评分思路但自动化了流程
- [[RLAIF-V]]: Divide-and-Conquer 策略的 AI 反馈方法，TPR 继承但改进为其 topic 级选择性替换

### 对比
- [[POVID]]: Rewriting-based 方法，依赖 GPT-4V 修改图像/注入幻觉，TPR 用自重采样替代外部模型
- [[HSA-DPO]]: Hallucination Severity-Aware DPO，focus 在严重度评估，TPR focus 在 reward gap 配置
- [[AMP]]: Multi-level preference，通过多尺度对比构造偏好，TPR 通过 topic 级替换构造

### 方法相关
- [[Topic-level Preference Rewriting]]: 核心方法
- [[Reward Gap Configuration]]: 核心理论概念
- [[Intra-Topic Self-Resampling]]: 核心组件之一
- [[Selective Topic Replacement]]: 核心组件之一
- [[In-Context Rewriting]]: 保证流畅性的关键技术
- [[Curriculum Learning]]: TPR-CL 的策略框架
- [[Hard Negative Mining]]: TPR-CL 中使用的具体策略

### 硬件/数据相关
- [[A100]]: 训练硬件
- [[LLaVA]]: 基座模型
- [[LLaVA-NeXT]]: Labeler 模型
- [[Qwen-VL]]: 跨架构泛化实验使用
- [[vLLM]]: 推理加速引擎
- [[CLIP]]: 视觉编码器用于 topic 聚类

---

## 速查卡片

> [!summary] TPR: Topic-level Preference Rewriting
> - **核心**: 通过 topic 级语义单元的选择性替换系统性优化 DPO 训练中的 Reward Gap 配置
> - **方法**: 响应分解 -> topic 聚类 -> intra-topic 自重采样 -> 选择性替换 -> in-context 重写 -> DPO 训练
> - **结果**: ObjectHal-Bench 幻觉降低 93%，多项基准 SOTA，超越先前方法平均 20%
> - **代码**: https://tpr-dpo.github.io

---

*笔记创建时间: 2026-05-19*
