---
title: "Do We Really Need External Tools to Mitigate Hallucinations? SIRA: Shared-Prefix Internal Reconstruction of Attribution"
method_name: "SIRA"
authors: [Tian Qin, Junzhe Chen, Yuqing Shi, Tianshu Zhang, Qiang Ju, Lijie Wen]
year: 2026
venue: arXiv
tags: [hallucination-mitigation, contrastive-decoding, lvlm, vision-language-model, inference-time, training-free, internal-reference]
zotero_collection: LLM幻觉检测
image_source: local
arxiv_html: https://arxiv.org/html/2605.14621
created: 2026-05-18
---

# 论文笔记：Do We Really Need External Tools to Mitigate Hallucinations? SIRA: Shared-Prefix Internal Reconstruction of Attribution

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University, The University of Sydney, Stanford University, Baichuan AI |
| 日期 | May 2026 |
| 项目主页 | — |
| 对比基线 | [[VCD]], [[OPERA]], [[DoLa]], [[ICD|ICT]], [[MaskCD]], [[SID]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.14621) |

---

## 一句话总结

> SIRA 提出一种无需训练的内部[[对比解码]]框架，通过共享前缀 + 晚层分支 + 图像token掩码，在单模型内构造反事实参考进行token级对比，无需外部扰动输入即可持续减少 LVLM [[幻觉抑制|幻觉]]。

---

## 核心贡献

1. **信息流视角重新审视对比解码**: 论证有效参考应保留早期多模态接地的同时限制后期图像token访问，而非在输入空间施加外部扰动。
2. **SIRA 框架**: 提出共享前缀内部归因重构，通过共享早期层、在后期层分叉并对图像token位置施加注意力掩码，构造模型内[[反事实分支|反事实参考]]。
3. **全面实证验证**: 在 POPE、CHAIR、AMBER 上跨 Qwen2.5-VL 和 LLaVA-v1.5 的实验表明 SIRA 持续降低[[幻觉]]，同时保持描述覆盖度，并且推理开销低于两遍对比解码。

---

## 问题背景

### 要解决的问题

大规模[[视觉语言模型|视觉语言模型 (LVLM)]]在视觉证据弱或模糊时，[[语言先验]]主导token预测，产生幻觉——描述图像中不存在的物体、属性或关系。现有的推理时缓解方法通过[[对比解码]]比较原始预测与扰动/退化输入下的参考预测来抑制幻觉，但有两个局限：(1) 通常需要额外的完整前向传播，(2) 输入空间的扰动可能引入离流形伪影。

### 现有方法的局限

现有 LVLM 特定的对比解码方法（[[VCD]]、[[ICD]]、[[MaskCD]]、[[SID]]）在输入空间构造参考（模糊、掩码、指令变化等），导致：
- **额外计算成本**: 需要第二遍完整前向传播
- **离流形伪影**: 输入扰动可能使 logit 差异反映的是扰动引入的不匹配，而非纯粹的视觉归因

### 本文的动机

作者从信息流角度重新思考：多模态 Transformer 的[[阶段化融合]]模式表明，早期层注入视觉信息形成跨模态语义表征，中期层进行跨模态整合，后期层视觉证据已被吸收。因此，一个有效的[[反事实分支]]不应在输入层分离（破坏早期视觉-文本接地），也不应太晚分离（视觉证据已吸收，掩码无效果）。应在共享早期计算的基础上，在后期层通过图像token掩码构造内部参考。

---

## 方法详解

### 模型架构

SIRA 采用 **单模型内部分支** 架构，无需外部模型、验证器或扰动输入：

- **输入**: 统一的多模态 token 序列 $x_{1:S}$，其中 $P_{img} \subseteq \{1,\dots,S\}$ 是图像token位置
- **共享前缀** (层 $0$ 到 $b-1$): 图像和文本token交互，形成对齐的多模态状态，保留 prompt 解释、解码历史、位置结构和早期[[视觉接地]]
- **分叉点** (层 $b = L-K$): 在后期 $K$ 层分叉为两个分支
- **全分支 (Full Branch)**: 标准的多模态推理路径，继续访问图像token
- **[[反事实分支]] (Counterfactual Branch)**: 从层 $b$ 起，对图像token位置的注意力被掩码，仅保留[[语言先验]]驱动的预测
- **输出**: [[对比解码|内部对比 logits]] $z_t^{cd}(v) = (1+\alpha)z_t^{full}(v) - \alpha z_t^{cf}(v)$
- **推理开销比**: $1 + K/L$（如中间分叉约 $1.5\times$，而 VCD 需要 $2\times$）

### 核心模块

#### 模块1: 共享前缀内部分支设计

**设计动机**: 利用[[多模态Transformer]]的[[阶段化融合]]特性，使两个分支从相同的图像条件语义状态出发，避免输入空间扰动导致的离流形伪影。

**具体实现**:
- 层 $0$ 到 $b-1$ 仅计算一次，两个分支共享 `F_{0:b-1}` 的输出 $h_t^b$
- 边界层 $b$ 放置在主要整合阶段之前（Qwen2.5-VL: $b=14$, LLaVA-v1.5: $b=16$）
- 分支分离仅作用在最后 $K$ 层：$h_t^{full,L} = F_{b:L-1}(h_t^b; M_t^{causal})$，$h_t^{cf,L} = F_{b:L-1}(h_t^b; M_t^{cf})$
- 共享前缀使 logit 差异 $z_t^{full} - z_t^{cf}$ 反映的是后期视觉贡献，而非表征不匹配

#### 模块2: 反事实分支的掩码注意力构造

**设计动机**: 通过[[注意力掩码]]在后期层阻断图像token的访问路径，构造一个[[语言先验]]主导的内部参考，保持在模型自身的token流形上。

**具体实现**:
- 定义反事实有效性掩码 $M_t^{cf}$：仅当 $M_t^{causal}(q,k)=1$ 且 $q \notin P_{img}$ 且 $k \notin P_{img}$ 时有效
- 图像token既不能作为 key 被其他 query 读取，也不能作为 query 产生写入文本流的表示
- 在 prefill 阶段：$q \notin P_{img}$ 条件活跃，移除图像token作为 query 行
- 在自回归解码阶段：每个新 query 是生成的文本token，$q \notin P_{img}$ 自动满足，仅 $k \notin P_{img}$ 条件活跃
- 掩码带来的残差漂移在实证上很小（Section 5.2）

#### 模块3: 内部对比解码规则

**设计动机**: 在 token 级别对比两个分支的 logits，抑制即使没有后期视觉访问也保持高分的token（语言先验驱动），增强依赖完整视觉通路的token。

**具体实现**:
- 两个分支共享最终的 Normalization 和输出投影 $W_{out}$
- 对比 $\Delta_t(v) = z_t^{full}(v) - z_t^{cf}(v)$ 测量每个 token 的增量视觉贡献
- $z_t^{cd}(v) = z_t^{full}(v) + \alpha \Delta_t(v) = (1+\alpha)z_t^{full}(v) - \alpha z_t^{cf}(v)$
- $\alpha = 0.5$ 为最优对比强度；$\alpha=0$ 退化为标准解码
- 最终 token 选择：$y_t = \arg\max_{v\in\mathcal{V}} z_t^{cd}(v)$

---

## 关键公式

### 公式1: [[反事实分支|双分支隐藏状态]]

$$
h_t^{full,L} = F_{b:L-1}(h_t^b; M_t^{causal}), \quad h_t^{cf,L} = F_{b:L-1}(h_t^b; M_t^{cf})
$$

**含义**: 两个分支从相同的边界状态 $h_t^b$ 出发，仅在注意力掩码上不同。全分支使用标准因果掩码，反事实分支额外屏蔽图像token位置。

**符号说明**:
- $h_t^b = F_{0:b-1}(u_t)$: 共享前缀层输出的边界表示
- $F_{b:L-1}$: 后边界 $K$ 层
- $M_t^{causal}$: 标准因果+填充掩码
- $M_t^{cf}$: 反事实掩码（屏蔽图像token）

### 公式2: [[注意力掩码|反事实有效性掩码]]

$$
M_t^{cf}(q, k) = \begin{cases}
1, & \text{if } M_t^{causal}(q, k) = 1, \; q \notin P_{img}, \; k \notin P_{img}, \\
0, & \text{otherwise}.
\end{cases}
$$

**含义**: 定义反事实分支中哪些注意力连接有效。图像token ($P_{img}$) 既不能作为 query 也不能作为 key，从而完全阻断后期视觉信息流。

**符号说明**:
- $q, k$: query 和 key 位置索引
- $P_{img}$: 图像token在原始prompt中的位置集合
- 有效条目接收 $0$ (无惩罚)，无效条目接收大负值 (屏蔽)

### 公式3: [[对比解码|分支Logits]]

$$
z_t^{full} = W_{out} \operatorname{Norm}(h_t^{full,L})_{last}, \quad z_t^{cf} = W_{out} \operatorname{Norm}(h_t^{cf,L})_{last}
$$

**含义**: 从两个分支的最后一层隐藏状态，通过共享的输出投影和归一化得到 logits。下标 $last$ 表示取序列最后位置的表示，驱动机器下一token预测。

**符号说明**:
- $\operatorname{Norm}(\cdot)$: 模型自身的最终归一化层
- $W_{out}$: 输出投影矩阵（两个分支共享）
- $(\cdot)_{last}$: 取序列末端位置的向量

### 公式4: [[内部对比解码|内部对比解码规则]]

$$
z_t^{cd}(v) = z_t^{full}(v) + \alpha \Delta_t(v) = (1 + \alpha)z_t^{full}(v) - \alpha z_t^{cf}(v)
$$

**含义**: 在 logit 空间进行近似基线减法。$z_t^{cf}$ 作为语言先验主导的参考，从 $z_t^{full}$ 减去它得到每个 token 的增量视觉贡献代理，实时抵消语言模板偏置。

**符号说明**:
- $\alpha \geq 0$: 对比强度超参数，最优值为 $0.5$
- $\Delta_t(v) = z_t^{full}(v) - z_t^{cf}(v)$: 内部对比项
- $\alpha=0$ 退化为标准解码

---

## 关键图表

### Figure 1: Comparison between SIRA and prior methods

![[SIRA_fig1_comparison.png]]

**说明**: 左列展示现有方法（VCD 等）通过外部输入扰动构造参考，需要两次完整前向传播且可能引入离流形伪影。右列展示 SIRA 通过共享前缀 + 内部分支 + 后期视觉访问移除，在单一模型内进行对比。

### Figure 2: SIRA Overview

![[SIRA_fig2_overview.png]]

**说明**: SIRA 完整架构。输入文本和图像token经过共享前缀层 ($F_{0:b-1}$) 形成对齐的多模态状态后，分叉为全分支和反事实分支。反事实分支在后期 $K$ 层被屏蔽图像token访问。两个分支的 logits 进行对比融合产生最终输出。右下角展示 token 级 logit 对比示例。

### Figure 3: Effect of Split Boundary and Contrast Strength

![[SIRA_fig3_ablation.png]]

**说明**: 左两图展示分叉边界层 $b$ 对 POPE 准确率的影响。Qwen2.5-VL ($L=28$) 在 $b=14$ 处最优，LLaVA-v1.5 ($L=32$) 在 $b=16$ 处最优。过早分叉破坏多模态接地，过晚分叉导致掩码无效。右两图展示对比强度 $\alpha$ 的影响，两个模型均在 $\alpha=0.5$ 处达到峰值。

### Figure 4: Contrastive Reference Analysis

<!-- 四子图: (a) 层级漂移 (b) 输出KL散度 (c) 阶段漂移 (d) KL vs 后期漂移定位 -->

**说明**: (a) 层级余弦相似度漂移：SIRA-CF 的漂移在分叉前几乎为零，分叉后增长；高斯模糊和patch shuffle在全网络漂移。(b) 输出 [[KL散度]]：SIRA-CF (0.012) 远低于 shuffle (0.172) 和 blur (0.231)。(c) 阶段漂移统计：SIRA-CF 的漂移集中在后期。(d) 参考定位：SIRA-CF 同时具有低输出KL和高后期漂移，表明它是更好的内部参考。

### Figure 5: Case Study and Edge-Case Analysis

![[SIRA_fig5_casestudy.png]]

**说明**: 使用 Qwen2.5-VL 在 AMBER 上的案例分析。上排展示对真实物体 (GT Words) 的 logit 比较：全分支 > 反事实分支，SIRA 进一步放大优势。下排展示对幻觉词 (Hallu Words) 的 logit 比较：反事实分支与全分支相当或更高，SIRA 将其压至最低。右侧边缘案例展示 SIRA 移除了"house"等幻觉物体，但引入了"black-and-white"属性归因错误（由阴影引起）。

### Figure 6: Additional Case Studies

<!-- 8 个 AMBER 案例，每个包含输入图像、GT Words 柱状图和 Hallu Words 柱状图 -->

**说明**: 附录中的 8 个额外 AMBER 案例，三个一致模式：(i) GT Words 上全分支 > 反事实分支，SIRA 进一步提升；(ii) Hallu Words 上反事实分支与全分支相当，SIRA 有效压制；(iii) GT Words 中全分支≈反事实分支时，SIRA 几乎不改变，实现自门控效应。

### Table 1: POPE Results

| 模型         | 数据集     | 设置          | Base  | DoLa  | OPERA | VCD   | ICT   | MaskCD | SID   | **SIRA**  |
| ---------- | ------- | ----------- | ----- | ----- | ----- | ----- | ----- | ------ | ----- | --------- |
|            |         | Random      | 85.23 | 86.53 | 87.31 | 88.63 | 87.53 | 90.05  | 87.90 | **91.13** |
|            | COCO    | Popular     | 84.53 | 85.93 | 87.44 | 87.12 | 86.76 | 88.65  | 86.90 | **89.70** |
|            |         | Adversarial | 83.37 | 83.47 | 84.78 | 84.26 | 86.16 | 86.05  | 83.83 | **87.83** |
|            |         | Random      | 86.40 | 87.40 | 88.19 | 89.22 | 88.96 | 90.55  | 88.49 | **92.10** |
| Qwen2.5-VL | A-OKVQA | Popular     | 85.77 | 87.53 | 87.91 | 87.85 | 87.43 | 89.05  | 87.62 | **89.67** |
|            |         | Adversarial | 80.37 | 81.83 | 80.82 | 81.27 | 83.60 | 82.75  | 80.84 | **83.63** |
|            |         | Random      | 85.10 | 87.13 | 86.02 | 85.59 | 88.96 | 89.25  | 84.86 | **92.23** |
|            | GQA     | Popular     | 80.87 | 82.53 | 81.97 | 81.83 | 86.43 | 86.35  | 81.61 | **88.37** |
|            |         | Adversarial | 78.77 | 82.00 | 80.24 | 80.01 | 84.10 | 83.25  | 79.58 | **84.27** |
|            |         | Random      | 83.29 | 85.97 | 89.20 | 87.73 | 89.18 | 88.55  | 88.05 | **89.27** |
|            | COCO    | Popular     | 81.88 | 82.93 | 86.64 | 85.38 | 86.07 | 86.25  | 85.83 | **86.83** |
|            |         | Adversarial | 78.96 | 77.17 | 81.24 | 80.88 | 83.30 | 81.80  | 81.41 | **84.07** |
|            |         | Random      | 83.45 | 83.23 | 88.03 | 86.15 | 89.00 | 87.45  | 86.47 | **89.13** |
| LLaVA-v1.5 | A-OKVQA | Popular     | 79.90 | 76.47 | 83.22 | 81.85 | 83.40 | 83.05  | 82.48 | **83.77** |
|            |         | Adversarial | 74.04 | 68.03 | 73.82 | 74.97 | 75.56 | 75.90  | 75.50 | **77.30** |
|            |         | Random      | 83.73 | 83.70 | 88.13 | 86.65 | 89.23 | 87.85  | 86.97 | **89.47** |
|            | GQA     | Popular     | 78.17 | 74.03 | 79.27 | 80.73 | 80.86 | 82.05  | 81.36 | **82.93** |
|            |         | Adversarial | 75.08 | 68.73 | 75.00 | 76.09 | 77.40 | 77.10  | 76.62 | **79.60** |

**说明**: SIRA 在所有 36 个 (2模型 x 3数据集 x 3设置 x 2指标) 指标上取得最佳。Qwen2.5-VL 平均提升 +5.4 Acc / +6.7 F1，LLaVA-v1.5 提升 +4.9 Acc / +6.0 F1。在语言先验压力更大的 GQA 上优势尤为明显。

### Table 2: CHAIR and AMBER Results

| 模型 | 方法 | $CS_{64}\downarrow$ | $CI_{64}\downarrow$ | $CS_{512}\downarrow$ | $CI_{512}\downarrow$ | CHAIR $\downarrow$ | Cover $\uparrow$ | Hal $\downarrow$ | Cog $\downarrow$ |
|------|------|-----|-----|------|------|------|------|-----|-----|
| | Baseline | 20.8 | 17.8 | 35.5 | 24.5 | 5.3 | 47.2 | 23.6 | 1.9 |
| | SID | 18.1 | 15.6 | 33.0 | 22.8 | 4.9 | 46.3 | 21.5 | 1.4 |
| | MaskCD | 16.9 | 14.6 | 31.8 | 22.0 | 4.8 | 47.4 | 22.0 | 1.5 |
| | DoLa | 20.1 | 17.3 | 35.9 | 24.6 | 5.5 | 46.8 | 25.1 | 2.2 |
| Qwen2.5-VL | OPERA | 18.6 | 15.9 | 32.9 | 22.8 | 5.1 | 46.2 | 21.6 | 1.7 |
| | VCD | 19.3 | 16.7 | 33.9 | 23.3 | 4.7 | 45.9 | 20.5 | 1.6 |
| | ICT | 17.0 | 14.5 | 31.9 | 22.1 | 4.9 | 45.1 | 21.8 | 1.2 |
| | **SIRA** | **16.4** | **14.0** | **31.2** | **21.7** | **4.6** | **47.8** | **20.2** | **0.9** |
| | Baseline | 31.5 | 20.3 | 50.6 | 30.4 | 7.3 | 50.7 | 33.7 | 3.8 |
| | SID | 26.0 | 17.2 | 46.2 | 27.6 | 6.1 | 50.2 | 27.5 | 2.8 |
| | MaskCD | 24.0 | 15.7 | 44.2 | 26.4 | 6.6 | 52.4 | 31.6 | 3.0 |
| | DoLa | 29.1 | 21.5 | 52.4 | 30.2 | 7.6 | 51.6 | 36.0 | 4.0 |
| LLaVA-v1.5 | OPERA | 26.7 | 17.9 | 46.9 | 27.8 | 7.3 | 49.6 | 32.0 | 3.5 |
| | VCD | 28.3 | 19.0 | 48.7 | 28.9 | 6.8 | 49.6 | 30.4 | 3.5 |
| | ICT | 24.2 | 15.9 | 44.5 | 26.6 | 5.2 | 51.3 | 23.1 | 2.1 |
| | **SIRA** | **23.1** | **15.1** | **43.6** | **26.0** | **4.8** | **53.7** | **21.6** | **1.8** |

**说明**: CHAIR：句子级 ($CS$) 和实例级 ($CI$) 幻觉率，两个生成长度。AMBER：CHAIR/Hal/Cog 越低越好，Cover 越高越好。SIRA 在降低幻觉的同时保持或提升覆盖度（LLaVA-v1.5 Cover 从 50.7 升至 53.7），与 OPERA/VCD 以牺牲覆盖度为代价降幻觉形成对比。Cog（认知幻觉）锐减 50% 以上，与设计目标一致。

### Table 3: Property Comparison of Methods

| Method | LVLM-specific | Preserves input | No external tools/models | Low decoding overhead |
|--------|:------------:|:--------------:|:------------------------:|:--------------------:|
| DoLa | x | v | v | v |
| VCD | v | x | v | x |
| OPERA | v | v | v | x |
| ICT | v | v | x | v |
| MaskCD | v | v | v | x |
| SID | v | v | v | x |
| **SIRA** | **v** | **v** | **v** | **v** |

**说明**: SIRA 是唯一同时满足四项需求的对比解码方法：LVLM特定、保留原始输入、无外部工具/模型、低解码开销。

### Table 4: Inference Efficiency

| 方法 | 32-Token (ms) | Rel. | 64-Token (ms) | Rel. | 128-Token (ms) | Rel. |
|------|:---------:|:----:|:---------:|:----:|:----------:|:----:|
| Qwen2.5-VL (Base) | 587 | x1.0 | 1000 | x1.0 | 1868 | x1.0 |
| VCD | 1205 | x2.1 | 2063 | x2.1 | 3847 | x2.1 |
| OPERA | 3100 | x5.3 | 7786 | x5.8 | 10966 | x5.9 |
| **SIRA** | **838** | **x1.4** | **1496** | **x1.5** | **2781** | **x1.5** |

**说明**: SIRA 的推理开销约为标准解码的 1.4-1.5 倍，远低于 VCD (2.1x) 和 OPERA (5-6x)，与理论比例 $1 + K/L$ 一致。

### Table 5 & 6: POPE Yes-Rate Decomposition

<!-- 完整的每个 (backbone, dataset, split, method) 的 Acc, F1, Prec, Recall, FPR, Yes-rate -->

**说明**: 附录中的完整 POPE 分解表（Table 5: Qwen2.5-VL, Table 6: LLaVA-v1.5）。关键发现：SIRA 的 Yes-rate 在所有 split 上均上升，但即使基线 Yes-rate 已超 50%（偏向 Yes）的 split 上准确率仍上升，证明增益来自选择性修正而非简单的 Yes/No 再平衡。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[POPE]] | COCO/A-OKVQA/GQA 子集 | 均衡 Yes/No 问题，三种负采样策略 (Random/Popular/Adversarial) | 判别式物体幻觉评估 |
| [[CHAIR]] | COCO 2014 val (500张) | 句子级 $CS$ 和实例级 $CI$ 幻觉率 | 自由形式字幕物体幻觉 |
| [[AMBER]] | 生成任务 | CHAIR/Hal/Cog/Cover 四维指标 | 多维字幕幻觉 + 覆盖度 |

### 实现细节

- **Backbone**: [[LLaVA|LLaVA-v1.5-7B]] ($L=32$), [[Qwen2.5-VL]]-7B ($L=28$)
- **超参数**: $\alpha=0.5$, Qwen2.5-VL: $K=14$ ($b=14$), LLaVA-v1.5: $K=16$ ($b=16$)
- **所有数据集固定超参数**，无需 per-dataset 调参
- **硬件**: 3x NVIDIA A100 40GB
- **对比精度**: Logit 对比在 float32 中计算，避免 bfloat16 半精度导致的尾部 rank reversal

### 可视化结果

1. **分叉边界分析** (Figure 3): 中间分叉最优，早/晚分叉均导致性能下降，支持[[阶段化融合]]视图
2. **参考对齐分析** (Figure 4): SIRA-CF 的层级漂移仅在分叉后出现，输出 [[KL散度]] 极低 (0.012)，同时保持足够的后期漂移用于有效对比
3. **Token 级效应** (Figure 5-6): 真实物体 token 被增强，幻觉 token 被压制，已正确的 token 不受影响（自门控）

---

## 批判性思考

### 优点
1. **无需训练的即插即用**: SIRA 不需要训练、外部验证器或扰动输入，仅需白盒推理访问，部署成本极低
2. **理论基础清晰**: 从多模态 Transformer 的阶段化融合特性出发，设计了合理的分叉策略和掩码机制，每个设计选择都有消融验证
3. **幻觉-覆盖度双赢**: 不同于 OPERA/VCD 以牺牲描述覆盖度为代价降低幻觉，SIRA 同时减少幻觉和保持/提升覆盖度
4. **效率优势显著**: 1.4-1.5x 推理开销远优于 VCD (2.1x) 和 OPERA (5-6x)，实际部署可行
5. **全面的实验评估**: 36 个 POPE 指标全胜，覆盖判别式和生成式场景，附录提供完整的 Yes-rate 分解

### 局限性
1. **仍需每步额外计算**: 虽远低于两遍对比解码，但 1.4-1.5x 开销非零，对实时性要求极高的场景仍有影响
2. **白盒访问限制**: 需要访问隐藏状态、注意力掩码和 KV Cache 状态，无法用于黑盒 API 模型（如 GPT-4V、Gemini）
3. **属性级幻觉未完全解决**: Figure 5 的边缘案例显示，SIRA 移除物体幻觉后可能引入属性归因错误（如光照引起的"black-and-white"描述）
4. **仅评估 7B 模型**: 未在更大规模 LVLM 上验证可扩展性

### 潜在改进方向
1. **自适应分叉策略**: 根据输入复杂度动态选择边界层 $b$ 和对比强度 $\alpha$，而非固定超参数
2. **多头/多层对比**: 探索在多个层进行对比或对不同注意力头施加不同程度掩码
3. **结合训练方法**: 在 SIRA 框架下微调模型以更好地利用内部对比信号
4. **更大模型验证**: 在 13B/34B/72B 等更大 LVLM 上验证可扩展性

### 可复现性评估
- [ ] 代码开源 (尚未，论文标注 Preprint)
- [ ] 预训练模型 (使用开源模型 LLaVA-v1.5 / Qwen2.5-VL)
- [x] 训练细节完整 (附录 A 包含完整实现细节，附录 B 包含伪代码)
- [x] 数据集可获取 (POPE/CHAIR/AMBER 均为公开数据集)

---

## 关联笔记

### 基于
- [[对比解码|Contrastive Decoding]]: SIRA 的核心思想来源，从 logit 空间对比不同条件消除偏置
- [[阶段化融合|Staged Fusion]]: 多模态 Transformer 的渐进式信息整合模式，是分叉设计的理论基础

### 对比
- [[VCD]]: 输入空间扰动构造参考，需两遍前向传播，SIRA 避免其离流形伪影和额外开销
- [[OPERA]]: 基于注意力惩罚和回溯分配，5-6x 推理开销，SIRA 更高效
- [[DoLa]]: 对比不同层 logits 但不针对 LVLM 设计，SIRA 专门利用多模态信息流
- [[ICD|ICT]]: 使用图像-物体跨层可信干预，但需要外部物体级信号
- [[MaskCD]]: 掩码图像头构造对比信号，仍需要额外处理
- [[SID]]: 自省解码构造额外参考，增加推理开销

### 方法相关
- [[内部对比解码|Internal Contrastive Decoding]]: SIRA 的核心解码机制
- [[共享前缀|Shared Prefix]]: 保证两个分支对齐的关键设计
- [[反事实分支|Counterfactual Branch]]: 通过注意力掩码构造语言先验参考
- [[注意力掩码|Attention Mask]]: 反事实分支中阻断视觉通路的具体手段

### 硬件/数据相关
- [[POPE]]: 物体幻觉检测的判别式基准
- [[CHAIR]]: 字幕级物体幻觉评估
- [[AMBER]]: 多维幻觉评估（含认知幻觉）

---

## 速查卡片

> [!summary] SIRA: Shared-Prefix Internal Reconstruction of Attribution
> - **核心**: 无需训练的内部对比解码框架，在模型内部构造反事实参考，通过共享前缀 + 晚层视觉掩码抑制 LVLM 幻觉
> - **方法**: 共享早期层的多模态接地，在后期层分叉并通过注意力掩码阻断图像token访问，对比两个分支的 logits 进行 token 级先验抵消
> - **结果**: POPE 36/36 指标全胜，CHAIR/AMBER 幻觉持续降低且覆盖度保持/提升，推理开销仅 1.4-1.5x
> - **代码**: 未开源 (Preprint)

---

*笔记创建时间: 2026-05-18*
