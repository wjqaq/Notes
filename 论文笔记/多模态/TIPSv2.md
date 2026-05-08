---
title: "TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment"
method_name: "TIPSv2"
authors: [Bingyi Cao, Koert Chen, Kevis-Kokitsi Maninis, Kaifeng Chen, Arjun Karpur, Ye Xia, Sahil Dua, Tanmaya Dabral, Guangxing Han, Bohyung Han, Joshua Ainslie, Alex Bewley, Mithun Jacob, René Wagner, Washington Ramos, Krzysztof Choromanski, Mojtaba Seyedhosseini, Howard Zhou, André Araujo]
year: 2026
venue: CVPR 2026
tags: [vision-language-pretraining, patch-text-alignment, contrastive-learning, masked-image-modeling, knowledge-distillation, zero-shot-segmentation, image-text-retrieval, dense-alignment]
zotero_collection: 多模态
image_source: mixed
arxiv_html: https://arxiv.org/html/2604.12012v1
created: 2026-05-08
---

# 论文笔记：TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Google DeepMind |
| 日期 | April 2026 |
| 项目主页 | [gdm-tipsv2.github.io](https://gdm-tipsv2.github.io/) |
| 对比基线 | [[CLIP]], [[SigLIP]], [[DINOv2]], [[TIPS]], [[DINOv3]] |
| 链接 | [arXiv](https://arxiv.org/abs/2604.12012) / [Code](https://github.com/google-deepmind/tips) |

---

## 一句话总结

> 通过 iBOT++（可见 token 也参与损失）、head-only EMA 和多粒度文本增强三个改进，将视觉-语言预训练的 patch-text 密集对齐能力大幅提升，蒸馏出的学生模型对齐甚至超越教师模型。

---

## 核心贡献

1. **iBOT++**: 对 iBOT 的升级，使未遮挡的 patch token 也直接参与损失计算，大幅提升 zero-shot 语义分割能力（ADE150 +14.1 mIoU）
2. **Head-only EMA**: 仅对投影头做 EMA 更新而非完整模型，减少 42% 训练参数，性能不降
3. **多粒度文本采样**: 利用 PaliGemma + Gemini Flash 生成不同粒度的合成描述，交替训练提升图文对齐
4. **关键发现**: 蒸馏过程中移除 mask + 随机初始化学生编码器，可使学生的 patch-text 对齐**超越教师模型**
5. **TIPSv2 模型族**: B/14、L/14、SO/14、g/14 四个规模，在 9 个任务 20 个数据集上验证

---

## 问题背景

### 要解决的问题

视觉-语言模型中**密集 patch 表示与文本嵌入的对齐**仍然困难。旗舰大模型在 dense 对齐任务上往往不如小模型。

### 现有方法的局限

- **[[CLIP]]**: 仅做全局图文对比学习，缺乏 dense-level 对齐，zero-shot 分割能力弱（PC60 仅 4.3）
- **[[DINOv2]]**: 擅长密集视觉任务但缺乏文本对齐能力
- **PE-core ([[Perception Encoder]])**: 图文对齐强但在密集任务上表现不足
- **[[TIPS]]**: 结合对比学习和 patch-level 蒸馏，但 iBOT 仅对遮挡 token 监督，大量可见 token 未被利用
- **[[SigLIP2]]**: 大规模模型在 patch-text alignment 上反而退步（Table 14：g/16 < B/16）

### 本文的动机

通过分析 TIPS 的蒸馏过程发现：移除 mask + 随机初始化学生编码器能释放强大的 patch-text 对齐信号。作者将这一发现系统化为 iBOT++ 方法，并结合训练效率优化（head-only EMA）和数据增强（多粒度描述）构建 TIPSv2。

---

## 方法详解

### 模型架构

TIPSv2 采用 **双塔架构**（[[Vision Transformer]] 图像编码器 + Transformer 文本编码器），核心训练目标由三部分组成：

- **图像编码器**: ViT (B/14, L/14, SO/14, g/14)，输出全局 `[CLS]` 嵌入 `e^g` 和 patch 嵌入 `{e_1, ..., e_N}`
- **文本编码器**: 标准 Transformer，输出文本嵌入 `e^t`
- **投影头**: 将图像/文本嵌入映射到共享空间，使用 [[Exponential Moving Average|EMA]] 中心和锐化防止坍塌
- **双 CLS 设计**: 两个独立的 `[CLS]` token——一个对应网络 alt-text，一个对应 PaliGemma/Gemini 合成描述
- **总损失**: $\mathcal{L} = \mathcal{L}_{CLIP} + \alpha\mathcal{L}_{DINO} + \beta\mathcal{L}_{iBOT}$，其中 $\alpha=1.0, \beta=2.0$

![Figure 3: TIPSv2 Pretraining Overview](https://arxiv.org/html/2604.12012v1/x2.png)

### 核心模块

#### 模块1: iBOT++ — 增强的 Patch-Level 蒸馏

**设计动机**: 原始 [[iBOT]] 仅对被 mask 的 patch token 计算损失，约 75% 的可见 token 未被监督。作者在蒸馏实验中发现移除 mask 能大幅提升 alignment。

**具体实现**:

原始 iBOT 损失:
- 对学生视图的遮挡 patch token，用教师模型对应位置的输出作为软目标
- 使用遮蔽指示器 $m_i$ 选择仅对遮挡 token 计算损失

iBOT++ 损失:
- **移除遮蔽指示器**，对**所有 patch token**（遮挡和未遮挡）都计算损失
- 未遮挡 token 的损失持续下降（Fig 2），表明它们被成功"锚定"到教师模型的表示

$$\mathcal{L}_{iBOT++} = -\sum_{i=1}^{N} h_t(f_t(I)_i)^T \log h_s(f_s(I_{mask})_i)$$

与 iBOT 的唯一区别：去掉了 $m_i$ 乘子，使得该损失作用于所有 $N$ 个 patch token。

#### 模块2: 蒸馏增强（Bridging Pretraining and Distillation）

**核心发现**: TIPS ViT-L 学生（蒸馏自 TIPS ViT-g 教师）在 zero-shot 分割上远超教师（Table 1）。进一步消融揭示两个关键因素:

1. **移除 patch-level mask（mask ratio = 0.0）** — 对所有 patch token 施加监督是蒸馏成功的关键
2. **学生随机初始化** — 使用预训练权重初始化会消除蒸馏收益；文本编码器是否固定影响较小
3. **冻结 vs 训练文本塔** — 差异不大，关键在视觉编码器的初始化方式

> "supervising all patch tokens, rather than just masked ones, is crucial for alignment during distillation"

#### 模块3: Head-only EMA

**设计动机**: 标准 SSL 方法用全模型 EMA 教师防止坍塌，但在 TIPSv2 中 [[InfoNCE Loss|CLIP 损失]] 已防止编码器坍塌。

**具体实现**:
- 设定 $f_t = f_s$（学生和教师共享视觉编码器权重）
- 仅对投影头 $h$ 做 [[Exponential Moving Average|EMA]] 更新：$h_t \leftarrow \lambda h_t + (1-\lambda) h_s$
- ViT-B 上减少 42% 训练参数
- 完全移除 EMA 会导致严重训练不稳定

![[TIPSv2_fig2.png|600]]

#### 模块4: 多粒度文本描述采样

**设计动机**: Web alt-text 和 PaliGemma 的描述存在信息缺失（Fig 4: 遗漏姿态、卡通属性、季节背景）。[[Gemini Flash]] 生成的描述过于全面反而弱化对比学习难度。

**具体实现**:
- 用 Gemini 1.5 Flash 基于 (图像, alt-text, PaliGemma 描述) 生成丰富的合成描述
- **随机交替**使用 PaliGemma 和 Gemini 描述（而非仅用更长的 Gemini 描述）
- 配合双 CLS 设计：一个 CLS 对应 alt-text，另一个对应合成描述
- 过长描述会"轻视化"对比损失，交替策略保留对比学习难度

![Figure 4: Captions at Different Granularities](https://arxiv.org/html/2604.12012v1/x3.png)

---

## 关键公式

### 公式1: [[InfoNCE Loss|CLIP 对比损失]]

$$
\mathcal{L}_{CLIP} = -\frac{1}{2B}\sum_{k=1}^{B} \left[ \log\frac{\exp(s(I_k, T_k)/\tau)}{\sum_{j}\exp(s(I_k, T_j)/\tau)} + \log\frac{\exp(s(T_k, I_k)/\tau)}{\sum_{j}\exp(s(T_k, I_j)/\tau)} \right]
$$

**含义**: 双向 InfoNCE，对齐全局图像嵌入 $e^g$ 与文本嵌入 $e^t$。TIPSv2 使用两个独立 CLS token 分别对应 web alt-text 和合成描述。

**符号说明**:
- $B$: batch size
- $s(\cdot,\cdot)$: cosine similarity
- $\tau$: 可学习温度参数

### 公式2: [[DINO|DINO 自蒸馏损失]]

$$
\mathcal{L}_{DINO} = -\sum_{c} P_t(c|I_g) \log P_s(c|I_l)
$$

**含义**: 全局级自蒸馏，教师（全局视图）指导学生（M 个局部裁剪）的 `[CLS]` token 分布。

**符号说明**:
- $I_g$: 全局视图（高分辨率 crop）
- $I_l$: 局部视图（低分辨率小 crop）
- $P_t, P_s$: 教师/学生的 softmax 概率分布（带温度 $\tau_t, \tau_s$）
- 使用 [[Exponential Moving Average|EMA]] 中心的教师防止坍塌

### 公式3: [[iBOT|iBOT 遮蔽图像建模]]

$$
\mathcal{L}_{iBOT} = -\sum_{i=1}^{N} m_i \cdot h_t(f_t(I)_i)^T \log h_s(f_s(I_{mask})_i)
$$

**含义**: Patch 级 MIM，学生处理遮挡图像，教师处理完整图像，仅对被遮挡 patch（$m_i=1$）计算损失。

**符号说明**:
- $f_s, f_t$: 学生/教师图像编码器
- $h_s, h_t$: 学生/教师投影头
- $m_i \in \{0, 1\}$: 遮挡指示器（原始 iBOT 仅对遮挡 token 计算损失）
- $I_{mask}$: 经过随机遮挡的输入图像

### 公式4: [[iBOT++]]

$$
\mathcal{L}_{iBOT++} = -\sum_{i=1}^{N} h_t(f_t(I)_i)^T \log h_s(f_s(I_{mask})_i)
$$

**含义**: iBOT++ 与 iBOT 的唯一区别是**去掉了 $m_i$**，使损失作用于所有 $N$ 个 patch token——遮挡和未遮挡的都参与。未遮挡 token 的 patch-level 损失在 iBOT++ 下持续下降（Fig 2），实现了对教师表示的成功"锚定"。

**符号说明**:
- 符号同上，区别仅在于无 $m_i$ 选择机制
- 最优遮挡比在 iBOT++ 中仍为 75%（Table 12）
- 仅在蒸馏阶段移除 mask 有效（因教师已提供强局部语义理解）；从头预训练时掩码仍是必需的

### 公式5: [[Exponential Moving Average|EMA 更新规则]]

$$
h_t \leftarrow \lambda h_t + (1-\lambda) h_s
$$

**含义**: Head-only EMA — 仅对投影头做动量更新，视觉编码器权重学生/教师共享。保持训练稳定性的同时减少 42% 参数量。

---

## 关键图表

### Figure 1: iBOT vs iBOT++ 对比

![Figure 1: iBOT vs iBOT++](https://arxiv.org/html/2604.12012v1/x1.png)

**说明**: 对比原始 iBOT（仅对被 mask patch 监督）和 iBOT++（对所有 patch 监督）的机制差异。iBOT++ 使可见 token 也直接贡献损失，显著提升 zero-shot 分割能力。

### Figure 2: 可见 Token 的 Patch-Level 损失下降

![Figure 2: Patch Loss for Visible Tokens](https://arxiv.org/html/2604.12012v1/figures/patch_loss.png)

**说明**: iBOT++ 下可见 token 的 patch 级损失持续下降，表明它们被成功"锚定"到教师表示。这在原始 iBOT 中不会发生。

### Figure 3: TIPSv2 预训练总览

![Figure 3: TIPSv2 Overview](https://arxiv.org/html/2604.12012v1/x2.png)

**说明**: TIPSv2 完整预训练流程：图像编码器（ViT）+ 文本编码器，三部分损失（CLIP + DINO + iBOT++），双 CLS 设计，head-only EMA，多粒度文本采样。

### Figure 4: 不同粒度图像描述示例

![Figure 4: Caption Granularities](https://arxiv.org/html/2604.12012v1/x3.png)

**说明**: 同一图像在不同粒度下的描述——alt-text 过于简短，PaliGemma 遗漏姿态和背景，Gemini 描述最全面但可能过于详尽而弱化对比学习难度。

### Figure 5: PCA 特征可视化（ViT-g, 1372/1568 分辨率）

![Figure 5: PCA Maps](https://arxiv.org/html/2604.12012v1/images/supp/tipsv2/vit_g/hike.png)

**说明**: 四组场景（dadaocheng/cph/hike/angus）下 TIPS vs SigLIP2 vs TIPSv2 的 PCA 特征图对比。TIPSv2 产生更平滑的特征图，物体边界清晰（背包、人物、登山杖等被精细分离）。

### Figure 6: Zero-Shot 分割可视化

![Figure 6: Zero-shot Segmentation](https://arxiv.org/html/2604.12012v1/images/dog_rug_v2.png)

**说明**: 三组场景（dog_rug/birds/bus）下 TIPSv2 vs TIPS vs SigLIP2 的 zero-shot 分割对比。TIPSv2 的分割 mask 显著更干净完整。

### Figure 7: PCA 特征图 ViT-L 对比（附录）

![Figure 7: PCA ViT-L](https://arxiv.org/html/2604.12012v1/images/supp/tipsv2/vit_l/dadaocheng.png)

**说明**: ViT-L 规模下 DINOv2 vs DINOv3 vs TIPSv2 的 PCA 特征图。DINOv3 特征更平滑，但 TIPSv2 特征语义聚焦更强。

### Figure 8: PCA 特征图 ViT-g/7B 对比（附录）

![Figure 8: PCA ViT-g](https://arxiv.org/html/2604.12012v1/images/supp/tipsv2/vit_g/dadaocheng.png)

**说明**: 最大规模下的 PCA 对比。TIPSv2 捕获更精确的语义细节。

### Figure 9: iBOT++ vs iBOT 定性对比（附录）

![Figure 9: iBOT++ vs iBOT](https://arxiv.org/html/2604.12012v1/images/supp/bicycle_ibotplus.png)

**说明**: 6 组场景下 iBOT++ 与 iBOT 的 zero-shot 分割对比。iBOT++ 产生显著更干净的分割 mask。

### Figure 10: TIPSv2 vs 竞品胜率（附录）

![Figure 10: Win Rates](https://arxiv.org/html/2604.12012v1/x4.png)

**说明**: TIPSv2 与各竞品在共享评估项上的 head-to-head 胜率。TIPSv2 对 PE-core、DINOv2、TIPS、CLIP、SigLIP2、OpenCLIP、SILC、DINOv3 均取得大多数胜利。

### Table 1: 蒸馏 vs 标准预训练 — 意外发现

| Model | PC59 | PC60 | VOC21 | ADE150 |
|-------|------|------|-------|--------|
| TIPS ViT-L (标准预训练) | 33.5 | 30.4 | 30.5 | 20.8 |
| TIPS ViT-g (教师，零样本) | 11.4 | 10.8 | 19.7 | 2.6 |

**关键发现**: 蒸馏出的 ViT-L 学生模型在 zero-shot 分割上大幅超越 ViT-g 教师，颠覆传统认知。

### Table 2: 蒸馏消融 — 初始化与 Masking

| # | 配置 | PC59 | PC60 | VOC21 | ADE150 |
|---|------|------|------|-------|--------|
| 1 | 标准预训练 (baseline) | 33.5 | 30.4 | 30.5 | 20.8 |
| 2 | 蒸馏 + mask 0.75 + 预训练初始化 | 20.3 | 16.6 | 26.1 | 9.7 |
| 3 | 蒸馏 + mask 0.75 + 随机初始化 | 25.3 | 23.7 | 27.6 | 13.4 |
| 4 | 蒸馏 + mask 0.0 + 随机初始化 | **31.4** | **28.6** | **30.8** | **20.0** |
| 5 | 蒸馏 + mask 0.0 + 随机初始化 + 文本冻结 | 29.2 | 24.6 | 28.5 | 19.7 |
| 6 | 蒸馏 + mask 0.0 + 随机初始化 + 文本训练 | 29.6 | 27.3 | 30.6 | 19.1 |
| 7 | 蒸馏 + mask 0.0 + 预训练初始化 | 22.7 | 22.4 | 24.4 | 12.1 |

**关键发现**: 移除 mask（0.0）+ 随机初始化是蒸馏成功的关键；预训练初始化会消除蒸馏收益。

### Table 3: iBOT vs iBOT++ Zero-Shot 分割对比

| Model | PC59 | PC60 | VOC21 | ADE150 |
|-------|------|------|-------|--------|
| TIPS (with iBOT) | 14.2 | 13.4 | 29.1 | 3.5 |
| TIPS with iBOT++ | **28.6** | **26.2** | **37.2** | **17.6** |

**关键发现**: iBOT++ 带来 zero-shot 分割的巨大提升，ADE150 从 3.5 跳升至 17.6。

### Table 4: 累积消融研究（100k steps, 224×224, ViT-g）

| Model | Seg.↑ | Depth↓ | Normals↓ | ImageNet↑ | I→T↑ | T→I↑ | ZS Seg.↑ |
|-------|-------|--------|----------|-----------|------|------|----------|
| TIPS ViT-g (复现) | 82.8 | 0.375 | 23.1 | 83.2 | 92.0 | 81.0 | 3.5 |
| + iBOT++ | 82.5 | 0.369 | 22.7 | 84.4 | 93.9 | 81.7 | **17.6** |
| + Multi-gran. Captions | 83.7 | 0.354 | 22.7 | 84.3 | 95.0 | 85.4 | **18.1** |
| + Head-only EMA | 83.8 | 0.353 | 22.8 | 84.1 | 95.4 | 85.4 | **19.1** |

**关键发现**: iBOT++ 贡献最大的 dense 对齐增益（+14.1 ZS ADE150）。多粒度文本提升全局和密集任务。Head-only EMA 性能持平甚至略优。

### Table 5: 密集图文评估 — Zero-Shot 分割（ViT-L）

| Method | PC59 | PC60 | VOC21 | ADE150 |
|--------|------|------|-------|--------|
| SigLIP2 SO/14 | — | 19.6 | 26.8 | 15.6 |
| SILC B/16 | 31.6 | — | — | 19.3 |
| DINOv2 (dino.txt) L/14 | 30.9 | — | — | 20.6 |
| TIPS L/14 | 33.5 | 30.4 | 30.5 | 20.8 |
| **TIPSv2 L/14** | **37.1** | **33.9** | **44.4** | **24.7** |

**关键发现**: TIPSv2 L/14 在所有 zero-shot 分割指标上全面领先，VOC21 上达 44.4 mIoU。

### Table 6: 全局图文评估 — 检索与分类

| Model | COCO I→T | Flickr I→T | DOCCI I→T | COCO T→I | Flickr T→I | DOCCI T→I | IN-1K |
|-------|-----------|------------|-----------|-----------|------------|-----------|-------|
| CLIP L/14 | 56.3 | 85.2 | 44.4 | 36.5 | 65.2 | 40.4 | 75.5 |
| OpenCLIP G/14 | 67.3 | 92.9 | — | 51.4 | 79.5 | — | 80.1 |
| SigLIP2 g/16 | 72.8 | 95.4 | — | 56.1 | 86.0 | — | 85.0 |
| TIPS g/14 | 74.0 | 93.0 | 57.2 | 59.4 | 84.5 | 58.8 | 79.9 |
| **TIPSv2 g/14** | **75.7** | **95.1** | **68.9** | **60.7** | **85.9** | **72.8** | **80.7** |

**关键发现**: TIPSv2 在 DOCCI 细粒度检索上提升最显著（+11.7 I→T, +14.0 T→I），多粒度文本训练贡献最大。

### Table 7: 纯图像评估

| Method | PASCAL Seg. | ADE20k Seg. | NYUv2 Depth↓ | NAVI Depth↓ | NYUv2 Normals↓ | NAVI Normals↓ | UnED | IN KNN | IN lin |
|--------|-------------|-------------|--------------|-------------|----------------|---------------|------|--------|--------|
| DINOv2 g/14 | 83.1 | 49.5 | 0.372 | 0.054 | 20.7 | 24.0 | 62.7 | 83.6 | 87.3 |
| TIPS g/14 | 83.6 | 49.9 | 0.353 | 0.058 | 21.9 | 24.2 | 68.2 | 83.3 | 86.2 |
| **TIPSv2 g/14** | **85.1** | **51.6** | **0.334** | **0.059** | **21.7** | **24.1** | **67.0** | **83.7** | **86.8** |

**关键发现**: TIPSv2 在纯视觉密集任务上也持续提升，验证了图文对齐训练对纯视觉特征质量的增益。

### Table 8: DINOv3 vs TIPSv2 (ViT-L)

| Model | ADE20k Seg.↑ | NYUv2 Depth↓ | IN 0-shot↑ | COCO I→T↑ | COCO T→I↑ | ADE150 ZS↑ |
|-------|-------------|--------------|------------|-----------|-----------|------------|
| DINOv3 L/16 | **54.9** | 0.352 | **82.3** | 63.7 | 45.6 | 24.7 |
| **TIPSv2 L/14** | 51.4 | **0.339** | 79.7 | **73.5** | **57.4** | **25.1** |

**关键发现**: DINOv3 在纯视觉任务略优，TIPSv2 在图文检索上大幅领先，展示两种范式各有优势。

### Table 9-16: 附录表格

**Table 9** — CLIP ViT-L + iBOT++: PC60 从 4.3→22.9, ADE20k seg 35.7→42.8, depth 0.571→0.434。

**Table 10** — CLIP + iBOT++ head-only EMA: PC60 4.3→22.1，验证 head-only EMA 在 CLIP 上也有效。

**Table 11** — CLIP 2 CLS ViT-g + iBOT++: PC60 18.2→28.2, ADE20k seg 40.1→47.2。

**Table 12** — iBOT++ 最优 mask ratio 为 75%（从预训练继承）。mask 仅在蒸馏时可移除（教师已提供强局部语义）。

**Table 13** — 多粒度文本消融：最优配置为 2 CLS (web / PaliGemma+Gemini 交替)，ADE20k 49.1, COCO I→T 76.2。

**Table 14** — SigLIP2 zero-shot 分割：B/16 (PC60 22.6, ADE150 16.4) 反超 g/16 (PC60 17.2, ADE150 13.9)，说明大模型 dense 对齐退化。

**Table 15** — TIPSv2 全模型族评估（B/L/SO/g）。

**Table 16** — 参数量：B/14 86.3M+109.6M=195.9M, L/14 304.0M+183.9M=487.9M, SO/14 413.3M+448.3M=861.7M, g/14 1.1B+389.1M=1.5B。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| WebLI (filtered) | 116M 图文对 | 网络爬取，含噪声 alt-text | 预训练 |
| PaliGemma Captions | 116M 合成描述 | 中等粒度 | 预训练文本增强 |
| Gemini 1.5 Flash Captions | 116M 合成描述 | 高粒度、丰富细节 | 预训练文本增强 |
| ADE150 / PASCAL Context / PASCAL VOC | — | 语义分割标注 | Zero-shot 分割评估 |
| Flickr30K / DOCCI / COCO | 1K / — / 5K | 图文检索 | 检索评估 (R@1) |
| ImageNet-1K | 1.28M/50K | 通用分类 | 零样本/线性分类 |
| NYUv2 / NAVI | — | 深度+法线标注 | 密集预测评估 |
| UnED | 8 域 | 细粒度检索 | 细粒度检索评估 |

### 实现细节

- **Backbone**: ViT-B/14, ViT-L/14, ViT-SO/14, ViT-g/14
- **优化器**: Adafactor
- **Batch Size**: 8192 (low-res 224), 4096 (high-res 336/448)
- **训练步数**: 90k (低分辨率) + 9k (高分辨率适配)
- **硬件**: 512 TPUv5 chips, ~2 days (ViT-g)
- **Loss 权重**: $\alpha=1.0$ (DINO), $\beta=2.0$ (iBOT/iBOT++)
- **Mask ratio**: 75% (iBOT++), 蒸馏阶段可降至 0
- **EMA 动量**: 投影头仅，中心+锐化防坍塌
- **数据增强**: 多局部裁剪 (DINO), 随机遮挡 (iBOT++)

### 可视化结果

- PCA 特征图：TIPSv2 生成更平滑、语义聚焦更强的特征（Fig 5, 7, 8）
- Zero-shot 分割：TIPSv2 的分割 mask 显著更干净、更完整（Fig 6, 9）
- iBOT++ vs iBOT：在所有定性场景中 iBOT++ 都优于 iBOT（Fig 9）

---

## 批判性思考

### 优点
1. **问题定义清晰**: 精准定位了 VLP 中 dense patch-text alignment 的核心瓶颈，并通过系统性实验验证
2. **方法简洁高效**: iBOT++ 仅去掉了 mask 指示器（一行代码改动），head-only EMA 节省 42% 参数，都在不增加复杂度的情况下取得显著增益
3. **实验极其充分**: 9 任务×20 数据集，16 张表格，10 张图，涵盖密集/全局/纯视觉任务，消融完备
4. **发现具有启发性**: 蒸馏学生可超越教师、大模型 dense 对齐退化等反直觉发现对社区有价值
5. **CLIP 兼容性**: 附录验证 iBOT++ 可独立应用于 CLIP，证明方法的普适性

### 局限性
1. **计算资源需求巨大**: 512 TPUv5 训练 ViT-g，难以复现；ViT-B 也需要大量资源
2. **Gemini 依赖性**: 多粒度文本需要 Gemini 1.5 Flash 生成，非开源方案
3. **DINOv3 对比不完整**: Table 8 仅在 ViT-L 上对比 DINOv3，且 DINOv3 在纯视觉任务上仍有优势
4. **文本编码器比较简单**: 使用标准 Transformer 而非更强的 LLM 文本塔，可能限制了图文对齐的上界
5. **WebLI 数据不可公开**: 预训练数据不开放，影响完全复现

### 潜在改进方向
1. **更强的文本编码器**: 用 PaliGemma 或 Gemini 级别的文本塔替代标准 Transformer
2. **结合 DINOv3**: 将 iBOT++ 和 head-only EMA 应用于 DINOv3 的训练流程，融合两种范式的优势
3. **扩展到 VLM**: 将 TIPSv2 的密集对齐能力注入多模态大模型（如 LLaVA），提升 grounding/VQA 中的空间理解
4. **降低计算门槛**: 探索在小规模数据上的有效性，或设计更高效的 iBOT++ 变体

### 可复现性评估
- [x] 代码开源 ([github.com/google-deepmind/tips](https://github.com/google-deepmind/tips))
- [x] 预训练模型 (提供 TIPSv2 模型族)
- [x] 训练细节完整 (Appendix A.9 提供超参数)
- [ ] 数据集可获取 (WebLI 不公开，PaliGemma/Gemini 描述未发布)

---

## 关联笔记

### 基于
- [[TIPS]]: TIPSv2 的前身，Text-Image Pretraining with Spatial Awareness (ICLR 2025)
- [[iBOT]]: 原始 iBOT 遮蔽图像建模方法，iBOT++ 的基础
- [[DINO]]: 自蒸馏范式，TIPSv2 的 DINO 损失来源
- [[CLIP]]: 对比语言-图像预训练的开创工作

### 对比
- [[DINOv2]]: 纯视觉自监督的强基线，密集任务出色但缺乏文本对齐
- [[DINOv3]]: DINOv2 的后续，Table 8 纯视觉任务仍略优于 TIPSv2
- [[SigLIP]]: Sigmoid loss 替代 softmax 的对比学习，SigLIP2 大规模下 dense 对齐退化
- [[Perception Encoder]]: 56% 更多参数 + 47x 更多训练数据，TIPSv2 仍超越

### 方法相关
- [[iBOT++]]: 核心创新——对所有 patch token 施加蒸馏损失
- [[Head-only EMA]]: 仅对投影头做 EMA，减少 42% 训练参数
- [[Multi-Granularity Caption Sampling]]: PaliGemma + Gemini 交替采样策略
- [[Patch-Text Alignment]]: 核心目标——密集 patch 表示与文本的对齐
- [[Knowledge Distillation]]: 蒸馏学生超越教师的关键发现
- [[Exponential Moving Average]]: EMA 中心化和动量更新机制

### 硬件/数据相关
- [[WebLI Dataset]]: Google 内部 116M 图文对数据集
- [[PaliGemma]]: 用于生成合成描述的 3B VLM
- [[Gemini Flash]]: Gemini 1.5 Flash 用于高粒度文本增强

---

## 速查卡片

> [!summary] TIPSv2: Advancing Vision-Language Pretraining
> - **核心**: iBOT++ 让所有 patch token 参与损失 + head-only EMA + 多粒度文本，大幅提升密集图文对齐
> - **方法**: 双塔 ViT-Text + CLIP/DINO/iBOT++ 三损失 + 蒸馏增强 + 多粒度交替描述
> - **结果**: 9 任务 20 数据集，zero-shot ADE150 3.5→19.1，DOCCI 检索 +14.0 (T→I)，纯视觉密集任务也持续提升
> - **代码**: [github.com/google-deepmind/tips](https://github.com/google-deepmind/tips)

---

*笔记创建时间: 2026-05-08*
