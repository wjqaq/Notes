---
title: "Vision Transformers Need More Than Registers"
method_name: "LaSt-ViT"
authors: [Cheng Shi, Yizhou Yu, Sibei Yang]
year: 2026
venue: arXiv
tags: [vision-transformer, cls-token, attention-artifacts, self-supervised, feature-aggregation, register-tokens]
zotero_collection: 多模态/ViT
image_source: online
arxiv_html: https://arxiv.org/html/2602.22394v2
created: 2026-05-08
---

# 论文笔记：Vision Transformers Need More Than Registers

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | ShanghaiTech University / The University of Hong Kong |
| 日期 | April 2026 |
| 项目主页 | (未提供) |
| 对比基线 | [[DINOv2]] + [[Register Tokens]] |
| 链接 | [arXiv](https://arxiv.org/abs/2602.22394) |

---

## 一句话总结

> ViT 的"伪影"根源不是高范数 token，而是 [[CLS Token]] 对背景 patch 的"惰性聚合"；[[LaSt-ViT]] 用 [[Frequency-aware Selective Aggregation|频域感知选择性聚合]] 把 CLS 锚定到前景，跨 12 个基准一致提升。

---

## 核心贡献

1. **提出两个诊断指标**: [[Patch Score]] 衡量 patch 与 CLS 的相似度，[[Point-in-Box]] 量化最显著 patch 是否落在前景框内，首次系统性揭露 ViT 的背景偏置。
2. **提出惰性聚合假设**: 将 [[Register Tokens|Register]] 只能治标的现象归因于两个根因——[[Coarse-grained Semantic Supervision|粗粒度语义监督]]与 ViT 的全局依赖捷径。
3. **提出 LaSt-ViT**: 通过 [[Frequency-aware Selective Aggregation|FFT 稳定性打分]]与 [[Channel-wise Top-K Pooling|通道级 Top-K 池化]]，把 CLS 的聚合限制在前景 patch 上，无需额外参数也无需重训 backbone。
4. **广泛验证**: 在全监督 / 弱监督 / 自监督三种范式、12 个下游基准上稳定涨点，包括 [[Open-Vocabulary Segmentation|开放词汇分割]]、[[Unsupervised Object Discovery|无监督物体发现]]。

---

## 问题背景

### 要解决的问题

[[Vision Transformer|ViT]] 在训练后会出现 **artifacts**（伪影）——背景 patch 错误地获得高 [[Patch Score]] 或极大的 feature norm，污染注意力图、降低 patch-level 语义质量。先前 [[Register Tokens]]（Darcet et al., ICLR 2024）通过加入额外"寄存器 token"来吸收这些高范数 outlier，但这类 artifacts 在其他监督范式（CLIP、DINO、有监督分类）下仍普遍存在。

### 现有方法的局限

- **Register Tokens**: 只消除了"高范数 token"这一表面现象，但 [[Patch Score]] 分布的背景偏置（artifacts 的根因）并未改善。
- **Window Attention / 局部注意力**: 限制感受野可部分缓解背景 shortcut，但牺牲全局建模能力。
- **post-hoc 修复**: 如特征归一化、DINOv2 后处理，无法从训练层面纠正偏置。

### 本文的动机

作者通过对多种 ViT（有监督、CLIP、DINO-v1、DINOv2）的 Patch Score 分布可视化发现：**无论是否使用 Register**，背景 patch 都倾向于获得比前景更高的 CLS 相似度——这说明 [[Register Tokens]] 没解决根本问题。作者提出假设：ViT 的 [[CLS Token]] 在训练早期就"偷懒"地利用海量背景 patch 作为全局语义的 shortcut，因为图像级监督信号缺乏空间 grounding。

---

## 方法详解

### 模型架构

[[LaSt-ViT]] 不改变 [[Vision Transformer|ViT]] 的 encoder，只替换**最终的 CLS 聚合层**：

- **输入**: 图像 $\mathbf{x}$ 经 [[Patch Embedding]] 得到 $N$ 个 patch token $\mathbf{x}_{patch} \in \mathbb{R}^{N \times D}$
- **Backbone**: 原 ViT encoder（可以是 [[DINOv2]]、[[CLIP]]、或有监督预训练 ViT），不改动
- **核心替换**: 原来的 [[Global Average Pooling|GAP]] 或 learnable CLS 聚合 $\rightarrow$ [[Frequency-aware Selective Aggregation|频域稳定性打分 + 通道级 Top-K 池化]]
- **输出**: 新的 $\mathcal{Q}_{CLS} \in \mathbb{R}^D$ 和每个 patch 的投票数 $v_i$（用于下游定位）
- **额外参数**: 0（仅有超参 $K$、高斯低通滤波器带宽）

### 核心模块

#### 模块1: [[Patch Score]] 与 [[Point-in-Box]] 诊断

**设计动机**: 先量化问题。给定训练完的 ViT，度量每个 patch 对 CLS 表征的贡献，并定位"贡献最大的 patch"是否落在前景。

**具体实现**:
- 对每个 patch 计算 [[Cosine Similarity|余弦相似度]] 与当前 CLS 向量之间的打分 $\mathcal{S}_p$
- 在 ImageNet 验证集单物体子集上，取 Top-1 patch，若其落在 ground-truth bbox 内则计 1

#### 模块2: [[Frequency-aware Selective Aggregation|FFT 稳定性打分]]

**设计动机**: 前景 patch 在通道维上的特征应当**信息密度高但冗余低**，即去除低频成分后差异显著；背景 patch 则更多是低频/平滑响应，去除低频后接近原值。

**具体实现**:
- 对 patch 特征沿**通道维**做 [[1D FFT]]
- 用高斯 [[Low-pass Filter|低通滤波]] 抑制高频
- [[Inverse FFT]] 取实部得到低频重建 $\hat{\mathbf{x}}_{patch}$
- 每个 (patch $i$, 通道 $j$) 计算稳定性 $\mathbf{S}_{i,j}$

#### 模块3: [[Channel-wise Top-K Pooling|通道级 Top-K 池化]]

**设计动机**: 不同通道编码不同语义，简单的跨通道聚合会被强背景通道主导；改为"每个通道独立挑选最稳定的 K 个 patch"，并把每个 patch 被选中的次数作为 [[Vote Count]] $v_i$。

**具体实现**:
- 每个通道 $j$ 取 $\mathbf{S}_{:,j}$ 的 [[Top-K Selection|Top-K]] 索引集 $\mathcal{I}_K(j)$
- 新 CLS 的第 $j$ 维 = 这 $K$ 个 patch 在通道 $j$ 上的均值
- Patch $i$ 的投票数 = 在所有通道里被选中的次数

#### 模块4: 下游迁移（Section 5.2）

**无监督物体发现**: 以投票数 $v_i$ 构造前景 mask（阈值 = mean + 1 std），用于 CorLoc 指标。

**开放词汇分割/检测**: 复用改进后的 patch-CLS 对齐——对每个 patch 计算它与文本 embedding 的相似度，实现零样本分割。无需训练 mask decoder。

---

## 关键公式

### 公式1: [[Vision Transformer|ViT GAP 聚合]]

$$
\mathbf{x}_{patch} = \mathcal{P}_{enc}(\mathcal{P}_{emb}(\mathbf{x})), \quad \mathcal{Q}_{CLS} = \text{Pooling}(\mathbf{x}_{patch})
$$

**含义**: 先 [[Patch Embedding]] 再 encoder 得到 patch tokens，再做池化得到全局表征。

**符号说明**:
- $\mathcal{P}_{emb}$: [[Patch Embedding]] 层
- $\mathcal{P}_{enc}$: Transformer encoder
- $\mathbf{x}_{patch} \in \mathbb{R}^{N \times D}$: $N$ 个 patch token
- $\mathcal{Q}_{CLS} \in \mathbb{R}^{D}$: 全局表征

### 公式2: [[CLS Token|可学习 CLS Token 聚合]]

$$
[\mathbf{x}_{patch}, \mathcal{Q}_{CLS}] = \mathcal{P}_{enc}([\mathcal{P}_{emb}(\mathbf{x}), \mathcal{O}_{CLS}])
$$

**含义**: 在 patch 前拼接一个可学习的 $\mathcal{O}_{CLS}$ token，随 patch 一起过 encoder，最终位置上的输出作为全局表征。

**符号说明**:
- $\mathcal{O}_{CLS}$: 可学习的查询 token 初始化
- $[\cdot, \cdot]$: 序列拼接

### 公式3: [[Patch Score|Patch Score 定义]]

$$
\mathcal{S}_p = \frac{\mathbf{x}_{patch} \cdot \mathcal{Q}_{CLS}}{\|\mathbf{x}_{patch}\|_2 \, \|\mathcal{Q}_{CLS}\|_2}
$$

**含义**: 每个 patch 与 CLS 的 [[Cosine Similarity|余弦相似度]]，用于度量 patch 对全局语义的贡献。

**符号说明**:
- $\mathbf{x}_{patch} \in \mathbb{R}^{N \times D}$: patch token
- $\mathcal{Q}_{CLS} \in \mathbb{R}^{D}$: CLS 向量
- $\mathcal{S}_p \in \mathbb{R}^{N}$: 每个 patch 的得分，越大越"被 CLS 信任"

### 公式4: [[Frequency-aware Selective Aggregation|频域低通重建]]

$$
\mathbf{x}_{FFT} = \text{FFT1D}(\mathbf{x}_{patch}), \quad \mathbf{x}_{LP} = \mathbf{x}_{FFT} \odot \mathbf{g}, \quad \hat{\mathbf{x}}_{patch} = \Re\{\text{IFFT1D}(\mathbf{x}_{LP})\}
$$

**含义**: 对每个 patch 在通道方向做 [[1D FFT]]，用高斯 [[Low-pass Filter|低通滤波器]] $\mathbf{g}$ 抑制高频后再反变换取实部。

**符号说明**:
- $\mathbf{g}$: 高斯低通核（仅保留低频成分）
- $\odot$: [[Hadamard Product|逐元素乘]]
- $\Re\{\cdot\}$: 取实部
- $\hat{\mathbf{x}}_{patch}$: 低频重建特征（近似"语义背景")

### 公式5: [[Frequency-aware Selective Aggregation|稳定性打分]]

$$
\mathbf{S}_{i,j} = \frac{\hat{\mathbf{x}}_{patch}[i,j]}{|\hat{\mathbf{x}}_{patch}[i,j] - \mathbf{x}_{patch}[i,j]| + \varepsilon}
$$

**含义**: (patch $i$, 通道 $j$) 的稳定性 = 低频重建幅值 / 去除低频后的残差幅值。比值越大说明该 patch 在该通道的响应越"稳定+有信息"，越可能是前景。

**符号说明**:
- $\hat{\mathbf{x}}_{patch}[i,j]$: 低频重建值
- $\mathbf{x}_{patch}[i,j] - \hat{\mathbf{x}}_{patch}[i,j]$: 高频残差
- $\varepsilon$: 数值稳定小量

### 公式6: [[Channel-wise Top-K Pooling|通道级 Top-K 索引]]

$$
\mathcal{I}_K(j) = \text{TopK}\big(\{\mathbf{S}_{i,j}\}_{i=1}^{N},\; K\big)
$$

**含义**: 每个通道 $j$ 独立取稳定性最高的 $K$ 个 patch 索引，形成通道专属的"前景集合"。

**符号说明**:
- $\mathcal{I}_K(j) \subset \{1, \ldots, N\}$: 通道 $j$ 上 Top-K patch 的索引
- $K$: 超参，论文在 14×14=196 个 patch 时取 $K \approx 49$

### 公式7: [[Channel-wise Top-K Pooling|新 CLS 聚合]]

$$
\mathcal{Q}_{CLS}[j] = \frac{1}{K} \sum_{i \in \mathcal{I}_K(j)} \mathbf{x}_{patch}[i, j]
$$

**含义**: 新 CLS 的每一维只聚合该通道的 Top-K patch 的对应通道值，避免背景 patch 污染。

**符号说明**:
- $j \in \{1, \ldots, D\}$: 通道索引
- 每个通道的聚合独立，因此**不同通道关注不同的前景区域**

### 公式8: [[Vote Count|Patch 投票数]]

$$
v_i = \sum_{j=1}^{D} \mathbb{1}\{i \in \mathcal{I}_K(j)\}
$$

**含义**: Patch $i$ 在所有通道中被选中的次数。$v_i$ 越大 = 该 patch 越可能是前景，可直接作为定位/分割的前景分数。

**符号说明**:
- $\mathbb{1}\{\cdot\}$: [[Indicator Function|指示函数]]
- $v_i \in \{0, 1, \ldots, D\}$: 投票数，用于下游无监督定位

---

## 关键图表

### Figure 1: LazyStrike 统一框架

![Figure 1 teaser](https://arxiv.org/html/2602.22394v2/fig/visv3/10.png)
![Figure 1 teaser](https://arxiv.org/html/2602.22394v2/fig/visv3/11.png)
![Figure 1 teaser](https://arxiv.org/html/2602.22394v2/fig/visv3/12.png)
![Figure 1 teaser](https://arxiv.org/html/2602.22394v2/fig/visv3/13.png)
![Figure 1 teaser](https://arxiv.org/html/2602.22394v2/fig/visv3/14.png)

**说明**: 展示 LaSt-ViT 在有监督 / [[CLIP]] / [[DINOv2]] 三种监督范式下的 [[Patch Score]] 热图对比。原始 ViT 的高响应散布在背景，[[LaSt-ViT]] 则集中在物体上。强调"一个框架跨范式一致缓解 artifacts"。

### Figure 2: Patch-score 分布与 masking probe

![Figure 2 patch score distribution](https://arxiv.org/html/2602.22394v2/x1.png)

**说明**: 左侧为前景 / 背景 patch score 的直方图——**原始 ViT 中背景 patch 的均值反而高于前景**；右侧通过 [[Masking Probe]]（渐进遮盖高分 patch 看分类准确率下降情况）验证：遮掉高分 patch 准确率下降慢（说明它们不是真正有判别性），遮掉前景（低分）反而掉得快。这是"惰性聚合"的直接证据。

### Figure 3: Training dynamics

![Figure 3 training dynamics](https://arxiv.org/html/2602.22394v2/fig/vis_stage.png)

**说明**: ImageNet-1k 训练过程中的 [[Point-in-Box]] 随 epoch 的演化。早期 ViT 就已经把 CLS 锚定到背景——随着训练推进 PiB 下降（越训越"懒"），说明 artifacts 是优化 shortcut 的结果而非模型容量问题。

### Figure 4: 粗粒度语义监督的影响

![Figure 4 coarse supervision effect](https://arxiv.org/html/2602.22394v2/fig/vis_stagev2.png)

**说明**: 对比不同监督粒度（image-level label vs. patch-level contrastive）下 [[Patch Score]] 分布的变化。[[Coarse-grained Semantic Supervision|粗粒度监督]] 下背景偏置更严重，验证假设。

### Figure 5: LaSt-ViT 的 CLS 关注位置

![Figure 5 CLS attention](https://arxiv.org/html/2602.22394v2/x2.png)

**说明**: 可视化 [[LaSt-ViT]] 的 [[Vote Count]] $v_i$，热图显示投票集中在物体主体区域。对比 DINOv2 + Register 的 attention 仍有散布的背景激��。

### Figure 6: Feature norm 评估

![Figure 6 feature norm](https://arxiv.org/html/2602.22394v2/fig/vis_normv3.png)

**说明**: 特征 norm 的空间分布。原始 ViT 存在稀疏的"高 norm outlier token"，[[Register Tokens|Register]] 把这些 outlier 转移到额外 token 上但背景 patch 仍偏高；[[LaSt-ViT]] 的 norm 分布整体更平滑且前景 norm 更突出，说明**根治而非"转移"**了问题。

### Figure 7: PCA 分量可视化

![Figure 7 PCA](https://arxiv.org/html/2602.22394v2/fig/vis_pca.png)

**说明**: 对 patch 特征做 [[Principal Component Analysis|PCA]] 取前 3 个主成分渲染为 RGB。[[LaSt-ViT]] 的主成分清晰勾勒物体轮廓，且不同物体类别有稳定的颜色语义；baseline 则边界模糊。

### Table 1: 不同监督范式下的 Point-in-Box

| Method | PiB |
|--------|-----|
| Supervised ViT-B | 42.7 |
| Supervised ViT-B + [[Register Tokens\|Register]] | 相近 |
| DINO-v1 | 44.5 |
| CLIP ViT-B/16 | 39.8 |

**说明**: 作者的核心"打脸证据"——各类预训练 ViT（不论是否加 Register）的 Top-1 patch 落在前景 bbox 中的比例都远低于随机对角估计，说明 artifacts 是普适问题。

### Table 2: Window Attention 消融（ViT-Small）

| 配置 | PiB |
|------|-----|
| Full Attention | 较低 |
| + [[Window Attention\|Window Attention]] | 上升 |

**说明**: 单纯限制感受野可以缓解一部分背景 shortcut（验证"全局依赖"是惰性聚合的帮凶之一），但牺牲全局建模能力，不是最优解。

### Table 3: LaSt-ViT 在 PiB 上的提升

| Method | PiB | +LaSt-ViT | $\Delta$ |
|--------|-----|-----------|----------|
| Supervised ViT-B | 42.7 | **55.1** | +12.4 |
| DINO-v1 ViT-B | 44.5 | **69.7** | +25.2 |
| CLIP ViT-B/16 | 39.8 | **50.1** | +10.3 |

**说明**: 跨三种监督范式都显著改善前景定位。[[DINO]] 提升最大（+25.2）是因为其 patch-level 表征本已较好，[[LaSt-ViT]] 的聚合策略可以充分利用。

### Table 4: [[Semantic Segmentation|语义分割]] mIoU（零样本，6 个基准）

| Method | ADE20K | (其他 5 基准均有涨点) |
|--------|--------|------------------------|
| CLIP ViT-B/16 | 3.1% | ... |
| + [[LaSt-ViT]] | **8.3%** | **+5.2** |
| CLIP ViT-L/14 | 1.6% | ... |
| + [[LaSt-ViT]] | **8.4%** | **+6.8** |

**说明**: 在零样本开放词汇分割上 mIoU 翻倍以上。原始 [[CLIP]] 的 patch 特征几乎无法用于稠密预测，LaSt-ViT 直接把它变成可用的分割特征。

### Table 5: [[Open-Vocabulary Detection|开放词汇检测/分割]]

| Method | Novel AP |
|--------|----------|
| F-ViT ViT-B/16 | 117.5 |
| F-ViT ViT-B/16 + LaSt-ViT | **133.3** (+15.8) |
| F-ViT ViT-L/14 | 124.7 |
| F-ViT ViT-L/14 + LaSt-ViT | **139.1** (+14.4) |

**说明**: 在 F-ViT 框架上替换聚合方式即获得 ~15 点的 novel 类提升，说明改善后的 CLS-patch 对齐更有利于 unseen 类别的定位。

### Table 6: VOC12 上 Patch Score 直接分割

**说明**: 不训练任何分割头，仅用 patch score > 阈值得到粗分割——[[LaSt-ViT]] 大幅超过原 ViT 的 patch score 分割结果，说明其 patch 打分已具备"前景/背景"判别力。

### Table 7: [[Unsupervised Object Discovery|无监督物体发现]] CorLoc

**说明**: 使用 [[Vote Count]] $v_i$ 阈值化得到前景 mask 计算 [[CorLoc]]，LaSt-ViT 在 VOC07/12 上超过 DINO + LOST 等专门方法。

### Table 8: Text-supervised ViT 消融（K 值）

| K (共 196 patch) | VOC 分割 |
|------------------|----------|
| 196（不选） | 13.5 |
| 98 | 中等 |
| **49** | **75.8** (峰值) |
| 24 | 下降 |

**关键发现**: $K \approx N/4$ 时最佳——太大则退化回原始聚合（保留背景），太小则丢失前景信息。

### Table 9: Label-supervised ViT 消融

**说明**: 对有监督预训练 ViT 做同样的 $K$ 扫描与滤波器带宽消融，结论一致：稳定性打分 + 通道级 Top-K 两者缺一不可。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[ImageNet-1k]] | 1.28M | 单物体分类 | 训练 + PiB 评估 |
| [[LAION-400M]] | 400M | 图文对 | CLIP 微调 |
| [[VOC07_12|VOC07/12]] | ~11K | 多物体检测 | 对象发现 / CorLoc |
| [[ADE20K]] | 20K | 密集分割 | 开放词汇分割 |
| [[COCO]] | 118K | 检测+分割 | 开放词汇检测 |

### 实现细节

- **Backbone**: ViT-S / ViT-B / ViT-L（14x14 或 16x16 patch）
- **$K$ 选择**: $\approx N/4$（例如 196 patch 取 K=49）
- **低通滤波器**: 高斯核，带宽约通道维的 10%-25%
- **训练方式**: 不需要从零训练——直接在 **预训练权重**（CLIP/DINOv2 官方权重）上替换聚合层即可评估；text-supervised 场景下可选在 LAION-400M 微调
- **硬件**: 论文未强调 GPU 需求；因无新增参数，推理开销主要来自 FFT + TopK，很小

### 可视化结果

[[LaSt-ViT]] 的 Patch Score / Vote Count 热图一致地聚焦物体主体；[[PCA]] 可视化显示不同物体类别在前三主成分上有稳定语义聚类（类似 DINOv2 但前景更干净）。

---

## 批判性思考

### 优点

1. **诊断工具独立有价值**: [[Patch Score]] + [[Point-in-Box]] 即使不使用 LaSt-ViT 也可作为 ViT 训练过程的监控指标。
2. **零新增参数**: 对预训练权重"即插即用"，无需重训 backbone，实际部署友好。
3. **跨范式一致涨点**: 监督、弱监督、自监督三种训练方式下都工作，说明触及的是 ViT 架构共性问题。
4. **理论解释到位**: 把"高范数 token"这一 Register 论文的观察重新定位为"惰性聚合的一种表现"，并通过粗粒度监督 + 全局依赖两条路径自洽地解释了现象。

### 局限性

1. **$K$ 是硬超参**: 最佳 $K$ 依赖 patch 数量和数据集，论文没有给自适应选择机制。
2. **FFT 方向的选择略启发式**: 在通道维做 FFT 假设相邻通道在语义上可比，但 Transformer 的通道是无序的，这一选择缺少更严格的理论依据。
3. **实际使用受限于"后处理"属性**: 不直接改善 backbone 内部表征——意味着 LaSt-ViT 的 CLS 虽好，但 patch 特征本身仍然受训练偏置影响。
4. **与 Register 的对比在训练强度上不完全对等**: Register 需重训，LaSt-ViT 是 post-hoc 替换，两者在"改动量"上不同，部分基准上更公平的比较应该是 Register + LaSt-ViT 叠加。

### 潜在改进方向

1. **可微的 Top-K**: 把硬 Top-K 改成 [[Gumbel Top-K]] 或稀疏注意力，让 LaSt-ViT 可端到端训练。
2. **训练阶段 loss**: 将 Patch Score 与前景 mask 的对齐设计为辅助损失，从根上消除惰性聚合。
3. **多尺度 FFT**: 在空间维（2D FFT on patch grid）或跨 head 维联合做频域选择。
4. **与 [[Register Tokens]] 融合**: Register 吸收 outlier + LaSt-ViT 做正向选择，可能相互补充。

### 可复现性评估

- [ ] 代码开源（截至笔记时未见官方仓库链接）
- [ ] 预训练模型（方法本身是 post-hoc，理论上可直接在官方 CLIP/DINOv2 权重上实现）
- [x] 训练细节完整（公式明确，超参给出范围）
- [x] 数据集可获取（均为公开基准）

---

## 关联笔记

### 基于

- [[Vision Transformer]]: 被分析的 backbone，论文基于其 CLS 聚合机制提出改进
- [[CLIP]] / [[DINO]] / [[DINOv2]]: 作为三种监督范式的代表基线
- [[Register Tokens]]: 前序工作，本文指出其只治标不治本

### 对比

- [[Register Tokens]]: 最直接的对比对象——两者都处理 ViT artifacts，但作用机理不同
- [[Window Attention]]: 作为"限制全局依赖"的消融基线

### 方法相关

- [[Patch Score]]: 核心诊断指标
- [[Point-in-Box]]: 前景定位评估
- [[Frequency-aware Selective Aggregation]]: 核心算法
- [[Channel-wise Top-K Pooling]]: 核心聚合方式
- [[Vote Count]]: 下游定位信号

### 下游应用

- [[Open-Vocabulary Segmentation]]
- [[Unsupervised Object Discovery]]

---

## 速查卡片

> [!summary] Vision Transformers Need More Than Registers
> - **核心**: ViT artifacts 的根因是 CLS 对背景的"惰性聚合"，不是高范数 token
> - **方法**: 通道维 FFT 稳定性打分 + 通道级 Top-K 池化，替换 CLS 聚合
> - **结果**: 跨监督/弱监督/自监督 12 个基准全部涨点，zero-shot 分割 mIoU 翻倍
> - **代码**: 暂未公开

---

*笔记创建时间: 2026-05-08*
