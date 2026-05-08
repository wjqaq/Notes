---
title: "Beyond SFT-to-RL: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL"
method_name: "PRISM"
authors: [Sudong Wang, Weiquan Huang, Xiaomin Yu, Zuhao Yang, Hehai Lin, Keming Wu, Chaojun Xiao, Chen Chen, Wenxuan Wang, Beier Zhu, Yunjian Zhang, Chengwei Qin]
year: 2025
venue: arXiv
tags: [multimodal-rl, distribution-alignment, on-policy-distillation, vlm, reasoning]
zotero_collection: _待整理
image_source: online
arxiv_html: https://arxiv.org/html/2604.28123
created: 2025-05-08
---

# 论文笔记：Beyond SFT-to-RL: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | HKUST (Guangzhou), Tsinghua University, Nanyang Technological University, Renmin University of China, USTC, University of Chinese Academy of Sciences |
| 日期 | April 2025 |
| 项目主页 | https://github.com/XIAO4579/PRISM |
| 链接 | [arXiv](https://arxiv.org/abs/2604.28123) / [Code](https://github.com/XIAO4579/PRISM) |

---

## 一句话总结

> PRISM 通过在 SFT 和 RLVR 之间插入分布对齐阶段，使用 MoE 判别器进行对抗性在线策略蒸馏，解决多模态模型后训练中的分布漂移问题。

---

## 核心贡献

1. **三阶段后训练范式**: 提出 PRISM 流水线（SFT → 分布对齐 → RLVR），显式修复 SFT 引入的分布漂移
2. **MoE 判别器设计**: 设计感知专家和推理专家的混合判别器，分别评估视觉描述和推理轨迹质量
3. **黑盒对抗蒸馏**: 无需教师 logits，仅通过样本级别的对抗游戏实现分布对齐

---

## 问题背景

### 要解决的问题

标准 SFT 后接 RLVR 的训练范式存在**分布漂移**问题：SFT 后的模型既偏离原始能力分布，也无法忠实地匹配监督分布。这种漂移在多模态模型中被放大，因为**感知错误和推理失败遵循不同的漂移模式并在后续 RL 中复合**。

### 现有方法的局限

- **SFT 阶段**: 将模型置于"妥协状态"，偏离演示策略分布和原始有利分布
- **RLVR 阶段**: 直接在漂移的分布上优化，导致错误复合
- **现有 OPD 方法**: 如 VOLD 仍将蒸馏耦合在 RL 阶段内，未解决 SFT 遗留的分布差距

### 本文的动机

多模态推理中的分布漂移具有**内在异质性**：视觉定位和逻辑推理以定性不同的方式退化。因此需要专门的对齐阶段，通过解耦的感知和推理专家提供针对性反馈。

---

## 方法详解

### 模型架构

PRISM 采用**三阶段流水线**架构：

- **Stage 1 - Cold-Start SFT**: 在精选演示数据上进行监督微调
- **Stage 2 - Distribution Alignment**: 使用 MoE 判别器进行对抗性在线策略蒸馏
- **Stage 3 - RLVR**: 使用可验证奖励进行强化学习

### 核心模块

#### 模块1: Mixture-of-Experts 判别器

**设计动机**: 多模态推理的分布漂移具有异质性，需要专门化专家分别评估感知和推理质量

**具体实现**:
- **感知专家 (D_v)**: 评估视觉描述 $c$，衡量视觉定位质量
- **推理专家 (D_r)**: 评估推理轨迹 $t$，衡量一致性和有效性
- **组合评分**: 加权融合两个专家的判断

#### 模块2: 对抗性在线策略蒸馏

**设计动机**: 通过对抗游戏使策略分布逼近监督分布，无需教师 logits

**具体实现**:
- 判别器通过 [[Bradley-Terry Model|Bradley-Terry 损失]]学习区分监督样本和策略输出
- 策略通过策略梯度最大化判别器奖励
- 使用组归一化计算优势函数，避免 KL 正则化以允许分布自由移动

#### 模块3: 结构化响应格式

**设计动机**: 显式分解视觉描述和推理轨迹，支持 MoE 判别器的专门化评估

**具体实现**:
- 输出格式: `<caption>` + `<reasoning>` + `<answer>`
- 从响应中提取 $c$（视觉描述）和 $t$（推理轨迹）
- 分别输入感知专家和推理专家

---

## 关键公式

### 公式1: [[MoE Discriminator|MoE 判别器组合评分]]

$$
r(x,y) = \alpha \cdot D_v(x,c) + (1-\alpha) \cdot D_r(x,t)
$$

**含义**: 加权融合感知专家和推理专家的判别分数

**符号说明**:
- $D_v(x,c)$: 感知专家对视觉描述 $c$ 的评分
- $D_r(x,t)$: 推理专家对推理轨迹 $t$ 的评分
- $\alpha$: 平衡权重（默认 0.5）

### 公式2: [[Bradley-Terry Loss|判别器损失]]

$$
\mathcal{L}_{D_k} = -\mathbb{E}_{(x,y^+,y^-)\sim\mathcal{T}}[\log \sigma(D_k(x,y^+_k) - D_k(x,y^-_k))], \quad k \in \{v,r\}
$$

**含义**: 使用 Bradley-Terry 模型训练判别器区分监督样本 $y^+$ 和策略输出 $y^-$

**符号说明**:
- $y^+$: 来自监督池的样本
- $y^-$: 策略生成的样本
- $\sigma(\cdot)$: Sigmoid 函数
- $k \in \{v,r\}$: 感知专家或推理专家

### 公式3: [[Group Normalization|优势计算]]

$$
A_i = \frac{r(x,y^-_i) - \text{mean}(\{r(x,y^-_j)\}_{j=1}^N)}{\text{std}(\{r(x,y^-_j)\}_{j=1}^N)}
$$

**含义**: 通过组归一化计算每个样本的优势函数，用于策略更新

**符号说明**:
- $N$: 组大小（默认 16）
- $r(x,y^-_i)$: 第 $i$ 个样本的判别器奖励
- $A_i$: 归一化后的优势值

### 公式4: [[Minimax Objective|对抗目标]]

$$
\min_\theta \max_\phi \mathbb{E}_{(x,y^+)\sim\mathcal{T}, y^-\sim G_\theta(\cdot|x)}[r_\phi(x,y^+) - r_\phi(x,y^-)]
$$

**含义**: 判别器最大化区分能力，策略最小化判别差距

**符号说明**:
- $\theta$: 策略参数
- $\phi$: 判别器参数
- $G_\theta(\cdot|x)$: 策略分布

### 公式5: [[RLVR Reward|可验证奖励]]

$$
r_v(x,y) = r_{acc}(x,y) + r_{fmt}(x,y)
$$

**含义**: RLVR 阶段的奖励函数，包含准确性和格式奖励

**符号说明**:
- $r_{acc}(x,y)$: 准确性奖励（权重 0.8）
- $r_{fmt}(x,y)$: 格式奖励（权重 0.2）

---

## 关键图表

### Figure 1: PRISM 流水线概览

![Figure 1](https://arxiv.org/html/2604.28123v2/x1.png)

**说明**: PRISM 的三阶段流水线。(a) SFT 引入策略与监督分布之间的漂移；(b) 对齐阶段使用 MoE 判别器通过对抗性在线策略蒸馏修复漂移；(c) 对齐后的策略为下游 RLVR 提供更强的初始化。

### Figure 2: 分布对齐架构

![Figure 2](https://arxiv.org/html/2604.28123v2/x2.png)

**说明**: 分布对齐阶段的架构。MoE 判别器通过 Bradley-Terry 损失训练以区分监督和策略输出；策略通过策略梯度更新以最大化 MoE 组合奖励。

### Figure 3: 训练动态

![Figure 3](https://arxiv.org/html/2604.28123v2/x3.png)

**说明**: 感知专家（左）和推理专家（右）的奖励差距（监督 - 策略）训练动态。训练 500 步，扩展至 900 步验证收敛稳定性。感知专家快速收敛，推理专家上升更缓慢且有更大波动。

### Figure 4: 分布对齐的结构代理

![Figure 4](https://arxiv.org/html/2604.28123v2/x4.png)

**说明**: PRISM 各阶段的推理步数（左）和每个描述项数（右）。对齐阶段显著缩小策略与监督分布之间的结构性差异。

### Figure 5: Token 效率对比

![Figure 5](https://arxiv.org/html/2604.28123v2/x5.png)

**说明**: MathVision、MathVerse 和 MMMU-Pro 上的 Token 效率对比（Qwen3-VL-4B）。PRISM+GRPO 在所有三个基准上以更少的 Token 达到更高的准确率。

### Table 1: 主要结果（准确率 %）

| Method | MathVista | MathVerse | MathVision | WeMath | MMMU | MMMU-Pro | HallusionBench | Avg |
|--------|-----------|-----------|------------|--------|------|----------|----------------|-----|
| **Qwen3-VL-4B** | | | | | | | | |
| Instruct | 74.9 | 59.0 | 36.5 | 70.7 | 63.6 | 45.1 | 68.2 | 59.7 |
| + SFT | 71.5 | 58.4 | 31.9 | 70.6 | 53.6 | 42.8 | 69.1 | 56.8 |
| + GRPO | 75.7 | 64.5 | 35.5 | 77.8 | 60.1 | 47.3 | 72.0 | 61.8 |
| **PRISM** | 71.0 | 59.5 | 30.6 | 67.5 | 56.3 | 42.8 | 72.6 | 57.2 |
| + GRPO | **77.9** | **68.6** | **45.4** | **82.9** | **64.1** | **49.7** | **74.8** | **66.2** |
| + DAPO | 77.8 | 68.2 | 46.7 | 83.9 | 64.1 | 50.4 | 72.9 | 66.3 |
| + GSPO | 77.5 | 66.6 | 46.7 | 82.3 | 63.2 | 51.1 | 72.9 | 65.8 |
| **Qwen3-VL-8B** | | | | | | | | |
| Instruct | 76.0 | 62.4 | 43.7 | 71.7 | 65.6 | 52.3 | 71.6 | 63.3 |
| + SFT | 70.2 | 60.4 | 32.6 | 73.4 | 56.3 | 42.9 | 71.2 | 58.1 |
| + GRPO | 75.9 | 66.9 | 37.1 | 79.7 | 62.6 | 48.8 | 71.9 | 63.3 |
| **PRISM** | 71.4 | 62.2 | 37.1 | 73.1 | 58.4 | 43.4 | 69.5 | 59.3 |
| + GRPO | **78.3** | **71.3** | **52.0** | **86.4** | **66.6** | **53.3** | **77.2** | **69.3** |

**关键发现**: PRISM+GRPO 相比 SFT→GRPO 在 4B 模型上提升 +4.4 平均分，在 8B 模型上提升 +6.0 平均分。对齐阶段改善分布而非立即提升准确率。

### Table 2: 消融实验（Qwen3-VL-4B）

| Setting | MathVista | MathVerse | MathVision | WeMath | MMMU | MMMU-Pro | HallusionBench | Avg |
|---------|-----------|-----------|------------|--------|------|----------|----------------|-----|
| PRISM (full) | 77.9 | 68.6 | 45.4 | 82.9 | 64.1 | 49.7 | 74.8 | 66.2 |
| Dense 4B disc. | 74.6 | 63.7 | 41.8 | 76.9 | 61.3 | 47.1 | 74.0 | 62.8 |
| Text-only disc. | 74.0 | 59.5 | 42.8 | 76.8 | 62.7 | 48.5 | 71.6 | 62.3 |
| w/o SFT | 62.4 | 47.6 | 25.9 | 55.7 | 51.4 | 36.5 | 66.1 | 49.4 |
| w/o Alignment | 75.7 | 64.5 | 35.5 | 77.8 | 60.1 | 47.3 | 72.0 | 61.8 |
| SFT-107K | 72.3 | 67.0 | 43.1 | 76.9 | 60.6 | 49.0 | 68.3 | 62.5 |
| SFT-1.37M | 77.9 | 68.6 | 45.4 | 82.9 | 64.1 | 49.7 | 74.8 | 66.2 |

**关键发现**:
- MoE 判别器: 使用密集判别器平均下降 3.4 分
- 三阶段流水线: 无对齐下降 4.4 分；无 SFT 下降 16.8 分
- 视觉-语言判别器至关重要: 纯文本判别器导致"鹦鹉对齐"
- SFT 数据规模: 仅 107K 样本平均下降 3.7 分

### Table 3: 训练超参数

| Component | SFT | PRISM | RLVR |
|-----------|-----|-------|------|
| Optimizer | AdamW | AdamW | AdamW |
| Learning Rate | 1e-5 | 1e-6 | 1e-6 |
| Epochs/Steps | 1 epoch | 500 steps | 1500 steps |
| Global Batch Size | 2 | 4 | 32 |
| Max Response Length | 8192 | 6144 | 8192 |
| Rollout Temperature | – | 1.0 | 1.0 |
| Group Size N | – | 16 | 16 |
| α (MoE weight) | – | 0.5 | – |
| Accuracy Reward Weight | – | – | 0.8 |
| Format Reward Weight | – | – | 0.2 |

### Table 4: 数据组成

| Stage | Source | Samples |
|-------|--------|---------|
| SFT | Gemini 3 Flash (curated) | 107K |
| SFT | Public demonstrations | 1.26M |
| Alignment | Gemini 3 Flash (curated) | 6K |
| RLVR | Difficulty-filtered subset | 2K |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Math Benchmarks | 4 datasets | MathVista, MathVerse, MathVision, WeMath | 数学推理测试 |
| General Benchmarks | 3 datasets | MMMU, MMMU-Pro, HallusionBench | 通用能力测试 |
| SFT Corpus | 1.37M | 107K 精选 + 1.26M 公开演示 | 监督微调 |
| Alignment Data | 6K | 最高质量精选 | 分布对齐 |
| RLVR Data | 2K | 难度过滤（通过率 0.2-0.8） | 强化学习 |

### 实现细节

- **Backbone**: Qwen3-VL-4B, Qwen3-VL-8B
- **MoE 判别器**: Qwen3-VL-MoE（4 × Qwen3-VL-2B，top-2 路由）
- **监督模型**: Gemini 3 Flash
- **SFT**: 1 epoch, lr=1e-5
- **Alignment**: 500 steps, lr=1e-6
- **RLVR**: 1500 steps, lr=1e-6
- **硬件**: 未明确说明

### 可视化结果

- 感知专家训练早期快速达到峰值并收敛
- 推理专家上升更缓慢，波动更大
- 对齐显著减少策略与监督分布之间的结构性差异（推理步数、描述项数）
- PRISM+GRPO 在 Token 效率上优于 baseline

---

## 批判性思考

### 优点

1. **问题洞察深入**: 识别并显式解决 SFT 引入的分布漂移问题，而非简单地在 RL 阶段修补
2. **架构设计合理**: MoE 判别器针对多模态推理的异质性设计，感知和推理专家提供解耦反馈
3. **方法通用性强**: 与多种 RL 算法（GRPO、DAPO、GSPO）兼容，无需教师 logits

### 局限性

1. **额外训练成本**: 对齐阶段和 MoE 判别器增加训练开销
2. **依赖结构化格式**: MoE 设计依赖显式的描述/推理分解，对非结构化输出可能不适用
3. **分布对齐代理**: 使用结构性代理（推理步数、描述项数）而非直接分布度量

### 潜在改进方向

1. 探索无需显式结构分解的自适应专家分配
2. 研究更直接的分布对齐度量方法
3. 扩展到更多模态（音频、视频）的感知-推理解耦

### 可复现性评估

- [x] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整
- [x] 数据集可获取（公开演示部分）

---

## 关联笔记

### 基于

- [[GRPO]]: RLVR 阶段使用的强化学习算法
- [[On-Policy Distillation]]: 分布对齐的核心技术
- [[Bradley-Terry Model]]: 判别器训练的基础

### 对比

- [[VOLD]]: 同样使用在线策略蒸馏，但将蒸馏耦合在 RL 阶段内
- [[DeepSeek-R1]]: RLVR 的代表性工作，但未解决 SFT 分布漂移

### 方法相关

- [[Mixture-of-Experts]]: 判别器架构
- [[Adversarial Training]]: 对抗性蒸馏方法
- [[Distribution Alignment]]: 核心问题

---

## 速查卡片

> [!summary] PRISM: Pre-alignment via Black-Box On-Policy Distillation
> - **核心**: 三阶段流水线（SFT → 分布对齐 → RLVR）解决多模态模型分布漂移
> - **方法**: MoE 判别器（感知专家 + 推理专家）进行对抗性在线策略蒸馏
> - **结果**: Qwen3-VL-4B/8B 上相比 baseline 提升 +4.4/+6.0 平均分
> - **代码**: https://github.com/XIAO4579/PRISM

---

*笔记创建时间: 2025-05-08*
