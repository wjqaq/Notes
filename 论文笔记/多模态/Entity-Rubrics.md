---
title: "Editor's Choice: Evaluating Abstract Intent in Image Editing through Atomic Entity Analysis"
method_name: "Entity-Rubrics"
authors: [Mor Ventura, Roy Hirsch, Yonatan Bitton, Regev Cohen, Roi Reichart]
year: 2026
venue: arXiv
tags: [image-editing, abstract-instructions, evaluation-benchmark, vlm-judge, instruction-following, entity-level-evaluation, benchmark-dataset]
zotero_collection: 多模态
image_source: online
created: 2026-05-18
---

# 论文笔记：Editor's Choice: Evaluating Abstract Intent in Image Editing through Atomic Entity Analysis

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Technion - Israel Institute of Technology; Google Research |
| 日期 | May 2026 |
| 项目主页 | [venturamor.github.io/EditorsChoice/](https://venturamor.github.io/EditorsChoice/) |
| 对比基线 | [[InstructPix2Pix]], [[MagicBrush]], [[EMU-Edit]], [[AnyEdit]], [[SmartEdit]], [[EditWorld]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.14842) / [Code & Data](https://venturamor.github.io/EditorsChoice/) |

---

## 一句话总结

> 首次形式化抽象图像编辑的定义与评估，提出 Entity-Rubrics 实体级评估框架和 AbstractEdit 基准数据集，揭示现有模型在抽象指令理解上的根本性不足。

---

## 核心贡献

1. **抽象图像编辑的形式化定义**: 沿 Identification（编辑什么）和 Specificity（怎么编辑）两个正交轴，定义了编辑自由度（eDoF）概念，将编辑意图分为 Explicit、Implicit、Abstract 三类
2. **Entity-Rubrics 评估框架**: 受 NLP 原子事实评估启发，将图像编辑评估分解为实体级别的三步流程（实体检测 -> 实体排序 -> 最终评分），与人类判断达到 Spearman's rho=0.66
3. **AbstractEdit 基准数据集**: 首个专门针对抽象图像编辑的基准，含 470 个人工验证测试样本 + 4k 训练样本，覆盖 Physical、Logical、Emotional、Social 四个域 12 个类别
4. **11 个模型的深度分析**: 闭源模型过度编辑牺牲保真度，开源模型编辑不足；高级 LLM 文本编码器和迭代思维是提升关键

---

## 问题背景

### 要解决的问题
人类自然通过抽象概念（如"氛围"、"季节感"）沟通编辑意图，但现有图像编辑基准几乎只关注显式、字面的指令（如"把玩具变绿"）。抽象指令具有一对多的映射特性，评估极为困难。

### 现有方法的局限
- 现有基准（[[InstructPix2Pix]]、[[MagicBrush]]）聚焦简单显式命令
- 复杂编辑基准（CompBench、SmartEdit、EditWorld）仅处理隐式引用或物理推理，仍是一对一映射
- 现有评估方法（[[CLIP]]、DINO、VIEScore）缺乏可解释的细粒度诊断
- 缺乏解耦高层语义推理与低层生成执行能力的评估手段

### 本文的动机
将 NLP 中的原子事实（Atomic Facts）评估理念迁移到视觉域，以图像实体为原子单位，实现对抽象编辑的精确、可解释评估。利用编辑任务的双重约束（指令遵循 + 上下文保真）缩小解释空间，使抽象指令评估变得可行。

---

## 方法详解

### 关键概念：抽象图像编辑的形式化定义

Entity-Rubrics 框架基于以下定义展开：

**两个正交轴**：
- **Identification**（识别/"编辑什么"）：确定需要修改的具体语义或视觉实体
- **Specificity**（具体化/"怎么编辑"）：确定对实体施加的精确视觉变换

**编辑自由度（[[Editing Degree of Freedom|eDoF]]）**：模型解释指令所需的自主权，与解释集 K 的大小成正比：$\text{eDoF} \propto |K(p|I_c)|$

- **Explicit 编辑**：两轴均明确，|K| ~ 1，一对一映射（如 "turn the toy green"）
- **Implicit 编辑**：需额外知识或检测能力，但仍为一对一映射（如 "remove the man's best friend" 需理解 "best friend" = dog）
- **Abstract 编辑**：至少一轴模糊，|K| >> 1，一对多映射（如 "make the dog look like after a long rainy trip day"）

### 模型架构：Entity-Rubrics 三步评估框架

Entity-Rubrics 采用 **VLM 驱动的三阶段结构化评估** 架构：
- **输入**: 上下文图像 $I_c$ + 编辑后图像 $I_e$ + 抽象指令 $p$
- **核心模块**: 实体检测（[[Entity Detection]]） -> 实体排序（[[Entity Ranking]]） -> 最终评分（[[Final Scoring]]）
- **输出**: 1-10 分评分 + 实体级诊断理由 + 可视化报告

#### 阶段 A：实体检测（Entity Detection）

**设计动机**: 受 NLP 中 "复杂声明分解为原子事实" 的启发，将图像实体视为评估的原子单元

**具体实现**:
- VLM 被提示识别并分类所有实体为三个语义组：
  - **Things**: 离散物体（如 "woman's face", "bench"）
  - **Stuff**: 无定形背景元素（如 "grass", "sky"），来自 COCO-Stuff 分类
  - **Global**: 全局属性（如 "lighting", "saturation"）
- 同时覆盖上下文图像和编辑后图像
- 确保捕捉意图驱动的变化在图像的每一层

#### 阶段 B：实体排序（Entity Ranking）

**设计动机**: 解决预期编辑与实际编辑结果之间的张力，采用 precision-over-recall 策略

**具体实现**（两个子阶段）:

1. **预期变换（Expected Transformation）**: VLM 基于抽象指令为每个实体分配预期状态：
   - **Change**: 该实体必须被修改
   - **Optional Change**: 可选的创造性修改
   - **Preserve**: 必须保持不变

2. **执行对齐（Execution Alignment）**: 对比编辑后图像，VLM 评估每个实体是否发生了预期变化：
   - 检测是否发生变化
   - 定性评估变换质量
   - 生成简洁的修改描述

每个实体获得一个排序分数，衡量观察到的变化与预期变化的一致性。VLM 额外提供全局失败画像，审计缺失的编辑或过度编辑伪影。

#### 阶段 C：最终评分（Final Scoring）

三个主要标准：
1. 模型遵循指令的程度
2. 是否达到预期变化水平
3. 最终图像的整体连贯性

生成 1-10 分评分和书面理由。10 分表示完美对齐，所有必要修改均已执行且保持必要上下文。

---

## 关键公式

### 公式1: [[Editing Degree of Freedom|编辑自由度 eDoF]]

$$
\text{eDoF} \propto |K(p|I_c)|
$$

**含义**: 编辑自由度正比于给定上下文图像 $I_c$ 和文本指令 $p$ 下所有视觉上不同的有效编辑解释集 $K$ 的大小。

**符号说明**:
- $K(p|I_c)$: 在上下文图像约束下，满足文本指令的所有视觉上不同的编辑结果集合
- $p$: 文本编辑指令
- $I_c$: 上下文图像
- eDoF: 模型必须行使的生成性自主权，是连续的谱系而非离散类别

### 公式2: 意图映射分类

$$
|K| \approx 1 \Rightarrow \text{One-to-One (Explicit/Implicit)}
$$

$$
|K| \gg 1 \Rightarrow \text{One-to-Many (Abstract)}
$$

**含义**: 解释集的大小直接决定编辑意图的映射类型，从一对一到一对多。

**符号说明**:
- $|K| \approx 1$: 低 eDoF，严格映射（Explicit/Implicit 编辑）
- $|K| \gg 1$: 高 eDoF，一对多映射（Abstract 编辑）

---

## 关键图表

### Figure 1: Teaser / 抽象图像编辑评估概览

<!-- arXiv HTML not available; figure described from paper text -->

**说明**: 展示 Entity-Rubrics 的核心工作流程。给定上下文图像和抽象指令（"Infuse the scene with empathy and a deep bond with animals"），框架通过分解场景为实体级别评估（Man's Expression, Bull's Pose, Sign, Crowd），生成每实体排名和最终评分 9/10。

### Figure 2: Taxonomy of Image Editing Intent / 编辑意图分类法

**说明**: 形式化编辑意图沿两个正交轴：Identification（"编辑什么"）和 Specificity（"怎么编辑"）。Explicit 和 Implicit 编辑为一对一映射，Abstract 编辑引入一对多关系（Referent 和 Target 均不确定）。

### Figure 3: Entity-Rubrics 三步评估框架

**说明**: (A) 实体检测识别 Things、Stuff、Global 三类实体。(B) 实体排序阶段，VLM 先确定每个实体的预期变换（Change/Optional/Preserve），再评估编辑后图像中的执行对齐。(C) 最终评分聚合为 1-10 分和综合理由。结果通过红（错误）到绿（正确）色标在图像上可视化。

### Figure 4: AbstractEdit 自动策展流程

**说明**: (A) Sourcing: 从 OpenImages 选取 1300 张自然图像，手动定义类别和示例，采样多样化 Persona。(B) Instruction Generation: LLM（Gemini 2.5 Pro）基于 few-shot 和随机 Persona 生成配对的抽象和显式指令。(C) Editing: 两条指令分别应用于上下文图像产生编辑对。

### Figure 5: Prompt Type Comparison / 提示类型对比

**说明**: 柱状图对比 11 个模型在显式（蓝色）和抽象（条纹酒红色）提示下的得分。闭源模型在抽象提示上表现更好（提升最多 9%），而开源模型依赖显式指令但会导致过度编辑——切换到抽象提示减少了过度编辑（平均 -13.3%）但转换为编辑不足。

### Figure 6: Qualitative Example / 定性编辑示例

**说明**: 以 "Upgrade the neighborhood to an ultra-luxury, high-end shopping street" 为例，展示 8 个代表性模型的编辑结果对比。Gemini 3.1 和 GPT-Image-1.5 得分最高（10），而 FLUX.1-Kontext 和 Step1X 得分最低（5 和 3）。

### Figure 7: Failure Rate per Entity Edit Action / 每实体编辑动作的失败率

**说明**: 基于实体排序阶段的分析，Object Count（>30% 失败率）是最大瓶颈，其次是 Perspective 和 Object Presence。Style、Texture、Attribute State 等"表面"编辑可靠处理。结论：**抽象指令遵循的主要瓶颈是结构化推理而非美学调整**。

### Figure 8: Diversity Analysis / 多样性分析

**说明**: 对比 FLUX.2（开源）和 Gemini 3.1（闭源）在抽象 vs 显式提示下的 [[Vendi Score]] 多样性分布。抽象提示始终比显式提示产生更高的平均语义和视觉多样性，闭源模型的差距尤其大——显式指令迫使模型走向狭窄输出，抽象提示释放更广阔的创意诠释。

### Figure 9: Context-Dependent eDoF / 上下文依赖的编辑自由度

**说明**: 同一指令 "Adjust for a younger audience" 在高端餐厅场景（高 eDoF，需创造性合成灯光、装饰、摆盘变化）和黑白填色书场景（低 eDoF，视觉上下文明确指向添加鲜艳色彩）下的不同解释需求。

### Figure 10: Expanded Cases of Editing Intent / 扩展编辑意图案例

**说明**: 扩展 Fig.2，展示更多抽象编辑子类型：Semantic/Contextual、Spatial reference、Detection-based reference、Style-Transfer 等，说明从一对一（Explicit）到一对多（Abstract）的连续谱系。

### Table 1: Comparison of Benchmark Subsets / 基准子集对比

| Benchmark | Subset | Size | Dom | Ctx Img | Nat. | Glob/Loc | eDoF | Example Instruction |
|-----------|--------|------|-----|---------|------|----------|------|---------------------|
| CompBench | Implicit | 100 | L | 100 | No | L | 1:1 | Remove the farthest tiger from the water |
| SmartEdit | Implicit | 60 | M | 30 | No | L | 1:1 | Remove the object that can be used to have meals |
| EditWorld | Logic | 60 | L | 60 | No | - | 1:1 | What happens if a hole appears in the balloon? |
| AnyEdit | Implicit | 10k | L | 10k | No | - | 1:1 | What would happen if the man can't catch the wave? |
| EMU-Edit | Global | 440 | M | 219 | Yes | G | 1:N | Change the scene to night time |
| Kontext | Global | 262 | M | 88 | Yes | G | 1:N | Make this image real |
| **AbstractEdit** | **Full Abs.** | **470** | **H** | **257** | **Yes** | **G,L** | **1:N** | **Make the dog look like after a long rainy trip day** |

**表格说明**: AbstractEdit 在规模、领域多样性、自然语言、编辑自由度和空间范围上全面超越现有基准。

### Table 2: Abstract Instruction Following Performance / 抽象指令遵循性能

| Cat. | Model | Hum↑ | Abs↑ | F. Prof. | Emotional | Logical | Physical | Social |
|------|-------|------|------|----------|-----------|---------|----------|--------|
| **OS** | Qwen-Image-Edit | - | 7.48±2.86 | ← | 8.00±2.89 | 6.89±3.02 | 7.43±2.92 | 8.14±2.41 |
| OS | FLUX.2 [dev] | 8.50 | 7.26±2.83 | ← | 8.29±2.04 | 7.34±2.89 | 6.85±3.24 | 7.16±2.57 |
| OS | HiDream-E1 | - | 5.38±3.15 | ← | 7.91±2.12 | 4.18±3.05 | 4.97±3.23 | 6.56±2.63 |
| OS | FLUX.1-Kontext [dev] | 7.69 | 5.10±3.46 | ← | 6.17±3.48 | 4.34±3.44 | 5.21±3.62 | 5.72±3.16 |
| **OS+Think** | Step1X-Think-Reflect | 7.97 | 6.90±3.20 | ← | 7.63±3.06 | 7.39±3.22 | 5.39±3.64 | 7.14±2.50 |
| OS+Think | Step1X | - | 6.55±3.41 | ← | 7.61±3.06 | 6.90±3.38 | 5.36±3.83 | 6.64±2.97 |
| OS+Think | Bagel-Think | - | 5.80±3.50 | ← | 5.85±3.43 | 4.84±3.67 | 5.97±3.49 | 6.92±2.91 |
| OS+Think | Bagel | 6.61 | 4.45±3.42 | ← | 4.89±3.45 | 3.31±3.21 | 4.90±3.37 | 5.53±3.29 |
| **Closed** | **Gemini 3.1 Flash** | **9.66** | **9.52±1.37** | → | **9.62±1.33** | **9.47±1.34** | **9.43±1.64** | **9.62±1.19** |
| Closed | GPT-Image | 9.67 | 9.34±1.37 | → | 9.49±1.43 | 9.19±1.42 | 9.36±1.46 | 9.49±1.20 |
| Closed | Gemini 3 Pro | - | 9.27±1.78 | → | 9.50±1.40 | 9.59±1.13 | 8.46±2.80 | 9.35±1.42 |
| Closed | Seed-Dream | - | 9.21±1.76 | → | 9.43±1.44 | 9.21±1.71 | 8.88±2.33 | 9.39±1.38 |
| Closed | Gemini 2.5 Flash Image | - | 8.67±2.40 | → | 9.23±1.70 | 9.09±2.04 | 7.57±3.26 | 8.73±2.05 |

**Failure Profile**: (Under-Editing) ← | → (Over-editing)

**表格说明**: 闭源模型平均 9.2 分，远超最佳开源模型 Qwen-Image-Edit (7.48)。闭源模型倾向过度编辑（红色偏差），开源模型倾向编辑不足（紫色偏差）。Thinking 机制显著提升性能（Step1X +5.3%，Bagel +30.3%）。

### Table 3: AbstractEdit Training Set Category Distribution / 训练集类别分布

| Domain | Count | Percentage (%) |
|--------|-------|----------------|
| Social | 1,470 | 35.7% |
| Physical | 1,470 | 35.7% |
| Logical | 882 | 21.4% |
| Emotional | 294 | 7.1% |
| **Total** | **4,116** | **100.0%** |

**表格说明**: Social 和 Physical 域各占训练集最大份额，反映抽象人际交互和物理推理的核心地位。

### Figure 11: Persona-Driven Instruction Diversity / 角色驱动的指令多样性

**说明**: 展示数据生成中使用的 Persona 特征空间（年龄、国家、职业、爱好等 10 个维度）以及领域和类别的指南文档。通过采样不同 Persona 作为独立变量，确保指令的语言风格、文化背景和创作动机高度多样化。

### Figure 12: AbstractEdit Distribution of Categories / 类别分布

**说明**: 展示测试集 470 个样本在 4 个域 12 个类别中的分布。Social 域（Culture 47、Socio-Economic 44、Role 37）和 Physical 域（POV/Composition 37、Season 34 等）占比最大；Emotional 域仅含 Mood/Emotion 类别 41 个样本，专注测试氛围解读能力。

### Figure 13: Explicit vs Abstract Prompt Length / 显式 vs 抽象提示长度

**说明**: 展示两种提示类型在各域的单词数分布。抽象指令平均约 10-11 词（限制 <15 词），显式指令显著更长：Social 域平均 209 词，Logical 域平均 143 词。长度差距量化了"语义鸿沟"——将简洁抽象概念转化为完整视觉步骤所需的信息量。

### Figure 14: Distribution of Common Entities in Context Images / 上下文图像实体分布

**说明**: 展示 AbstractEdit 中 1300 张上下文图像的语义分析。大多数图像含 2-12 种不同实体类型，峰值在 5 种。高频实体以人本中心（Clothing 254 次、Person 232 次、Human body 196 次）。实例数量分布显示大多数图像含 15-35 个边界框实例，峰值 16 个——刻意的高视觉密度引入真实场景干扰物。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| AbstractEdit (test) | 470 samples | 4 域 12 类别，人工验证抽象性 | 测试 |
| AbstractEdit (train) | 4,000 samples | 自动生成 + 显式指令配对 | 训练 |
| Open Images v7 | 1,300 上下文图像 | 复杂多实体场景 | 图像来源 |

### 评估模型（11 个）

**开源模型（8 个）**：[[Qwen-Image-Edit]]、FLUX.2 [dev]、HiDream-E1、FLUX.1-Kontext [dev]、Step1X、Step1X-Think-Reflect、Bagel、Bagel-Think

**闭源模型（5 个）**：Gemini 3.1 Flash、GPT-Image、Gemini 3 Pro、Seed-Dream 4.5、Gemini 2.5 Flash Image

### 实现细节

- **评估 VLM**: Gemini 3 Flash（两步调用：先分析上下文图像建立预期，再对比编辑后图像）
- **API 调用**: 总共 24,440 次基础 API 调用（13 模型 x 470 样本 x 2 提示类型 x 2 步骤）
- **推理硬件**: NVIDIA A100 80GB GPU，开源模型推理约 150 GPU 小时
- **数据生成**: Gemini API，1300 次调用生成每张上下文图像的抽象+显式指令对
- **人类评估**: AMT，60 个分层样本，每个任务 3 名标注者，共 1,080 条响应，quadratic weighted Fleiss' kappa = 0.47

### 关键发现

1. **开源 vs 闭源的提示权衡**: 闭源模型在抽象指令上表现更好（最高提升 9%），开源模型依赖显式指令但会导致过度编辑。切换到抽象提示虽然减少了过度编辑（平均 -13.3%），但转换为编辑不足——无法解读高层意图。

2. **领域差异**: 开源模型在 Emotional 和 Social 域表现更好（均值 6.88），因为情感和社交编辑常映射到强编码的语义模式（颜色偏移、面部表情）。Physical 和 Logical 域需要"多跳"组合推理（均值 5.7）。

3. **失败模式分析**: Object Count（>30% 失败率）是最大瓶颈，其次是 Perspective 和 Object Presence。Style、Texture、Attribute State 等表面编辑可靠。

4. **Thinking 和文本编码器的驱动力**: FLUX.2（Mistral+T5, 7.26）远超 FLUX-Kontext（CLIP, 5.1）；Step1X-Think 比基础版提升 5.3%；Bagel-Think 比基础版提升 30.3%。但 Thinking 模型在显式指令上有"精确性税"——过度思考反而不利于直接执行。

5. **抽象性释放多样性**: [[Vendi Score]] 分析表明，抽象提示始终比显式提示产生更多样化的视觉概念，闭源模型差异尤为显著。

### 可视化结果

图 6 展示了典型编辑案例 "Upgrade the neighborhood to an ultra-luxury, high-end shopping street" 下各模型的表现。Gemini 3.1 和 GPT-Image-1.5 完美完成（10 分），而开源模型如 FLUX.1-Kontext（5 分）和 Step1X（3 分）要么编辑不足要么产生视觉不一致的结果。

---

## 批判性思考

### 优点
1. **定义清晰、系统性**: 首次用两个正交轴（Identification/Specificity）和 eDoF 概念完整形式化抽象图像编辑，文献综述全面且结构良好
2. **评估方法论创新**: 将 NLP 原子事实评估成功迁移到视觉领域，precision-over-recall 策略在高度主观域中实用且合理
3. **人机相关性验证**: Spearman's rho = 0.66，优于 VIEScore (0.54) 和 CLIP (0.41)，有 AMT 实验支撑
4. **诊断性而非黑盒**: 框架不仅给出总分，还提供实体级失败诊断，可直接指导模型改进
5. **全面的实验覆盖面**: 11 个模型（开源+闭源+Thinking变体），4 个域，多种分析维度

### 局限性
1. **VLM 评估噪声**: 当前 VLM 仍偶有复杂场景理解错误，将噪声引入评估管道；评估质量受限于 VLM 能力
2. **跨实体关系被忽略**: 评估聚焦于独立实体属性，但抽象编辑常涉及实体间关系（如"make them look in love"），这属于 $O(n^2)$ 复杂度问题未被处理
3. **文化偏差**: 数据集为英文构建，可能反映西方对"专业"、"氛围"等抽象概念的解释
4. **数据集继承偏差**: 上下文图像来自 OpenImages，其人口统计和地理代表性偏差自然传递到 AbstractEdit
5. **无训练改进验证**: 虽然提供了 4k 训练样本，但未展示在此数据上训练后的模型改进效果

### 潜在改进方向
1. **跨实体关系建模**: 扩展评估框架支持实体间关系的评估，处理"使场景更具浪漫感"这类涉及多实体协同的指令
2. **Agentic 循环集成**: 将 Entity-Rubrics 的细粒度反馈用于测试时 critique-and-revise 循环，结合外部图像检索增强评估
3. **因果链评估**: 论文附录提及的"编辑级联"（earthquake -> cracked ground -> structural damage -> panicked pedestrians）值得深入探索
4. **多语言/跨文化扩展**: 覆盖非英语和非西方文化背景的抽象概念理解
5. **作为奖励模型训练**: 利用 Entity-Rubrics 的实体级评分信号直接优化扩散模型的偏好对齐

### 可复现性评估
- [x] 代码开源（项目页面承诺）
- [x] 数据可获取（项目页面承诺）
- [x] 训练细节完整（附录含参数和算力统计）
- [x] 数据集可获取（AbstractEdit benchmark 公开）

---

## 关联笔记

### 基于
- [[InstructPix2Pix]]: 指令引导图像编辑的奠基性工作
- [[Atomic Facts Evaluation]]: NLP 中原子事实分解评估方法，Entity-Rubrics 的核心灵感来源
- [[Natural Language Inference|NLI]]: 实体评估中的语义对齐理论基础

### 对比
- [[VIEScore]]: 现有 VLM 评估方法，仅提供全局评分，缺乏实体级可解释性
- [[CLIP]]: 传统编码评估，对细粒度语义不敏感，常奖励过度编辑
- [[MagicBrush]]: 早期指令编辑基准，仅覆盖简单显式命令
- [[AnyEdit]]: 大规模隐式编辑基准，但仍为物理推理领域的一对一映射
- [[EMU-Edit]]: 全局编辑基准，但 eDoF 和领域多样性远不如 AbstractEdit

### 方法相关
- [[Entity-Rubrics]]: 核心方法，实体级 VLM 评估框架
- [[Editing Degree of Freedom]]: eDoF 概念，形式化抽象编辑的连续谱系
- [[Vendi Score]]: 用于量化视觉多样性
- [[Iterative Thinking]]: 推理时迭代思考机制，显著提升抽象指令遵循

### 硬件/数据相关
- [[Open Images v7]]: 上下文图像来源
- [[Gemini 2.5 Pro]]: 用于数据生成的 LLM
- [[Gemini 3 Flash]]: 用于评估的 VLM

---

## 速查卡片

> [!summary] Editor's Choice: Evaluating Abstract Intent in Image Editing through Atomic Entity Analysis
> - **核心**: 首次形式化抽象图像编辑（Identification x Specificity 轴 + eDoF 概念），提出 Entity-Rubrics 实体级评估框架和 AbstractEdit 基准
> - **方法**: 三阶段 VLM 评估——实体检测(Things/Stuff/Global) -> 实体排序(Change/Optional/Preserve + 执行对齐) -> 最终评分 1-10
> - **结果**: 闭源模型 9.2 vs 开源模型 7.5（最佳）；高级 LLM 编码器+Thinking 是关键驱动力；Object Count 是最大瓶颈（>30% 失败率）
> - **代码**: https://venturamor.github.io/EditorsChoice/

---

*笔记创建时间: 2026-05-18*
