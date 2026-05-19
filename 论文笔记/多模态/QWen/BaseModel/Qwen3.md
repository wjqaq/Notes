---
title: "Qwen3 Technical Report"
method_name: "Qwen3"
authors: [An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao]
year: 2025
venue: arXiv
tags: [llm, mixture-of-experts, reasoning, multilingual, post-training, thinking-mode, strong-to-weak-distillation, reinforcement-learning]
zotero_collection: 多模态/QWen/BaseModel
image_source: local
created: 2025-05-18
---

# 论文笔记：Qwen3 Technical Report

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Qwen Team (Alibaba) |
| 日期 | May 2025 |
| 项目主页 | https://huggingface.co/Qwen |
| 对比基线 | [[DeepSeek-V3]], [[DeepSeek-R1]], [[Llama-4]], [[GPT-4o]], [[Gemini-2.5-Pro]] |
| 链接 | [arXiv](https://arxiv.org/abs/2505.09388) / [Code](https://github.com/QwenLM/Qwen3) / [ModelScope](https://modelscope.cn/organization/qwen) |

---

## 一句话总结

> Qwen3 将思考模式与非思考模式统一到单一模型中，支持 thinking budget 控制，预训练 36T tokens 覆盖 119 种语言，旗舰 MoE 模型 235B 参数 / 22B 激活，性能全面对标 GPT-4o 和 o1。

---

## 核心贡献

1. **Thinking/Non-Thinking 双模式统一**: 一个模型同时具备快响应（非思考）和深度推理（思考）能力，通过 `/think` 和 `/no_think` 指令动态切换，无需部署两个模型。
2. **Thinking Budget 机制**: 用户可通过设定思考 token 预算控制推理深度与延迟的权衡，预算越大性能越好且平滑提升。
3. **Strong-to-Weak Distillation**: 轻量模型（0.6B-30B-A3B）通过 off-policy + on-policy 蒸馏从大模型学习，仅需 RL 训练的 1/10 GPU 小时，且 pass@64 更高。
4. **多语言扩展**: 从 Qwen2.5 的 29 种语言扩展到 119 种语言和方言，显著增强跨语言能力。
5. **MoE 架构改进**: 去除共享专家、引入 global-batch load balancing loss、fine-grained expert segmentation（128 专家 / 8 激活），以更少参数超越 DeepSeek-V3。

---

## 问题背景

### 要解决的问题
现有 LLM 存在两类模型：chat-optimized（快速但推理弱）和 dedicated reasoning（推理强但慢），用户需要切换模型。Qwen3 旨在统一两者。

### 现有方法的局限
- [[GPT-4o]] 等 chat 模型缺乏深度推理能力
- [[QwQ-32B]] 等推理模型在简单任务上也产生冗长思考
- 部署两套模型增加成本和复杂性

### 本文的动机
通过后训练将两种模式融合到一个模型中，用户按需选择，且可通过 thinking budget 灵活控制。

---

## 方法详解

### 模型架构

Qwen3 系列包含 6 个 [[Dense Model|Dense]] 模型（0.6B, 1.7B, 4B, 8B, 14B, 32B）和 2 个 [[Mixture of Experts|MoE]] 模型（30B-A3B, 235B-A22B）。

**Dense 架构**（与 [[Qwen2.5]] 类似）:
- **Attention**: [[Grouped Query Attention|GQA]]（Q/KV head ratio 因模型规模而异），引入 [[QK-Norm]] 稳定训练
- **FFN**: [[SwiGLU]] 激活
- **位置编码**: [[RoPE|Rotary Position Embeddings]]（长上下文阶段 ABF 将 base frequency 从 10k 提升到 1M）
- **归一化**: [[RMSNorm]] with pre-normalization
- **移除**: Qwen2 中的 QKV-bias
- **Tokenizer**: byte-level [[Byte-Pair Encoding|BBPE]]，词汇量 151,669

**MoE 架构增强**:
- [[Fine-grained Expert Segmentation]]: 128 个专家，每 token 激活 8 个
- 移除共享专家（不同于 [[Qwen2.5-MoE]]）
- [[Global-batch Load Balancing Loss]] 促进专家专业化

**架构参数总览**:

| Models | Layers | Heads (Q/KV) | Tie Embedding | Context Length |
|--------|--------|-------------|---------------|----------------|
| Qwen3-0.6B | 28 | 16/8 | Yes | 32K |
| Qwen3-1.7B | 28 | 16/8 | Yes | 32K |
| Qwen3-4B | 36 | 32/8 | Yes | 128K |
| Qwen3-8B | 36 | 32/8 | No | 128K |
| Qwen3-14B | 40 | 40/8 | No | 128K |
| Qwen3-32B | 64 | 64/8 | No | 128K |

| Models | Layers | Heads (Q/KV) | Experts (Total/Activated) | Context Length |
|--------|--------|-------------|--------------------------|----------------|
| Qwen3-30B-A3B | 48 | 32/4 | 128/8 | 128K |
| Qwen3-235B-A22B | 94 | 64/4 | 128/8 | 128K |

### 预训练

#### 预训练数据
- **总量**: 36T tokens，119 种语言和方言
- **数据构造**:
  - 用 [[Qwen2.5-VL]] 对大量 PDF 做 OCR 提取文本
  - 用 [[Qwen2.5-Math]] 和 [[Qwen2.5-Coder]] 合成数学和代码数据
  - 多语言数据标注系统标注 30T+ tokens 的多维标签（教育价值、领域、安全性）
- **数据配比**: instance-level 优化（vs. source/domain level）

#### 三阶段预训练

1. **General Stage (S1)**: 30T+ tokens, seq len 4,096，建立语言能力和通用知识
2. **Reasoning Stage (S2)**: ~5T tokens, seq len 4,096，增加 STEM/代码/推理/合成数据比例，加速学习率衰减
3. **Long Context Stage (S3)**: 数千亿 tokens, seq len 32,768
   - 75% 文本 16K-32K，25% 文本 4K-16K
   - 用 [[ABF]]（Adjusted Base Frequency）将 RoPE base 从 1 万提到 100 万
   - 引入 [[YARN]] 和 [[Dual Chunk Attention|DCA]] 实现推理时 4 倍长度外推

### 后训练（四阶段）

#### Stage 1: Long-CoT Cold Start
- 数据集覆盖数学、代码、逻辑推理、STEM，每个问题有验证答案或测试用例
- 两阶段过滤：query 过滤（去除 [[Qwen2.5]] 无需 CoT 就能答对的）+ response 过滤（去除错误/重复/猜测/不一致的回复）
- 用 [[QwQ-32B]] 生成候选回复，人工标注筛选
- 目的：建立基础推理模式，不追求即时性能

#### Stage 2: Reasoning RL
- 收集 3,995 个 query-verifier pairs
- 使用 [[GRPO]]（Group Relative Policy Optimization）更新参数
- 大批量 + 高 rollout 数 + off-policy 训练提高样本效率
- 控制模型熵保持稳定增长以平衡 exploration/exploitation
- 例：Qwen3-235B-A22B 的 AIME'24 从 70.1 提升至 85.1（170 步 RL）

#### Stage 3: Thinking Mode Fusion

**SFT 数据构造**:
- Thinking 数据：Stage 2 模型在 Stage 1 查询上的 rejection sampling
- Non-thinking 数据：覆盖编码、数学、指令遵循、多语言、创意写作、角色扮演等

**Chat Template 设计**:

| | Thinking Mode | Non-Thinking Mode |
|---|---|---|
| **User** | `{query} /think` | `{query} /no_think` |
| **Assistant** | `<think>{reasoning}</think>\n\n{response}` | `<think></think>\n\n{response}` |

- 默认 thinking 模式（`/think` 可省略）
- 多轮对话中随机插入 `/think` / `/no_think`，模型遵循最后一个标志
- 非思考模式保留空 `<think></think>` 块确保格式一致性

**Thinking Budget**: 当思考 token 达到用户设定阈值时，插入停止指令 `"Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"`，模型基于已有推理生成最终回复。此能力是 Thinking Mode Fusion 的**自然涌现**，并非显式训练所得。

#### Stage 4: General RL
- 覆盖 20+ 任务类别的奖励系统
- 核心能力：指令遵循、格式遵循、偏好对齐、Agent 能力（多轮工具调用与真实环境交互）、专业场景（RAG 等）
- 三种奖励：Rule-based Reward / Model-based Reward with Reference Answer / Model-based Reward without Reference Answer

#### Strong-to-Weak Distillation（轻量模型专用）

适用于 5 个 Dense（0.6B-14B）和 1 个 MoE（30B-A3B）：

1. **Off-policy Distillation**: 教师模型（32B/235B-A22B）同时生成 `/think` 和 `/no_think` 输出，学生学习推理能力和模式切换
2. **On-policy Distillation**: 学生生成 on-policy 序列，与教师 logits 对齐，最小化 KL 散度

关键结论：蒸馏在 AIME'24 pass@1/pass@64 上全面优于 RL，且仅需 1/10 GPU 小时。

---

## 关键图表

### Figure 1: Post-training Pipeline / 后训练流程

![[Qwen3_fig1_pipeline.png]]

**说明**: Qwen3 系列的后训练流水线。旗舰模型经历四阶段（Long-CoT Cold Start -> Reasoning RL -> Thinking Mode Fusion -> General RL），轻量模型通过 Strong-to-Weak Distillation 从旗舰模型蒸馏能力。

### Figure 2: Thinking Budget Scaling / 思考预算曲线

![[Qwen3_fig2_thinking_budget.png]]

**说明**: Qwen3-235B-A22B 在四个基准（数学、编码、STEM）上随 thinking budget 增加的缩放曲线，性能平滑提升。若进一步扩展输出长度超过 32K，性能预计可继续提升。

---

## 关键公式

### 公式1: 全局批负载均衡损失 ([[Global-batch Load Balancing Loss]])

$$
\mathcal{L}_{balance} = \alpha \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

**含义**: 在整个 global batch 上计算专家负载均衡损失，鼓励 token 均匀分配到各专家，促进专家专业化。

**符号说明**:
- $N$: 专家总数（128）
- $f_i$: 专家 $i$ 被路由到的 token 比例
- $P_i$: 专家 $i$ 的 gating 概率均值
- $\alpha$: 平衡损失权重系数

### 公式2: [[GRPO]]（Group Relative Policy Optimization）

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}(\frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon) A_i \right) \right]
$$

**含义**: 对每个 query 采样 G 个 rollout，使用组内相对优势更新策略。与 PPO 不同，GRPO 无需额外的 critic 模型。

**符号说明**:
- $q$: 输入 query
- $G$: 每个 query 的 rollout 数量
- $o_i$: 第 $i$ 个 rollout 输出
- $A_i$: 组内相对优势函数
- $\epsilon$: clip 超参数

### 公式3: On-policy Distillation 损失

$$
\mathcal{L}_{distill} = \text{KL}\left(p_{\theta_{student}}(y|x) \parallel p_{\theta_{teacher}}(y|x)\right)
$$

**含义**: 通过最小化学生模型与教师模型的输出分布 KL 散度，实现知识迁移。

**符号说明**:
- $\theta_{student}, \theta_{teacher}$: 学生和教师模型参数
- $x$: 输入 prompt
- $y$: 输出序列

---

## 实验

### 数据集

| 数据集 | 任务类型 | 用途 |
|--------|---------|------|
| MMLU / MMLU-Pro / MMLU-Redux | 通用知识 | 预训练+后训练评估 |
| BBH / SuperGPQA | 推理/专业知识 | 预训练+后训练评估 |
| GPQA-Diamond | 研究生级科学 | 后训练评估 |
| GSM8K / MATH / MATH-500 | 数学推理 | 预训练+后训练评估 |
| AIME'24 / AIME'25 | 竞赛数学 | 后训练评估 |
| HumanEval / MBPP / MultiPL-E | 代码生成 | 预训练评估 |
| LiveCodeBench v5 / CodeForces | 代码竞赛 | 后训练评估 |
| BFCL v3 | 函数调用/Agent | 后训练评估 |
| MGSM / MMMLU / INCLUDE / Belebele | 多语言 | 预训练+后训练评估 |
| IFEval / Arena-Hard / AlignBench | 指令遵循/对齐 | 后训练评估 |
| RULER | 长上下文 | 后训练评估 |
| Multi-IF / MT-AIME2024 / PolyMath / MLogiQA | 多语言专项 | 后训练评估 |
| ZebraLogic / AutoLogi | 逻辑推理 | 后训练评估 |

### 实现细节

- **Backbone**: Dense / MoE Transformer（自研架构）
- **Tokenizer**: BBPE, vocab 151,669
- **Context Length**: 32K（训练），128K（推理，YARN + DCA 外推）
- **预训练数据**: 36T tokens
- **优化器**: 基于 scaling law 预测的最优超参数
- **推理采样**: Thinking: T=0.6, top_p=0.95, top_k=20; Non-thinking: T=0.7, top_p=0.8, top_k=20, presence_penalty=1.5
- **输出长度**: 默认 32,768 tokens; AIME 扩展至 38,912

### 关键预训练结果（Base Model）

#### Qwen3-235B-A22B-Base vs Baselines

| Benchmark | Qwen2.5-72B (Dense) | DeepSeek-V3 (671B/37B MoE) | Qwen3-235B-A22B (235B/22B MoE) |
|-----------|---------------------|----------------------------|--------------------------------|
| MMLU | 86.06 | 87.19 | **87.81** |
| MMLU-Pro | 58.07 | 59.84 | **68.18** |
| BBH | 86.30 | 86.22 | **88.87** |
| GPQA | 45.88 | 41.92 | **47.47** |
| MATH | 62.12 | 62.62 | **71.84** |
| EvalPlus | 65.93 | 63.75 | **77.60** |

- 以约 1/3 总参数和 2/3 激活参数超越 DeepSeek-V3
- 在 14/15 基准上超越 DeepSeek-V3

#### Qwen3-32B-Base 亮点
- Qwen3-32B 在 10/15 基准上超越 Qwen2.5-72B（仅一半参数）
- MMLU-Pro: 65.54（Qwen2.5-32B 为 55.10）
- 全面超越 Llama-4-Scout（109B MoE）

### 关键后训练结果（Instruct Model）

#### Qwen3-235B-A22B Thinking Mode

| Benchmark | o1 | DeepSeek-R1 | Gemini 2.5 Pro | Qwen3-235B-A22B |
|-----------|-----|-------------|----------------|-----------------|
| AIME'24 | 74.3 | 79.8 | 92.0 | **85.7** |
| AIME'25 | 79.2 | 70.0 | 86.7 | **81.5** |
| LiveCodeBench v5 | 63.9 | 64.3 | 70.4 | **70.7** |
| CodeForces Rating | 1891 | 2029 | 2001 | **2056** |
| BFCL v3 | 67.8 | 56.9 | 62.9 | **70.8** |

#### Qwen3-235B-A22B Non-Thinking Mode

| Benchmark | GPT-4o | DeepSeek-V3 | Qwen3-235B-A22B |
|-----------|--------|-------------|-----------------|
| Arena-Hard | 85.3 | 85.5 | **96.1** |
| AIME'24 | 11.1 | 39.2 | **40.1** |
| CodeForces Rating | 864 | 1134 | **1387** |

- Non-thinking 模式在 18/23 基准上超越 GPT-4o

#### Qwen3-32B 亮点
- Thinking: 17/23 基准超越 QwQ-32B，与 o3-mini(medium) 持平
- Non-thinking: 几乎全面超越 Qwen2.5-72B-Instruct

### 消融实验：各后训练阶段影响（Table 22，Qwen3-32B）

| Benchmark | Stage 2 (Thinking) | Stage 3 (Thinking/Non-thinking) | Stage 4 (Thinking/Non-thinking) |
|-----------|-------------------|--------------------------------|--------------------------------|
| LiveBench | 68.6 | 70.9 / 57.1 | 74.9 / 59.8 |
| Arena-Hard | 86.8 | 89.4 / 88.5 | 93.8 / 92.8 |
| AIME'24 | 83.8 | 81.9 / 28.5 | 81.4 / 31.0 |
| ThinkFollow* | - | 88.7 | 98.9 |
| ToolUse* | 63.3 | 70.4 / 73.2 | 85.5 / 86.5 |

- Stage 3 成功融合 non-thinking 模式，但 AIME 等硬推理任务在 thinking 模式下有轻微退化
- Stage 4 增强通用能力但进一步轻微牺牲硬推理上限（作者选择接受此权衡）

### 蒸馏 vs RL（Table 21，Qwen3-8B）

| Method | AIME'24 (pass@1/pass@64) | AIME'25 (pass@1/pass@64) | GPU Hours |
|--------|--------------------------|--------------------------|-----------|
| Off-policy Distill | 55.0 (90.0) | 42.8 (83.3) | - |
| + RL | 67.6 (90.0) | 55.5 (83.3) | 17,920 |
| + On-policy Distill | **74.4 (93.3)** | **65.5 (86.7)** | **1,800** |

### 长上下文（RULER Benchmark）

- Non-thinking: Qwen3 各尺寸全面超越同量级 Qwen2.5
- Thinking: 略有退化（推理内容对检索任务帮助有限，可能干扰检索过程），承诺未来改进

---

## 批判性思考

### 优点
1. **模式统一设计优雅**: 一个模型两种模式 + thinking budget，部署简化且用户体验灵活
2. **蒸馏效率极高**: 仅需 1/10 GPU 小时就超越 RL，且 pass@64 更高（说明蒸馏保留了探索能力）
3. **多语言覆盖广**: 119 种语言和方言 + 全面的多语言评测，实际可部署性极强
4. **开源生态完整**: Apache 2.0 许可，全系列从 0.6B 到 235B 全部开源
5. **thinking budget 自然涌现**: 无需显式训练即可按预算截断思考，说明模式融合训练方式的有效性

### 局限性
1. **Thinking 模式在通用 RL 后推理退化**: AIME 等硬任务在 Stage 3/4 后性能下降，说明通用能力与专用推理能力存在 trade-off
2. **Thinking 模式在长上下文任务中反而变差**: RULER 上 thinking 模式不如 non-thinking 模式
3. **缺乏与 o3/o4-mini 等最新模型的对比**: 主要对比 o1 和 o3-mini，未涉及更新模型
4. **训练细节有限**: 未公开具体数据配比、超参数、scaling law 推导过程
5. **缺乏多模态**: 仅文本模型，未像 Qwen2.5 系列那样提供 VL 变体（需等待 Qwen3-VL）

### 潜在改进方向
1. 解耦推理能力与通用能力的后训练，减少 trade-off
2. 提升 thinking 模式在检索/长上下文任务中的有效性
3. 引入多模态版本 Qwen3-VL
4. 扩展输出长度限制以进一步提升 thinking budget 收益

### 可复现性评估
- [x] 代码开源（GitHub: QwenLM/Qwen3）
- [x] 预训练模型（HuggingFace / ModelScope 全系列开放）
- [ ] 训练细节完整（部分缺失：数据配比细节、RL 超参数）
- [ ] 数据集可获取（训练数据未公开，仅描述构造方法）

---

## 关联笔记

### 基于
- [[Qwen2.5]]: 直接前身，共享基础架构，Qwen3 在此基础上大幅扩展
- [[Qwen2.5-VL]]: 用于预训练数据的 PDF OCR 提取
- [[Qwen2.5-Math]] / [[Qwen2.5-Coder]]: 用于合成数学和代码训练数据
- [[QwQ-32B]]: 用于 Long-CoT Cold Start 阶段生成候选回复

### 对比
- [[DeepSeek-V3]]: 主要对标 MoE 模型（671B/37B vs 235B/22B）
- [[DeepSeek-R1]]: 推理模型对比
- [[Llama-4]]: 同期开源 MoE 对比
- [[GPT-4o]]: 闭源非推理模型对标
- [[Gemini-2.5-Pro]]: 闭源推理模型对标

### 方法相关
- [[Mixture of Experts]]: MoE 架构核心
- [[Grouped Query Attention]]: GQA 注意力
- [[GRPO]]: Reasoning RL 阶段使用的强化学习算法
- [[Strong-to-Weak Distillation]]: 轻量模型的核心训练方式
- [[Thinking Mode Fusion]]: SFT 阶段融合两种模式
- [[Thinking Budget]]: 推理预算控制机制
- [[Global-batch Load Balancing Loss]]: MoE 专家负载均衡
- [[Fine-grained Expert Segmentation]]: MoE 细粒度专家

### 硬件/数据相关
- [[RoPE]]: 旋转位置编码
- [[YARN]]: 长上下文外推
- [[Dual Chunk Attention]]: 两倍序列长度推理
- [[BBPE]]: Byte-level BPE tokenizer

---

## 速查卡片

> [!summary] Qwen3 Technical Report
> - **核心**: 统一 Thinking/Non-Thinking 双模式 + Thinking Budget 控制
> - **方法**: Dense + MoE, 36T tokens 预训练, 4 阶段后训练, Strong-to-Weak Distillation
> - **结果**: 旗舰 235B 全面对标 GPT-4o/o1, 32B 超越 QwQ-32B 和 Qwen2.5-72B
> - **代码**: https://github.com/QwenLM/Qwen3

---

*笔记创建时间: 2025-05-18*



