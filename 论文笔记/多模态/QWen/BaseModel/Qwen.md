---
title: "Qwen Technical Report"
method_name: "Qwen"
authors: [Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, Tianhang Zhu]
year: 2023
venue: arXiv
tags: [llm, pretraining, alignment, code-generation, math-reasoning, rlhf]
zotero_collection: 多模态/QWen/BaseModel
image_source: online
arxiv_html: https://arxiv.org/html/2309.16609
created: 2026-05-18
---

# 论文笔记：Qwen Technical Report

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Qwen Team, Alibaba Group |
| 日期 | September 2023 |
| 项目主页 | https://github.com/QwenLM/Qwen |
| 对比基线 | [[LLaMA]], [[LLaMA 2]], [[ChatGLM2]], [[Baichuan2]], [[InternLM]], [[GPT-3.5]], [[GPT-4]] |
| 链接 | [arXiv](https://arxiv.org/abs/2309.16609) / [GitHub](https://github.com/QwenLM/Qwen) |

---

## 一句话总结

> 阿里通义千问团队开源的首个 [[LLM]] 系列：包含 1.8B/7B/14B 基座模型、Chat/SFT/RLHF 对齐模型，以及代码和数学专用模型，在多项 benchmark 上超越同规模开源模型。

---

## 核心贡献

1. **Qwen 基座模型系列**: 在高达 3 万亿 token 的多语言数据上预训练，含 1.8B/7B/14B 三个规模，在 MMLU、C-Eval、GSM8K、MATH、HumanEval、MBPP、BBH 上全面超越此前同规模 SOTA。
2. **完整的对齐流水线**: [[Supervised Fine-Tuning]] + [[RLHF]]（含 [[Reward Model]] 训练和 [[PPO]] 策略优化），人工评估显示 RLHF 版本显著优于 SFT 版本。
3. **专用模型**: [[Code-Qwen]]（代码继续预训练+微调）和 [[Math-Qwen-Chat]]（数学 SFT），分别在 HumanEval/MBPP 和 GSM8K/MATH 上领先开源模型。
4. **Agent 能力**: 通过 [[Self-Instruct]] 策略使模型具备 [[ReAct Prompting]] 工具调用、[[Code Interpreter]] 代码解释器、Hugging Face Agent 等能力。
5. **长上下文扩展**: 结合 [[NTK-aware Interpolation]]、[[LogN-Scaling]] 和 [[Window Attention]] 实现训练无关的上下文长度扩展至 8192+ tokens。

---

## 问题背景

### 要解决的问题
如何构建一个全面、可复现、开源的大型语言模型系列，使其在通用能力、对齐效果、专业领域（代码、数学）和 Agent 能力上均达到有竞争力的水平。

### 现有方法的局限
- 已有开源 LLM（[[LLaMA]]、[[Falcon]]、[[MPT]] 等）在多语言（尤其是中文）和综合 benchmark 上表现有限
- 对齐方法（SFT/RLHF）对开源社区而言缺乏完整的参考实现和评估
- 代码和数学专用模型通常需要从头预训练，成本高昂且可能丢失通用能力

### 本文的动机
通过系统的预训练、对齐和专用化流程，构建一个包含多规模、多能力的 LLM 系列，并全部开源以促进社区研究和应用。

---

## 方法详解

### 模型架构

Qwen 采用基于 [[Transformer]] 的改进架构：

- **Backbone**: 修改版 [[LLaMA]] 架构（Decoder-only [[Transformer]]）
- **输入/输出嵌入**: [[Untied Embedding]]（输入嵌入与输出投影不共享权重）
- **位置编码**: [[RoPE]]（Rotary Positional Embedding），使用 FP32 精度存储逆频率矩阵
- **归一化**: Pre-Norm + [[RMSNorm]]（替代传统 [[Layer Normalization]]）
- **激活函数**: [[SwiGLU]]（Swish + Gated Linear Unit），FFN 维度从 4x 隐藏大小缩减为 $\frac{8}{3}$ 隐藏大小
- **偏置**: 大多数层移除偏置，但在 [[Attention]] 的 QKV 层保留偏置以增强 [[Length Extrapolation|长度外推]] 能力
- **注意力机制**: [[Flash Attention]]
- **总参数**: 1.8B / 7B / 14B 三个规模

### 模型规格

| 参数量 | 隐藏维度 | 注意力头数 | 层数 | 学习率 | Batch Size | 训练 Token 数 |
|--------|----------|-----------|------|--------|-----------|--------------|
| 1.8B | 2048 | 16 | 24 | $3.0 \times 10^{-4}$ | 4M | 2.2T |
| 7B | 4096 | 32 | 32 | $3.0 \times 10^{-4}$ | 4M | 2.4T |
| 14B | 5120 | 40 | 40 | $3.0 \times 10^{-4}$ | 4M | 3.0T |

### 核心模块

#### Tokenization

- 基于 [[Byte Pair Encoding|BPE]]（从 `tiktoken` 的 `cl100k_base` 词表出发）
- 扩充中文常用字词和其他多语言词汇
- 数字拆分为单个数字（同 [[LLaMA]]）
- 最终词表大小约 152K
- 在多数语言上的压缩率优于 [[LLaMA]]、[[Baichuan2|Baichuan]]、[[InternLM]] 的 tokenizer

#### 预训练数据

- 总计约 3 万亿 token
- 来源：公开网页、百科、书籍、代码等；多语言（大量中英文）
- 数据清洗：HTML 提取、语言识别、精确去重 + [[MinHash]]/[[Locality-Sensitive Hashing|LSH]] 模糊去重
- 质量过滤：规则 + ML 模型评分（语言模型、文本质量、攻击性内容检测）
- 选择性上采样高质量来源
- 预训练中混入高质量指令数据以增强 zero-shot/few-shot 能力
- 与测试集 13-gram 重叠的数据已过滤

#### 上下文长度扩展

采用三种训练无关的技术组合，仅推理时应用：

1. **[[NTK-aware Interpolation|动态 NTK-aware 插值]]**: 不等比缩放 [[RoPE]] 各维度，避免高频信息丢失；按 chunk 动态调整缩放因子
2. **[[LogN-Scaling]]**: 根据上下文长度与训练长度的比值缩放 Q-K 点积，保持注意力熵稳定
3. **[[Window Attention]]**: 限制注意力到有限窗口，且按层分配不同窗口大小（低层短窗口、高层长窗口），因为低层对长度扩展更敏感

组合使用后，14B 模型在 16384 长度上 PPL 从 3168.35 降至 3.42。

#### 对齐：SFT

- **数据**: 多风格对话标注（含安全相关数据）、[[ChatML]] 格式标注角色和信息类型
- **训练**: 下一 token 预测，对系统和用户输入施加 loss mask
- **优化器**: [[AdamW]] ($\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$)
- **超参**: 序列长度 2048，batch size 128，4000 步，warmup 1430 步，峰值学习率 $2 \times 10^{-6}$，weight decay 0.1，dropout 0.1，梯度裁剪 1.0

#### 对齐：RLHF

**Reward Model 训练**:
- 先 [[Preference Model Pretraining|PMP]]（偏好模型预训练）在大规模比较数据上，再在高质量标注数据上微调
- 标注系统含约 6600 个细粒度标签，平衡采样确保多样性和复杂性
- 使用不同规模和采样策略的 Qwen 模型生成多样化回复
- 基于相同规模预训练 Qwen 初始化，添加 pooling 层提取句子级别 reward
- 学习率 $3 \times 10^{-6}$，batch size 64，序列长度 2048，训练 1 epoch

**PPO 训练**:
- 四模型架构：policy model、value model、reference model、reward model
- 先单独更新 value model 50 步以适应 reward model
- 每 query 同时采样 2 个回复
- KL 散度系数 0.04，reward 按 running mean 归一化
- Policy LR: $1 \times 10^{-6}$，Value LR: $5 \times 10^{-6}$，value loss clipping 0.15
- 推理时 top-p = 0.9
- 使用预训练梯度（pretrained gradient）缓解 [[Alignment Tax|对齐税]]

#### Agent 能力

通过 [[Self-Instruct]] 策略迭代生成高质量样本（最终约 2000 条）：

1. 利用 Qwen 自身的 in-context learning 能力生成 [[ReAct Prompting|ReAct]] 格式样本
2. 规则 + 人工筛选去噪
3. 混入通用 SFT 数据共同训练，保留通用能力

---

## 关键公式

### 公式1: [[Autoregressive Language Modeling|自回归语言建模]]

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)
$$

**含义**: 标准的下一个 token 预测损失，用于预训练和 SFT。

**符号说明**:
- $x_t$: 序列中第 $t$ 个 token
- $x_{<t}$: 前 $t-1$ 个 token 构成的上下文
- $\theta$: 模型参数
- $T$: 序列长度

### 公式2: [[RoPE|旋转位置编码]]

$$
\langle f_q(x_m, m), f_k(x_n, n) \rangle = g(x_m, x_n, m - n)
$$

**含义**: RoPE 通过旋转变换将相对位置信息编码到注意力计算中，使注意力得分仅依赖于 token 间的相对位置。

**符号说明**:
- $x_m, x_n$: 位置 $m$ 和 $n$ 处的输入向量
- $f_q, f_k$: 施加旋转变换后的 query 和 key 表示
- $m - n$: 相对位置

### 公式3: [[SwiGLU|SwiGLU 激活函数]]

$$
\text{SwiGLU}(x) = \text{Swish}(xW_1 + b_1) \odot (xW_2 + b_2)
$$

其中

$$
\text{Swish}(x) = x \cdot \sigma(x)
$$

**含义**: SwiGLU 结合了 Swish 激活和门控线性单元，在保持非线性的同时通过门控机制控制信息流。

**符号说明**:
- $W_1, W_2$: 前馈网络的权重矩阵
- $\odot$: 逐元素乘法
- $\sigma$: Sigmoid 函数
- 为补偿参数量，FFN 隐藏维度从 $4d$ 缩减为 $\frac{8}{3}d$

### 公式4: [[PPO|PPO 目标函数]]

$$
\mathcal{L}^{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right] - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

**含义**: PPO 通过 clipped surrogate objective 限制策略更新幅度，并加入 KL 惩罚项防止模型偏离参考策略过远。

**符号说明**:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$: 新旧策略的概率比
- $\hat{A}_t$: 优势函数估计
- $\epsilon$: clip 范围（本文设为 0.15）
- $\beta$: KL 散度系数（本文设为 0.04）
- $\pi_{\text{ref}}$: 参考模型（SFT 模型）

### 公式5: [[LogN-Scaling|对数缩放注意力]]

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k} \cdot \frac{\log n}{\log L}} \right) V
$$

**含义**: LogN-Scaling 将 Q-K 点积除以一个与序列长度相关的因子，使注意力熵在长序列时保持稳定。

**符号说明**:
- $n$: 当前推理时的序列长度
- $L$: 训练时的序列长度（2048）
- $d_k$: key 的维度

---

## 关键图表

### Figure 1: Model Lineage / Qwen 系列模型谱系

<!-- 图片说明：展示从 Qwen 基座模型出发，通过 SFT/RLHF 派生出 Qwen-Chat、Qwen-Chat-RLHF，进一步派生 Code-Qwen 系列、Math-Qwen-Chat 系列、Qwen-VL 系列 -->

**说明**: Qwen 系列的模型演化关系。从 Qwen 预训练基座模型出发：(1) 通过 SFT 得到 Qwen-Chat；(2) 通过 RLHF（PMP + RM + PPO）得到 Qwen-Chat-RLHF；(3) 基于 Qwen 继续代码预训练得到 Code-Qwen，再 SFT 得到 Code-Qwen-Chat；(4) 基于 Qwen 数学 SFT 得到 Math-Qwen-Chat；(5) 此前开源的 Qwen-VL 和 Qwen-VL-Chat 也基于 Qwen 基座模型。

### Figure 2: Performance Overview / 综合性能概览

<!-- 图片说明：雷达图展示 GPT-4、GPT-3.5、此前 13B SOTA 和 Qwen-14B 在 12 个数据集上的表现 -->

**说明**: Qwen-14B 在 12 个覆盖语言理解、知识、推理等领域的数据集上与 GPT-4、GPT-3.5、此前 13B SOTA 的对比。Qwen 显著超越此前同规模 SOTA，但与 GPT-3.5 和 GPT-4 仍有差距。

### Figure 3: Encoding Compression Rates / 编码压缩率对比

<!-- 图片说明：柱状图对比 LLaMA-7B、Baichuan-7B、ChatGLM2-6B、InternLM-7B 和 Qwen 在 18 种语言 + 代码上的压缩率 -->

**说明**: 以 [[XLM-R]] 为基准（值=1），Qwen 在保证中英文和代码高效解码的同时，在许多其他语言（泰语、希伯来语、阿拉伯语、韩语、越南语、日语、土耳其语等）上也实现了高压缩率，降低了多语言服务的推理成本。

### Figure 4: Human Evaluation / 人类评估结果

<!-- 图片说明：柱状图展示 Qwen-7B-Chat(SFT)、Qwen-14B-Chat(SFT)、Qwen-14B-Chat(RLHF) 和 GPT-4 相对于 GPT-3.5 的胜/平/负率 -->

**说明**: 在 300 条中文指令（知识、语言理解、创意写作、编码、数学）上对比 GPT-3.5 的人工评估。Qwen-14B-Chat(RLHF) 在平均胜率上显著优于 SFT 版本，但仍落后于 GPT-4。RLHF 对创意写作等任务的提升尤为明显。

### Table 1: Model Sizes, Architecture, and Optimization Hyper-parameters

| 参数量 | 隐藏维度 | 注意力头 | 层数 | 学习率 | Batch Size | 训练 Token |
|--------|----------|---------|------|--------|-----------|-----------|
| 1.8B | 2048 | 16 | 24 | $3.0 \times 10^{-4}$ | 4M | 2.2T |
| 7B | 4096 | 32 | 32 | $3.0 \times 10^{-4}$ | 4M | 2.4T |
| 14B | 5120 | 40 | 40 | $3.0 \times 10^{-4}$ | 4M | 3.0T |

**说明**: 三个规模的 Qwen 基座模型的结构和训练超参数。

### Table 2: Overall Performance on Widely-Used Benchmarks (Base Models)

| Model | Params | MMLU 5-shot | C-Eval 5-shot | GSM8K 8-shot | MATH 4-shot | HumanEval 0-shot | MBPP 3-shot | BBH 3-shot |
|-------|--------|-------------|---------------|--------------|-------------|------------------|-------------|------------|
| LLaMA | 65B | 63.7 | 40.4 | 54.4 | 10.6 | 23.7 | 37.7 | 58.4 |
| LLaMA 2 | 70B | 69.8 | 50.1 | 63.3 | 13.5 | 29.9 | 45.0 | 64.9 |
| **Qwen** | **1.8B** | **44.6** | **54.7** | **21.2** | **5.6** | **17.1** | **14.8** | **28.2** |
| **Qwen** | **7B** | **58.2** | **63.5** | **51.7** | **11.6** | **29.9** | **31.6** | **45.0** |
| **Qwen** | **14B** | **66.3** | **72.1** | **61.3** | **24.8** | **32.3** | **40.8** | **53.4** |

**说明**: Qwen-14B 在所有 7 个 benchmark 上超越此前 13B SOTA，在 3 个任务上甚至超过 LLaMA2-70B。Qwen-7B 超越 LLaMA2-13B，与 Baichuan2-13B 持平。

### Table 3: Long-Context Inference Results (PPL)

| Model | 1024 | 2048 | 4096 | 8192 | 16384 |
|-------|------|------|------|------|-------|
| Qwen-7B (no ext.) | 4.23 | 3.78 | 39.35 | 469.81 | 2645.09 |
| + dynamic ntk | 4.23 | 3.78 | 3.59 | 3.66 | 5.71 |
| + dynamic ntk + logn | 4.23 | 3.78 | 3.58 | 3.56 | 4.62 |
| + dynamic ntk + logn + window attn | 4.23 | 3.78 | 3.58 | 3.49 | 4.32 |

**说明**: 组合三种技术后，模型在远超训练长度（2048）的上下文中仍保持低 PPL，无需额外微调。

### Table 4: PMP and Reward Model Accuracy

| Dataset | Helpful-base | Helpful-online | Anthropic Helpful-base | Anthropic Helpful-online | OpenAI Summ. | Stanford SHP | PRM800K |
|---------|-------------|----------------|----------------------|------------------------|-------------|-------------|---------|
| PMP | 62.68 | 61.62 | 76.52 | 65.43 | 69.60 | 60.05 | 70.59 |
| RM | 74.78 | 69.71 | 73.98 | 64.57 | 69.99 | 60.10 | 70.52 |

**说明**: PMP 在分布外数据上展示出高泛化能力，RM 在 Qwen 自有的偏好数据集上有显著提升。

### Table 5: Performance of Aligned Models on Benchmarks

| Model | Params | MMLU 0/5-shot | C-Eval 0/5-shot | GSM8K 0/8-shot | HumanEval 0-shot | BBH 0/3-shot |
|-------|--------|---------------|-----------------|----------------|------------------|--------------|
| GPT-3.5 | - | -/69.1 | -/52.5 | -/78.2 | 73.2 | -/70.1 |
| GPT-4 | - | -/83.0 | -/69.9 | -/91.4 | 86.6 | -/86.7 |
| Qwen-Chat | 14B | 64.6/66.5 | 69.8/71.7 | 60.1/59.3 | 43.9 | 46.9/58.7 |

**说明**: Qwen-14B-Chat 在所有 benchmark 上超越同规模开源模型（LLaMA2-Chat、Baichuan2-Chat、ChatGLM2、InternLM-Chat），HumanEval 上尤为突出。

### Table 6: Tool Use via ReAct Prompting (Chinese Benchmark)

| Model | Params | Tool Selection (Acc.) | Tool Input (Rouge-L) | False Positive Error (%) |
|-------|--------|----------------------|---------------------|-------------------------|
| GPT-4 | - | 95 | 90 | 15.0 |
| GPT-3.5 | - | 85 | 88 | 75.0 |
| Qwen-Chat | 14B | 98 | 93 | 2.4 |

**说明**: Qwen 在中文工具调用 benchmark 上表现优异，误触发率（2.4%）远低于 GPT-3.5（75.0%）。

### Table 7: Code Interpreter - Executability

| Model | Params | Math (%) | Visualization (%) | General (%) | All (%) |
|-------|--------|---------|-------------------|-------------|---------|
| GPT-4 | - | 91.9 | 85.9 | 82.8 | 86.8 |
| GPT-3.5 | - | 89.2 | 65.0 | 74.1 | 72.9 |
| Qwen-Chat | 14B | 89.2 | 84.1 | 65.5 | 81.7 |

**说明**: Qwen-14B-Chat 在代码可执行性上显著超越 CodeLlama-Instruct（68.8%），接近 GPT-4。

### Table 8: Code Interpreter - Correctness

| Model | Params | Math (%) | Vis.-Hard (%) | Vis.-Easy (%) | Vis.-All (%) |
|-------|--------|---------|--------------|--------------|-------------|
| GPT-4 | - | 82.8 | 66.7 | 60.8 | 63.8 |
| GPT-3.5 | - | 47.3 | 33.3 | 55.7 | 44.2 |
| Qwen-Chat | 14B | 58.4 | 53.6 | 59.5 | 56.4 |

**说明**: Qwen-14B-Chat 在最终答案正确性上远超开源替代方案，在可视化难题上（需多步规划）表现突出。

### Table 9: Hugging Face Agent Benchmark

| Task | Model | Params | Tool Selection | Tool Used | Code Correctness |
|------|-------|--------|---------------|-----------|-----------------|
| Run Mode | GPT-4 | - | 100 | 100 | 97.4 |
| Run Mode | Qwen-Chat | 14B | 93.5 | 94.4 | 87.0 |
| Chat Mode | GPT-4 | - | 97.9 | 97.9 | 98.5 |
| Chat Mode | Qwen-Chat | 14B | 97.9 | 97.9 | 95.5 |

**说明**: Qwen-14B-Chat 在 Hugging Face Agent 评测中与 GPT-4 差距很小，Chat Mode 下代码正确率达 95.5%。

### Table 10: Code-Qwen - HumanEval & MBPP Pass@1

| Model | Params | HumanEval | MBPP |
|-------|--------|-----------|------|
| GPT-4 | - | 86.6 | - |
| GPT-3.5 | - | 73.2 | - |
| WizardCoder-Python | 34B | 73.2 | 61.2 |
| Code-Qwen-Chat | 14B | 66.4 | 52.4 |
| Code-Qwen | 14B | 45.1 | 51.4 |

**说明**: Code-Qwen-Chat-14B（66.4%）在 HumanEval 上大幅超越同规模模型，逼近 WizardCoder-34B。

### Table 11: Code-Qwen - HumanEvalPack Zero-shot Pass@1

| Model | Params | Python | JS | Java | Go | C++ | Rust | Avg |
|-------|--------|--------|-----|------|-----|-----|------|-----|
| GPT-4 | - | 86.6 | 82.9 | 81.7 | 72.6 | 78.7 | 67.1 | 78.3 |
| WizardCoder | 15B | 59.8 | 49.5 | 36.1 | 36.4 | 40.9 | 20.2 | 40.5 |
| Code-Qwen-Chat | 14B | 66.4 | 58.5 | 56.1 | 47.6 | 54.2 | 28.7 | 51.9 |

**说明**: Code-Qwen-Chat-14B 在 6 种编程语言的多语言代码生成上平均 pass@1 达 51.9%，远超 WizardCoder-15B（40.5%）。

### Table 12: Math-Qwen-Chat - Mathematical Reasoning

| Model | Params | GSM8K | MATH | Math401 | Math23K |
|-------|--------|-------|------|---------|---------|
| GPT-4 | - | 92.0 | 42.5 | 83.5 | 74.0 |
| GPT-3.5 | - | 80.8 | 34.1 | 75.1 | 60.0 |
| Minerva | 62B | 52.4 | 27.6 | - | - |
| WizardMath | 70B | 81.6 | 22.7 | - | - |
| Math-Qwen-Chat | 7B | 62.5 | 17.2 | 80.8 | 75.4 |
| Math-Qwen-Chat | 14B | 69.8 | 24.2 | 85.0 | 78.4 |

**说明**: Math-Qwen-7B-Chat 在 MATH 上超越 Minerva-8B；14B 版本在 GSM8K 和 MATH 上追赶 Minerva-62B 和 GPT-3.5，在算术能力和中文数学题上表现更优。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[MMLU]] | 57 学科 | 多任务语言理解 | 知识/理解评测 |
| [[C-Eval]] | 52 学科 | 中文多层级多学科评测 | 中文能力评测 |
| [[GSM8K]] | 8.5K 题 | 小学数学应用题 | 数学推理评测 |
| [[MATH]] | 12.5K 题 | 竞赛级数学题 | 数学推理评测 |
| [[HumanEval]] | 164 题 | Python 函数补全 | 代码生成评测 |
| [[MBPP]] | ~1K 题 | Python 编程题 | 代码生成评测 |
| [[BBH]] | 23 任务 | BIG-Bench 难题子集 | 推理能力评测 |
| HumanEvalPack | 多语言扩展 | 6 种语言代码生成 | 多语言代码评测 |

### 实现细节

- **Backbone**: Decoder-only Transformer（基于 LLaMA 架构改进）
- **优化器**: [[AdamW]] ($\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$)
- **Batch Size**: 4M tokens
- **训练 Token**: 1.8B: 2.2T / 7B: 2.4T / 14B: 3.0T
- **精度**: BFloat16 混合精度
- **学习率调度**: Cosine schedule，decay 至峰值的 10%
- **序列长度**: 预训练 2048（代码继续预训练 8192）

### 人工评估细节

- 300 条中文指令，覆盖知识、语言理解、创意写作、编码、数学
- 每条指令 3 位标注者基于 helpfulness、informativeness、validity 排名
- RLHF 版本在创意写作上提升最显著，SFT 版本在知识类问答上也不差
- Qwen-14B-Chat-RLHF 在所有类别上均优于 SFT 版本，整体仍落后 GPT-4

---

## 批判性思考

### 优点
1. **覆盖面广**: 同时覆盖基座模型、对齐模型、代码专用、数学专用、Agent 能力，一套完整的技术体系
2. **中文优化出色**: C-Eval 等多中文 benchmark 上超越 LLaMA2 等模型，tokenizer 对中文压缩效率高
3. **长上下文扩展巧妙**: 三种训练无关技术的组合使用，无需额外训练即可扩展 4-8 倍上下文
4. **Agent 能力突出**: 工具调用和代码解释器的表现远超同类开源模型，代码可执行性甚至接近 GPT-4
5. **训练数据规模大**: 3T tokens 预训练在当时开源模型中处于第一梯队

### 局限性
1. **最大规模仅 14B**: 相比 GPT-3.5/GPT-4 差距仍大，未探索更大规模（如 70B+）模型的 Scaling Law
2. **人工评估规模较小**: 仅 300 条中文指令，覆盖面有限，难以全面衡量与 GPT-4 的差距
3. **RLHF 效果增量有限**: 从人工评估看 RLHF 提升并非碾压级，部分指标上 SFT 已足够好
4. **缺乏安全性系统评估**: 虽提到安全数据标注，但未报告安全性 benchmark 结果
5. **预训练细节有限**: 数据的领域分布、质量过滤的具体阈值、指令数据比例等未披露

### 潜在改进方向
1. 探索更大规模模型（如 72B、110B）的性能边界
2. 引入 [[MoE]] 架构提高参数效率
3. 评估更全面的对齐效果，添加安全性、真实性等维度
4. 扩展 [[RLHF]] 数据多样性和人类评估规模
5. 探讨直接偏好优化（[[DPO]]）等替代 RLHF 的更简洁对齐方法

### 可复现性评估
- [x] 代码开源 (GitHub: QwenLM/Qwen)
- [x] 预训练模型 (7B, 14B 权重开源)
- [x] 训练细节完整 (超参数、数据规模、优化器配置均披露)
- [ ] 数据集可获取 (预训练数据未开源，仅描述来源和处理流程)

---

## 关联笔记

### 基于
- [[LLaMA]]: Qwen 架构的基础，Decoder-only Transformer
- [[GPT-3]] / [[GPT-4]]: 对齐方法和整体范式的参照
- [[LLaMA 2]]: 对齐 baseline 对比

### 对比
- [[ChatGLM2]]: 中文 LLM 竞争对手
- [[Baichuan2]]: 同时期中文开源 LLM
- [[InternLM]]: 中文多语言 LLM
- [[Falcon]] / [[MPT]]: 开源 LLM baseline
- [[CodeLlama]]: 代码专用模型对比
- [[WizardMath]] / [[Minerva]]: 数学专用模型对比

### 方法相关
- [[RoPE]]: 位置编码方案
- [[RMSNorm]]: 归一化层
- [[SwiGLU]]: 激活函数
- [[Flash Attention]]: 高效注意力实现
- [[RLHF]]: 人类偏好对齐
- [[PPO]]: 策略优化算法
- [[Reward Model]]: 奖励建模
- [[Self-Instruct]]: 指令数据生成
- [[ReAct Prompting]]: 工具调用格式
- [[Code Interpreter]]: 代码执行增强
- [[NTK-aware Interpolation]]: 上下文扩展
- [[LogN-Scaling]]: 注意力缩放
- [[Window Attention]]: 窗口注意力

### 衍生模型
- [[Qwen-VL]]: 多模态视觉语言模型
- [[Qwen1.5]]: 后续改进版本
- [[Qwen2]]: 第二代
- [[Qwen2.5]]: 2.5 代
- [[Qwen3]]: 第三代

### 专用模型
- [[Code-Qwen]]: 代码专用
- [[Math-Qwen-Chat]]: 数学专用

### 硬件/数据相关
- [[MMLU]]: 多任务语言理解评测
- [[C-Eval]]: 中文能力评测
- [[GSM8K]] / [[MATH]]: 数学推理评测
- [[HumanEval]] / [[MBPP]]: 代码生成评测

---

## 速查卡片

> [!summary] Qwen Technical Report
> - **核心**: 阿里 Qwen 团队的首个 LLM 系列，含 1.8B/7B/14B 基座和 Chat 模型 + 代码/数学专用模型
> - **方法**: Decoder-only Transformer + RoPE/RMSNorm/SwiGLU + 3T tokens 预训练 + SFT/RLHF 对齐 + 专用继续训练
> - **结果**: 14B 模型在 7 个 benchmark 上全面超越此前同规模 SOTA，代码/数学专用模型领先开源方案
> - **代码**: https://github.com/QwenLM/Qwen

---

*笔记创建时间: 2026-05-18*
