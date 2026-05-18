---
title: "UAM: A Dual-Stream Perspective on Forgetting in VLA Training"
method_name: "UAM"
authors: [Jianke Zhang, Yuanfei Luo, Yucheng Hu, Xiaoyu Chen, Yanjiang Guo, Ziyang Liu, Hongbin Xu, Tian Lan, Jianyu Chen]
year: 2026
venue: arXiv
tags: [vla, embodied-ai, catastrophic-forgetting, dorsal-expert, semantic-preservation, visual-dynamics, world-model, robot-manipulation]
zotero_collection: 2-具身智能
image_source: local
arxiv_html: https://arxiv.org/html/2605.15735
created: 2026-05-18
---

# 论文笔记：UAM: A Dual-Stream Perspective on Forgetting in VLA Training

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University, ByteDance Seed |
| 日期 | May 2026 |
| 项目主页 | https://cladernyjorn.github.io/Unified-Action-Model.github.io |
| 对比基线 | [[π0]], [[OpenVLA]], [[π0.5]], [[ChatVLA]], [[BagelVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.15735) |

---

## 一句话总结

> 受生物视觉双流（腹侧/背侧）启发，提出 UAM 架构：在 VLM 旁路增加生成式 Dorsal Expert 处理视觉控制特征，配合视觉动力学监督，实现端到端训练保留 >95% VLM 多模态能力，无需冻结参数或辅助数据。

---

## 核心贡献

1. **提出并量化 "Embodiment Tax"**: 系统性地度量了 VLA 微调过程中 VLM 多模态能力的流失，定义为 $\Delta(f_{VLA}) = 1 - S(f_{VLA})/S(f_{VLM})$
2. **诊断表征瓶颈**: 论证遗忘的根因在于单一编码器被强制同时承担语义理解（腹侧）和视觉运动控制（背侧）两个功能
3. **提出 UAM 架构 + Dorsal Expert**: 并行三专家 MoT 架构，Dorsal Expert 由生成式 UMM 初始化，辅以视觉动力学损失，实现语义保留的端到端动作学习
4. **设计空间实证研究**: 系统比较 6 种 Dorsal Expert 变体（随机/VLM/生成式初始化，有/无动力学监督），证明"生成先验 + 视觉动力学监督"是唯一同时满足动作性能和语义保留的配置

---

## 问题背景

### 要解决的问题
VLA 模型的标准范式是用 VLM 初始化后在动作数据上微调。但这种微调会**系统性地侵蚀 VLM 的多模态能力**（开放词汇识别、属性推理、指令跟随等），即 "Embodiment Tax"。这在长尾场景中尤为明显：模型仍能模仿熟悉的轨迹，但在未见物体、语言变异、利用先验知识消歧等方面能力退化。

### 现有方法的局限
1. **冻结 VLM**: 保护多模态能力，但控制依赖静态特征，不匹配操作任务的空间和动态需求
2. **辅助 VL 数据联合训练**（如 ChatVLA, $\pi_{0.5}$+KI）: 保留效果依赖辅助语料的规模、多样性和可用性，且需要额外的知识隔离机制防止目标干扰

两种方法都是**治标不治本**——它们回避了根本问题：为什么动作学习会侵蚀语义能力？

### 本文的动机
神经科学启示：灵长类视觉皮层采用**双流组织**——腹侧通路处理物体识别和语义，背侧通路处理空间布局和视觉运动控制，两个通路保持功能上独立的表征。当前 VLA 缺少这种分离：VLM 微调时，密集的动作损失信号会覆盖 VLM 原有的语义特征。**假设**：这一表征瓶颈是 VLA 遗忘的主因，架构层面的分离才能从根本上解决。

---

## 方法详解

### 模型架构

[[UAM]] 采用 **三专家并行 MoT** 架构，耦合语义 VLM、Dorsal Expert、动作专家：

- **输入**: 观测 $I_t$（三视角）+ 语言指令 $L$
- **Esem (Semantic Expert)**: 预训练 VLM 主干（Qwen2.5-7B/Bagel），编码语义 tokens $Z_{sem}$，类似腹侧通路
- **Edor (Dorsal Expert)**: 并行视觉通路，由生成式 [[Unified Multimodal Model|UMM]] 初始化，编码控制导向 tokens $Z_{dor}$，类似背侧通路
- **Eact (Action Expert)**: 同时 attend $Z_{sem}$ 和 $Z_{dor}$，预测低层动作 $a_t$（末端执行器位姿 + 夹爪状态）
- **视觉编码**: ViT tokens 路由到 Esem，VAE tokens 路由到 Edor
- **耦合方式**: 并行 [[Mixture-of-Transformers|MoT]] routing，三个专家通过不同的 [[Attention Mask]] 实现信息交换
- **总参数**: 7B (Esem) + 7B (Edor) + 2B (Eact) = ~16B MoT

### 核心模块

#### 模块1: Dorsal Expert（背侧专家）

**设计动机**: 利用[[双流假说|生物视觉双流组织]]的启发，为 VLA 提供第二条专用视觉通路，减轻 VLM 编码器同时处理语义和控制特征的表征瓶颈

**具体实现**:
- 由预训练**生成式 UMM**（[[Bagel]]）初始化，自带对视觉生成和场景变化的先验
- 接收视觉 tokens（VAE 编码），而非可学习查询 tokens
- 通过 [[Mixture-of-Transformers|MoT]] 并行路由与 Esem 和 Eact 耦合
- **关键**: 辅以视觉动力学辅助损失 $\mathcal{L}_{wm}$，迫使 Edor 进行中层级推理（预测视觉状态演变）

#### 模块2: 视觉动力学世界模型损失 ($\mathcal{L}_{wm}$)

**设计动机**: 仅有生成先验不足以让 Dorsal Expert 成为真正的背侧通路——需要匹配的监督信号驱动其进行独立的视觉推理

**具体实现**:
- 采用**单步去噪**机制（BagelVLA 风格），预测目标观测 $\hat{I}_{t+1}$
- 在 [[Bagel|Bagel]] 的双流 [[Flow Matching]] 框架内，简化为直接训练而非大规模预训练
- $\mathcal{L}_{wm}$ 不要求完整图像重建，仅利用中间去噪步骤的表征进行动作生成

#### 模块3: 三专家 MoT 路由

**设计动机**: 实现语义识别和控制特征在架构层面的自动分流

**具体实现**:
- Esem: 输入观测 $I_t$ + 语言 $L$，输出语义 tokens $Z_{sem}$
- Edor: 输入视觉 tokens $X_{dor}$（观测 $I_t$），输出控制 tokens $Z_{dor}$
- Eact: 同时 attend $Z_{sem}$ 和 $Z_{dor}$，通过并行 MoT 注意力掩码控制信息流
- 各专家的注意力掩码隔离不同类型的 token 交互（详见 Fig. 7）

---

## 关键公式

### 公式1: [[VLA|VLA 策略公式]]

$$
a_{i,t} = \pi_\theta(I_{i,t}, L_i)
$$

**含义**: VLA 策略将观测和语言指令映射为低层动作

**符号说明**:
- $I_{i,t}$: 轨迹 $i$ 的第 $t$ 步观测
- $L_i$: 自然语言指令
- $a_{i,t}$: 低层动作（末端执行器位姿 + 夹爪状态）
- $\pi_\theta$: 策略模型，由 VLM 初始化，在动作数据集上微调

### 公式2: [[Embodiment Tax|遗忘度量]]

$$
\Delta(f_{\text{VLA}}) = 1 - \frac{S(f_{\text{VLA}})}{S(f_{\text{VLM}})}
$$

**含义**: 量化 VLA 微调后 VLM 多模态能力的相对损失

**符号说明**:
- $f_{\text{VLM}}$: 原始（动作微调前）VLM 主干
- $f_{\text{VLA}}$: 动作微调后的 VLA 版本
- $S(\cdot)$: 多模态理解基准套件的平均得分（MMMU, MME, MMBench 等，越高越好）
- $\Delta = 0$: 无能力损失；$\Delta = 1$: 多模态得分归零

### 公式3: [[UAM|UAM 前向传播]]

$$
Z_{\text{sem}} = E_{\text{sem}}(I_t, L; \theta_{\text{sem}})
$$

$$
Z_{\text{dor}} = E_{\text{dor}}(X_{\text{dor}}; \theta_{\text{dor}})
$$

$$
a_t = E_{\text{act}}(Z_{\text{sem}}, Z_{\text{dor}}; \theta_{\text{act}})
$$

**含义**: UAM 三专家并行编码后联合预测动作

**符号说明**:
- $E_{\text{sem}}$: 语义专家（VLM 主干，"腹侧通路"类似物）
- $E_{\text{dor}}$: Dorsal Expert（"背侧通路"类似物）
- $E_{\text{act}}$: 动作专家，预测低层动作
- $X_{\text{dor}} \in \{I_t, q\}$: Dorsal Expert 输入——原始视觉 tokens 或可学习查询 tokens
- $Z_{\text{sem}}, Z_{\text{dor}}$: 动作专家联合 attend 的 token 集合

### 公式4: [[UAM|UAM 训练总损失]]

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{act}} + \lambda \mathcal{L}_{\text{wm}}(\hat{I}_{t+1}, I_{t+1})
$$

**含义**: 动作克隆损失 + 视觉动力学世界模型损失

**符号说明**:
- $\mathcal{L}_{\text{act}}$: 标准行为克隆/动作预测损失
- $\mathcal{L}_{\text{wm}}$: 视觉动力学（世界模型）损失——目标观测预测
- $\hat{I}_{t+1}$: 预测的目标观测
- $I_{t+1}$: 真实目标观测
- $\lambda$: 权重系数

---

## 关键图表

### Figure 1: 系统概览 — Reducing the Embodiment Tax with UAM

![[UAM_fig1_overview.png]]

**说明**: 三面板图：(左) 直接动作微调覆盖语义表征，导致泛化下降；(中) UAM 架构通过 MoT 路由和 Dorsal Expert 解耦语义与控制；(右) 生成式专家 + 视觉动力学最小化语义-动作鸿沟，达到高动作精度同时不牺牲 VLM 性能。

### Figure 2: VLM Score 与 Action Accuracy 的散点图

![[UAM_fig2_forgetting_scatter.png]]

**说明**: 在 Qwen2.5-7B 和 PaliGemma 两个主干上，比较 Freeze-VLM、+MoT、+MLP 三种耦合方式。解冻 VLM 显著提升 Action Accuracy，但 VLM Score 大幅下降；MoT 的 VLM Score 损失明显小于 MLP。同一模式跨主干成立，说明退化不是特定 VLM 的问题。

### Figure 3: UAM 架构与 Dorsal Expert 设计空间

![[UAM_fig3_architecture.png]]

**说明**: (底部) 三专家宏观架构：Esem、Edor、Eact 通过并行路由耦合；(顶部) Dorsal Expert 设计空间，变化初始化方式（随机/VLM/生成式）和输入模态（视觉 tokens/查询 tokens），探讨有无辅助视觉动力学目标。

### Figure 4: Dorsal Expert 设计评估

![[UAM_fig4_design_eval.png]]

**说明**: 在真实世界 OOD 任务和仿真任务上评估 6 种变体。UAM (3b)（生成式初始化 + 视觉动力学监督）在 Esem 冻结条件下仍保持接近未冻结基线的性能，证明 Dorsal 通路承载了大部分控制相关的视觉信息。

### Figure 5: OOD 操作任务成功率

![[UAM_fig5_ood_success.png]]

**说明**: UAM 在未见物体、新物体-目标组合、指令变异（拼音、中英混码、全中文）等 OOD 场景中取得最高平均成功率。VLM-init dorsal expert 在分布内任务上接近 2-expert 基线性能，但在 OOD 任务上反而恶化，说明仅用 VLM 初始化背侧通路不足以有效传递语义泛化能力。

### Figure 6: 动作生成过程中的注意力图可视化

![[UAM_fig6_attention_maps.png]]

**说明**: Esem tokens 的注意力高度集中在任务相关的语义实体（目标物体、目标区域），几乎完全忽略机器人本体；Edor tokens 的注意力持续聚焦于机器人末端执行器、交互边界和空间关系。验证了 **功能分叉自然涌现**：UAM 架构自动将视觉信号的语义需求和动态需求路由到不同通路。

### Figure 7: 不同 Dorsal Expert 设计的注意力掩码

![[UAM_fig7_attention_masks.png]]

**说明**: (a) 2-expert 基线（无 Dorsal）；(b) Random/VLM/Gen-init 三专家掩码；(c) UAM 掩码。ViT tokens 路由到 Esem，VAE tokens 路由到 Edor。

### Figure 8-9: UAM 语义 OOD 任务评估演示

![[UAM_fig8_eval_demos1.png]]

![[UAM_fig9_eval_demos2.png]]

**说明**: 真实世界双臂 ALOHA 平台上的评估任务截图。展示了未见物体、新组合、语言变异等场景。

### Table 1: Dorsal Expert 设计空间变体

| Variant | $E_{dor}$ 初始化 | $X_{dor}$ 输入 | 辅助损失 |
|---------|------------------|---------------|---------|
| No Dorsal ($\pi_0$-style) | — | — | 无 |
| Random-init | 随机 | 视觉 tokens | 无 |
| VLM-init vision (2a) | 预训练 VLM | 视觉 tokens | 无 |
| VLM-init query (2b) | 预训练 VLM | 可学习查询 | 无 |
| Gen-init only (3a) | 生成式 UMM | 视觉 tokens | 无 |
| **UAM (3b)** | **生成式 UMM** | **视觉 tokens** | **视觉动力学 $\mathcal{L}_{wm}$** |

**说明**: 6 种变体覆盖两条架构轴（初始化、输入模态），并仅在最终变体引入视觉动力学目标。

### Table 2: 多模态理解基准评估

| Method | #Params | MMMU | MME-P | MME-S | MMBench | MM-Vet | MathVista | MMStar | TextVQA |
|--------|---------|------|-------|-------|---------|--------|-----------|--------|---------|
| **VLMs (上界)** |
| Qwen2-VL | 7B | 54.1 | — | 2327 | 83.0 | 62.0 | 58.2 | 60.7 | 84.3 |
| Qwen2.5-VL | 7B | 58.6 | — | 2347 | 83.5 | 67.1 | 68.2 | 63.9 | 84.9 |
| BAGEL | 7B MoT | 55.3 | 1687 | 2388 | 85.0 | 67.2 | 73.1 | — | — |
| **VLAs (动作微调后)** |
| OpenVLA | 7B | 0 | 0 | — | 0 | 0 | 0 | 0 | 0 |
| ECOT* | 7B | 5.4 | 0 | — | — | — | — | 0 | 0 |
| DiVLA* | 2B | 17.2 | 187 | — | — | — | — | 21.1 | 7.5 |
| ChatVLA* | 2B | 37.4 | 1435 | — | 69.0 | — | — | 47.2 | 71.2 |
| $\pi_{0.5}$-base* | 2B MoT | 18.7 | 1032 | 1241 | 7.3 | — | — | — | — |
| **UAM (Ours)** | **7B MoT** | **53.7** | **1607** | **2289** | **83.7** | **63.4** | **68.2** | **61.3** | **84.2** |

**说明**: * 号表示使用 VL 联合训练（VQA co-training）的方法。UAM 仅用动作数据训练，无参数冻结，保留 >95% 的 VLM 多模态能力，在所有 VLA 方法中排名第一且接近 VLM 上界。OpenVLA 等标准端到端 VLA 训练导致 VLM 多模态理解能力**完全归零**。

### Table 3: 图 2 的详细数据（不同 VLA 架构的遗忘情况）

| VLA Arch (VLM+Action Head) | Action | VQA-AVG | MME | MMMU | MMBench |
|---------------------------|--------|---------|-----|------|---------|
| Qwen2.5-Freeze + MLP (7B+10M) | 30.12 | 74.32 | 2354 | 53.7 | 85.2 |
| Qwen2.5 + MoT (7B+1B) | 65.98 | 37.94 | 1675 | 30.46 | 23.54 |
| Qwen2.5 + MLP (7B+10M) | 71.14 | 0 | 0 | 0 | 0 |
| Paligemma-Freeze + MLP (2.3B+10M) | 22.18 | 53.18 | 1670 | 34.66 | 65.23 |
| Paligemma + MoT (2.3B+0.3B) | 70.18 | 10.83 | 400 | 6.2 | 12.00 |
| Paligemma + MLP (2.3B+10M) | 70.12 | 0 | 0 | 0 | 0 |

**说明**: (1) MoT 架构比 MLP 更好地保留语言能力（相同动作精度下）；(2) 大模型 (7B) 比小模型 (2B) 对遗忘更具韧性；(3) MLP 头在 Qwen 和 PaliGemma 上均导致 VQA 完全归零——灾难性遗忘。

### Table 4: 推理速度对比

| Model | Size | Inference Speed |
|-------|------|----------------|
| $\pi_{0.5}$ | 2.3B + 0.3B MoT | 250ms |
| Qwen7B + MLP Head | 7B | 1000ms |
| Qwen7B-$\pi_0$ | 7B + 2B MoT | 1300ms |
| **UAM** | **7B + 7B + 2B MoT** | **1500ms** |

**说明**: UAM 采用单步去噪机制，绕过世界模型重建整张图像的需求。仅通过中间去噪步骤的表征进行动作生成，3-expert UAM 的推理延迟与 2-expert Qwen-$\pi_0$ 相当。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[CALVIN]] (ABC-D) | 长时程任务 | 语言条件操作 | 仿真 in-domain 动作评估 |
| [[RoboTwin]] | 50 任务（选 16 难任务） | 双臂操作域随机化 | 仿真 in-domain 测试 |
| ALOHA 双臂数据 | 3,000 条演示轨迹 | 真实世界双臂操作 | 训练（无预训练） |
| OOD 真实世界任务 | 自定义（未见物体/目标组合/语言变异） | 泛化评估 | 语义泛化测试 |

### 实现细节

- **Esem Backbone**: Qwen2.5-7B (Bagel 权重)
- **Edor Backbone**: Bagel 生成专家 (7B)
- **视觉编码**: ViT (patch=14) for Esem, VAE (stride=16) for Edor
- **优化器**: Adam, lr=1e-5 (Qwen), lr=5e-5 (PaliGemma)
- **框架**: FSDP + packed datasets
- **训练步数**: 30,000 步
- **动作块大小**: 10 (CALVIN) / 16 (RoboTwin) / 24 (真实机器人)
- **硬件**: 8x A800 GPU
- **推理**: 单步去噪（无需完整图像重建）

### 关键发现

1. **Embodiment Tax 是真实且可量化的**: 即使 MoT 路由的 VLA 在动作微调后也会丢失大量 VLM 能力
2. **架构分离本身就能保留语义**: UAM 在零冻结、零辅助数据、零梯度停止条件下保留 >95% VLM 能力
3. **仅有并行通路不够**: 随机初始化的 Edor 表现差于 2-expert 基线；VLM 初始化虽有改善但 OOD 泛化反而退步
4. **生成先验 + 动力学监督是关键配方**: Variant 3b 同时满足 (i) 动作准确率不低于完全微调，(ii) 不侵蚀 VLM Score
5. **功能分叉自然涌现**: 无任何像素级或 grounding 监督，注意力自动分离——Esem 关注语义实体，Edor 关注机器人臂和交互区域
6. **语义保留 → 动作语义泛化**: UAM 在 OOD 任务（未见物体、语言变异）上取得最高平均成功率

---

## 批判性思考

### 优点
1. **问题定义清晰且可度量**: "Embodiment Tax" 通过 $\Delta$ 度量被形式化，提供了可复现的量化评估框架
2. **从原理到设计的闭环**: 测量-诊断-设计的路径完整——先量化遗忘，再提出瓶颈假说，最后通过设计空间研究验证解决方案
3. **实验设计严谨**: 6 变体 ablation 系统性地隔离了初始化、输入模态、辅助监督三个因素的贡献；"冻结探针"技巧巧妙区分了 Edor 的实际贡献
4. **零辅助数据的端到端训练**: 不依赖 VL co-training 或参数冻结，证明了语义保留可以是架构本身的属性
5. **可解释的注意力分析**: 注意力可视化验证了功能分叉的涌现，为理论假说提供了定性证据

### 局限性
1. **世界模型模块增加训练和推理复杂度**: 额外 7B 的 Edor + 动力学损失增加了训练开销（Table 4 中 UAM 1500ms vs $\pi_{0.5}$ 250ms）
2. **仅在 Bagel/Qwen 主干上验证**: 应用于 PaliGemma 系列需要额外对齐步骤，限制了方法的即插即用性
3. **动作数据集规模较小**: 仅用 3,000 条 ALOHA 轨迹训练，大规模多具身数据上的表现未知
4. **仅有行为克隆**: 未探索与 RL/RLHF 结合的潜力
5. **缺少 Sim-to-Real 评估**: 真实世界评估仅在 ALOHA 平台上，缺少在更多不同机器人平台上的验证

### 潜在改进方向
1. **轻量化 Dorsal Expert**: 探索更小的 Edor（如 1B 级别）以减少推理延迟，或采用蒸馏技术
2. **多具身探索**: 在大规模异构机器人数据集（Open X-Embodiment）上验证 UAM 的跨具身泛化能力
3. **结合 RL 微调**: 在行为克隆后使用 RL 进行策略优化，可能进一步提升 OOD 性能
4. **视觉动力学损失泛化**: 探索更丰富的动力学目标（如未来多步预测、潜在动作预测）替代简单目标观测预测
5. **与 VLM-freeze 方法结合**: 在 Esem 部分冻结的策略下测试 UAM，探索是否有组合收益

### 可复现性评估
- [ ] 代码开源（本文提交时尚未公开，但作者有相关开源工作如 UP-VLA）
- [ ] 预训练模型（使用公开的 Bagel 权重）
- [x] 训练细节完整（学习率、步数、batch size、硬件配置）
- [ ] 数据集可获取（ALOHA trajectories 需自行采集；CALVIN/RoboTwin 公开）

---

## 关联笔记

### 基于
- [[Bagel]]: 提供预训练 VLM + 生成专家权重（7B MoT），UAM 的 Esem 和 Edor 均基于此初始化
- [[BagelVLA]]: 单步去噪机制和双流 Flow Matching 框架的直接来源
- [[双流假说]]: 生物视觉腹侧/背侧通路的核心启发
- [[VLM4VLA]]: VLA 中表征瓶颈假说的前序工作
- [[π0]]: 2-expert MoT 架构的基线设计

### 对比
- [[OpenVLA]]: 标准 VLA 全参数微调导致 VLM 能力完全归零
- [[ChatVLA]]: VL 联合训练方法，但通用理解能力仍受损
- [[π0.5]]: KI (Knowledge Insulation) 机制，与 UAM 的架构分离互补
- [[DiVLA]]: 扩散+自回归 VLA，语义保留有限

### 方法相关
- [[Embodiment Tax]]: 核心概念——VLA 微调中 VLM 多模态能力的系统性侵蚀
- [[Dorsal Expert]]: 核心方法——并行视觉通路，生成式 UMM 初始化 + 视觉动力学监督
- [[Mixture-of-Transformers]]: 多专家并行路由机制
- [[Flow Matching]]: 动作生成框架
- [[World Model]]: 视觉动力学预测的底层框架

### 硬件/数据相关
- [[ALOHA]]: 双臂操作平台，用于数据采集和真实世界评估
- [[CALVIN]]: 仿真长时程操作基准
- [[RoboTwin]]: 仿真双臂操作域随机化基准

---

## 速查卡片

> [!summary] UAM: A Dual-Stream Perspective on Forgetting in VLA Training
> - **核心**: 生物视觉双流启发——腹侧 VLM（语义）+ 背侧 Dorsal Expert（控制）= 无需冻结/辅助数据，保留 >95% VLM 能力
> - **方法**: 三专家 MoT：Esem(ViT) + Edor(VAE/生成式UMM初始化+视觉动力学损失Lwm) + Eact → 端到端训练
> - **结果**: Embodiment tax <5%；在 OOD 操作任务中取得最高平均成功率；注意力可视化验证自动功能分叉
> - **代码**: 尚未公开，项目主页 https://cladernyjorn.github.io/Unified-Action-Model.github.io

---

*笔记创建时间: 2026-05-18*
