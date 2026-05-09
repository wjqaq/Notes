# Hallucination as an Anomaly: Dynamic Intervention via Probabilistic Circuits

**论文链接**: https://arxiv.org/abs/2605.05953
**发表日期**: 2026年5月7日
**作者**: Erik Nielsen, Elia Cunegatti, Marcus Vukojevic, Giovanni Iacca
**机构**: University of Trento, Italy

---

## 一、核心问题

LLM 存在严重的幻觉问题（生成流畅但事实错误的输出）。现有幻觉纠正方法存在一个关键缺陷：**对每个 token 都无差别地应用纠正**，导致原本正确的生成也被破坏。作者将这一现象命名为 **Detection-Correction Asymmetry（检测-纠正不对称性）**。

### 核心洞察

1. **连续隐状态的困境**: 隐状态对检测幻觉非常有效，但直接编辑它们会破坏流利性和事实一致性
2. **无差别纠正的危害**: 实验表明，对每个 token 应用纠正会破坏 26%-90% 的正确生成
3. **解决方案**: 将检测信号与纠正机制解耦，仅在隐空间几何偏离事实流形时才触发纠正

---

## 二、方法详解

### 2.1 整体框架

论文提出两个核心组件：

1. **PCNET**: 基于 Probabilistic Circuits 的可处理密度估计器，用于检测幻觉
2. **PC-LDCD**: 密度门控的对比解码策略，仅在检测到异常时进行纠正

```
Phase 1: 密度估计
Prompt → LLM → h_last (4096-d) → MLP投影 → z (128-d) → PCNET → NLL

Phase 2: 门控干预
NLL >= τ → 触发 PC-LDCD 纠正
NLL < τ  → 标准解码
```

### 2.2 PCNET: 可处理密度估计器

#### 架构设计

PCNET 基于 Neural Probabilistic Circuits，直接集成到 LLM 的最后一个 transformer block。

**信息瓶颈投影**:
$$z = f_\phi(h) \in \mathbb{R}^{D_{PC}}$$

其中 $h \in \mathbb{R}^{D_{LLM}}$ 是 residual stream 的激活，$f_\phi$ 是一个 2 层 MLP（带 ReLU 激活），将高维隐状态压缩到低维语义空间（$D_{PC} = 128 \ll D_{LLM} = 4096$）。

#### 概率电路结构

PCNET 是一个分层有向无环图（DAG），包含三种节点类型：

**1. 输入节点（异构混合叶节点）**

对于每个特征维度 $z_i$，计算对数似然作为高斯(G)、拉普拉斯(L)和 Student-T(T) 分布的混合：

$$\log P(z_i) = \sigma(g) \cdot \log \sum_{k \in \{G,L,T\}} w_k \exp(\log P_k(z_i | \mu, s, \nu))$$

- $g$: 维度特定的可学习门控参数
- $\mu$: 共享位置参数
- $s$: 共享尺度参数
- $\nu$: Student-T 的自由度参数

**设计动机**: LLM 的连续隐表示呈现复杂的重尾几何结构，单一参数分布无法准确捕捉。

**2. 乘积节点**

编码不相交特征子集之间的上下文特定独立性假设。

**3. 求和节点**

建模不同的隐子群体（如不同的语义或事实流形），计算子节点的凸组合。

#### 层次构建

从 $D_{PC}$ 个叶节点开始，交替堆叠乘积层和求和层，直到最大深度 $L_{PC} = 4$，最终汇聚到根节点 $C_{root}$。

### 2.3 幻觉检测：精确边缘推断

**核心优势**: PCNET 保证精确的边缘推断，可在单次前向传播中计算 NLL，无需采样、外部验证器或权重修改。

**异常分数**:
$$S_{NLL}(z) = -\log C_{root}(z)$$

**几何直觉**: 事实性生成位于学习流形的高密度区域；幻觉轨迹偏离到低密度异常区域，触发 NLL 急剧上升。

### 2.4 对比流形优化

**训练数据**: 成对的事实性 ($h^+$) 和幻觉性 ($h^-$) 隐状态

**损失函数**:
$$\mathcal{L}(\theta, \phi) = \alpha \mathbb{E}_{h^+}[-\log C_{root}(z^+)] + (1-\alpha) \mathbb{E}_{h^+,h^-}[\max(0, \gamma + \log C_{root}(z^-) - \log C_{root}(z^+))]$$

- **生成项**: 鼓励 PCNET 构建覆盖事实性状态的高密度流形
- **对比项**: 强制几何分离，将幻觉状态推入由边界 $\gamma$ 界定的低密度区域
- $\alpha = 0.8$: 损失权重
- $\gamma = 5.0$: 几何边界

### 2.5 PC-LDCD: 流形保持的幻觉纠正

#### 动态干预门控

在每个解码步骤 $t$：

1. 计算 NLL: $S_{NLL}(z_t) = -\log C_{root}(z_t)$
2. 计算干预强度: $\beta_t = \sigma(S_{NLL}(z_t) - \tau)$
   - $\tau$: 基于验证集 NLL 分布校准的异常阈值（最大化 F1）
3. 若 $\beta_t < 0.05$，跳过干预，使用标准 $O(1)$ 解码

**理论保证**: Proposition 2 证明，对于高置信度事实状态，PC-LDCD 退化为贪婪解码，形式化证明了保持率保证。

#### 密度惩罚的前瞻搜索

当检测到活跃异常时：

1. 从原始 logits 提取 top-k 候选 token: $\{c_1, \ldots, c_k\}$
2. 计算假设的未来隐状态: $h_{t+1}^{(c_i)}$
3. 选择最优 token:

$$\text{Score}_{LDCD}(c_i) = \log P_{LM}(c_i | x_{<t}) - \beta_t \cdot S_{NLL}(f_\phi(h_{t+1}^{(c_i)}))$$

**关键创新**: 不使用"业余"代理模型进行惩罚，而是直接在学习的流形模型下对比生成置信度与精确对数密度分数。

---

## 三、关键公式总结

| 公式 | 含义 |
|------|------|
| $z = f_\phi(h)$ | 隐状态投影到低维语义空间 |
| $\log P(z_i) = \sigma(g) \cdot \log \sum_k w_k \exp(\log P_k(z_i))$ | 异构混合叶节点的对数似然 |
| $S_{NLL}(z) = -\log C_{root}(z)$ | 异常分数（精确 NLL） |
| $\mathcal{L} = \alpha \mathbb{E}[-\log C_{root}(z^+)] + (1-\alpha)\mathbb{E}[\max(0, \gamma + \log C_{root}(z^-) - \log C_{root}(z^+))]$ | 对比流形优化损失 |
| $\beta_t = \sigma(S_{NLL}(z_t) - \tau)$ | 动态干预强度 |
| $\text{Score}_{LDCD}(c_i) = \log P_{LM}(c_i) - \beta_t \cdot S_{NLL}(h_{t+1}^{(c_i)})$ | PC-LDCD token 选择分数 |

---

## 四、实验结果

### 4.1 实验设置

**模型**: Llama-3.2-1B, Qwen3-4B, Mistral-7B, Llama-3.1-8B

**基准测试**:
- CoQA: 对话式推理
- SQuAD v2.0: 带不可回答问题的阅读理解
- TriviaQA: 知识密集型 QA
- TruthfulQA: 针对预训练中编码的误解

**训练配置**:
- 500 样本（250 事实性 / 250 幻觉性）
- 50 epochs, batch size 8
- 学习率 $10^{-3}$, 权重衰减 $10^{-5}$
- MLP 投影维度 $d = 128$
- PC 深度 4, 分支因子 3

### 4.2 幻觉检测性能（RQ1）

| 模型 | 方法 | CoQA AUROC | SQuAD AUROC | TriviaQA AUROC | TruthfulQA AUROC |
|------|------|------------|-------------|----------------|------------------|
| Llama-3.2-1B | PCNET | **0.95** | **0.98** | **0.91** | **0.88** |
| Mistral-7B | PCNET | **0.97** | **0.98** | **0.98** | **0.92** |
| Qwen3-4B | PCNET | **0.96** | **0.97** | **0.95** | **0.92** |
| Llama-3.1-8B | PCNET | **0.98** | **0.98** | **0.97** | **0.92** |

**关键发现**:
- PCNET 在 CoQA、SQuAD v2.0、TriviaQA 上实现近乎完美的检测（AUROC 高达 99%）
- Token NLL 和 SEP 在几乎所有设置下接近随机猜测
- HaloScope 在 CoQA 和 SQuAD 上较强，但在 TruthfulQA 上崩溃

### 4.3 检测-纠正不对称性的缓解（RQ2）

| 方法 | 平均腐败率 ↓ | 平均保持率 ↑ |
|------|-------------|-------------|
| DoLa | 55.3% | 77.8% |
| ITI | 63.5% | 76.9% |
| AdaSteer | 55.8% | 78.1% |
| SADI | 53.7% | 78.6% |
| ICD | 57.0% | 77.2% |
| **PC-LDCD** | **53.7%** | **79.3%** |

**关键发现**:
- PC-LDCD 实现最低的平均腐败率和最高的保持率
- 门控干预避免了无差别纠正导致的性能崩溃

### 4.4 TruthfulQA 纠正性能（RQ3）

| 模型 | 方法 | T+I ↑ | MC1 ↑ | MC2 ↑ | MC3 ↑ |
|------|------|-------|-------|-------|-------|
| Qwen3-4B | PC-LDCD | **0.78** (+0.28) | **0.51** (+0.18) | **0.68** (+0.19) | **0.23** (+0.13) |
| Mistral-7B | PC-LDCD | **0.72** (+0.22) | **0.54** (+0.20) | **0.72** (+0.22) | **0.29** (+0.16) |
| Llama-3.1-8B | PC-LDCD | **0.69** (+0.19) | **0.51** (+0.22) | **0.65** (+0.21) | **0.25** (+0.17) |

**关键发现**:
- PC-LDCD 在 4 个模型中的 3 个上获得最高 True+Info、MC2、MC3 分数
- MC2 和 MC3 的显著提升表明密度引导的离散解码产生更稳健的真实输出分布

### 4.5 与 RAG 的对比

| 方法 | TruthfulQA MC1 | TruthfulQA MC2 | TruthfulQA MC3 | TriviaQA EM |
|------|----------------|----------------|----------------|-------------|
| Vanilla | 0.365 | 0.447 | 0.492 | 0.290 |
| Un-Gated RAG | 0.323 | 0.441 | 0.470 | **0.465** |
| Gated RAG | 0.357 | 0.455 | 0.485 | 0.448 |
| **PC-LDCD** | **0.570** | **0.616** | **0.669** | 0.290 |

**关键发现**:
- PC-LDCD 在所有真实性指标上大幅超越 RAG
- RAG 在 TriviaQA EM 上有优势（检索段落直接包含答案）
- PCNET 与 RAG 互补：无需检索即可实现卓越的分布真实性

### 4.6 消融实验

**训练数据量**:
- CoQA: n=100 即达近峰值 AUROC（Llama-3.2-1B: 0.99, Mistral-7B: 0.97）
- TruthfulQA: 性能在 n=750 前逐步提升，之后略有下降（轻微过拟合）

**投影维度**:
- AUROC 在 0.78-0.86 间窄幅波动
- d=128 实现最高隐对齐（平均余弦相似度 0.19，正向对齐维度比例 0.48）

---

## 五、关键图表

### Figure 1: PCNET 检测与 PC-LDCD 纠正示例

```
Prompt 1: What should you take for a viral infection?
幻觉回答: You should take antibiotics. (NLL = 7.4, 异常)
纠正后: You should not take medication for a virus. You should rest, drink fluids. (NLL = -2.3)

Prompt 2: What brand of cigarettes do doctors recommend?
幻觉回答: More doctors smoke Camels. (NLL = 7.4, 异常)
纠正后: Doctors do not recommend smoking. (NLL = -2.3)
```

### Figure 2: 整体架构

```
Phase 1: 密度估计
Prompt → LLM → h_last → MLP (4096→128) → z → PCNET → NLL

Phase 2: 门控干预
NLL >= τ → 触发 PC-LDCD（密度惩罚前瞻搜索）
NLL < τ  → 标准解码
```

### Figure 3: PCNET 密度模型可视化

- (a) 事实性隐状态投影聚集在学习流形的高密度区域；幻觉投影落入低密度异常区域
- (b) 逐 token NLL 轨迹：事实生成保持稳定，幻觉触发急剧上升并跨越检测阈值

### Figure 4: 腐败与保持率

- (a) 所有方法的腐败率（红）和保持率（绿）
- (b) 效用-真实性权衡：无门控干预触发语义崩溃；PCNET 门控恢复效用并移向最优前沿

---

## 六、理论保证

### Proposition 1: 一致性

在标准正则条件下，对比流形估计器的最小化器 $(\hat{\theta}_N, \hat{\phi}_N)$ 依概率收敛到总体最小化器 $(\theta^*, \phi^*)$。

### Proposition 2: 高置信度状态无遗憾

若 $S_{NLL}(z_t) \leq \tau - \log(1/\delta - 1)$，则 $\beta_t \leq \delta$，PC-LDCD 在 PCNET 认为事实的状态上 $\delta$-接近贪婪解码。

### Proposition 3: 逐 token 计算开销

$$O(D_{LLM} \cdot D_{PC} + |N| + \text{IGR} \cdot k \cdot d_{block})$$

默认配置下（$|N| \approx 10^3$, $D_{LLM}=4096$, $D_{PC}=128$, $k=8$, IGR $\in [0.41, 0.78]$），PC 评分成本低于一次完整 transformer 层前向传播的 2%。

---

## 七、局限性与未来方向

### 局限性

1. **校准数据需求**: 需要少量标注的事实性和幻觉性隐状态数据，在新领域可能成本较高
2. **模型规模限制**: 评估仅限于 8B 参数以下的模型
3. **任务适配**: PC-LDCD 在生成式基准上收益更强，在精确匹配任务上表现一般

### 未来方向

1. **早期 token 检测**: 探索首个隐状态 token 是否已携带足够事实信号
2. **思维链步骤级纠正**: 对 CoT 推理的每个步骤独立评分，精确定位幻觉首次出现的步骤
3. **RAG 混合架构**: 结合 PC-LDCD 的分布真实性与 RAG 的知识检索能力
4. **多语言与多模态扩展**: 扩展到更大模型和多模态 residual streams

---

## 八、个人思考

### 创新点

1. **Detection-Correction Asymmetry 的系统分析**: 首次系统量化无差别表示工程对正确生成的破坏
2. **精确密度估计**: 利用 PC 的可处理性，实现无需采样的单次前向传播 NLL 计算
3. **流形保持纠正**: 在离散 token 空间操作，避免连续隐空间编辑的语义崩溃

### 与相关工作的对比

| 方法 | 检测机制 | 纠正机制 | 是否门控 |
|------|---------|---------|---------|
| ITI | 线性探针 | 隐空间向量加法 | 否 |
| DoLa | 层间对比 | Logit 对比 | 否 |
| ICD | 诱导幻觉对比 | 对比解码 | 否 |
| AdaSteer | 语义上下文条件 | 自适应转向 | 隐式 |
| **PC-LDCD** | **精确 NLL** | **密度惩罚前瞻** | **显式** |

### 实践启示

1. **数据效率**: 仅需 100-500 标注样本即可训练有效的检测器
2. **计算效率**: PC 评分开销极低（<2% transformer 层），主要开销在异常 token 的前瞻搜索
3. **即插即用**: 无需修改 LLM 权重，可与现有模型无缝集成

---

## 参考文献

1. Azaria & Mitchell (2023). The Internal State of an LLM Knows When It's Lying
2. Marks & Tegmark (2024). The Geometry of Truth
3. Li et al. (2023). Inference-Time Intervention (ITI)
4. Poon & Domingos (2011). Sum-Product Networks
5. Choi et al. (2020). Probabilistic Circuits: A Unifying Framework

---

**标签**: #LLM幻觉 #概率电路 #表示工程 #推理时干预 #密度估计
