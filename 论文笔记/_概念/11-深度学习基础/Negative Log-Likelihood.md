---
type: concept
aliases: [NLL, 负对数似然, Negative Log Likelihood]
---

# Negative Log-Likelihood

## 定义
概率模型对观测数据拟合程度的度量，定义为 $-\log P(x;\theta)$，值越小表示模型对数据的解释越好。在神经网络中作为标准损失函数和不确定性度量。

## 数学形式

$$
\text{NLL}(x) = -\log P_{\theta}(x)
$$

用于异常检测时，高 NLL 表示低密度异常点：

$$
\mathcal{S}_{\text{NLL}}(z) = -\log \mathcal{C}_{\text{root}}(z)
$$

## 核心要点
1. 等价于交叉熵在 one-hot 标签下的特例
2. 无需采样即可评估，适合推理时的实时异常检测
3. [[Probabilistic Circuit]] 保证精确 NLL 在单次前向传播中计算
4. 相比 token 概率更可靠（LLM 存在过度自信问题）

## 代表工作
- [[PCNet]]: 用 PC 根节点的 NLL 作为幻觉异常分数
- Normalizing Flows: 通过变量替换精确计算 NLL
- Maximum Likelihood Estimation: 以最小化 NLL 为训练目标

## 相关概念
- [[Density Estimation]]
- [[Probabilistic Circuit]]
- [[Anomaly Detection]]
- [[Cross-Entropy Loss]]
