---
type: concept
aliases: [异常检测, Outlier Detection, Novelty Detection]
---

# Anomaly Detection

## 定义
识别不符合预期模式（正常数据分布）的数据点或事件的技术。在 LLM 上下文中，将幻觉检测形式化为隐空间几何异常检测。

## 数学形式

PCNet 的异常分数：

$$
\mathcal{S}_{\text{NLL}}(z) = -\log \mathcal{C}_{\text{root}}(z), \quad \text{Anomaly} \iff \mathcal{S}_{\text{NLL}}(z) \geq \tau
$$

## 核心要点
1. 密度估计法：低密度区域 = 异常，高密度区域 = 正常
2. [[PCNet]] 利用 [[Probabilistic Circuit]] 的可处理性做单次前向异常检测
3. 与采样法（多次生成评估不确定性）相比，计算开销极低
4. 关键优势：无需外部验证器、无需权重修改、无需采样

## 代表工作
- [[PCNet]]: 将 LLM 幻觉检测形式化为隐空间异常检测
- Kossen et al. (2024): 语义熵探针（SEP）
- HaloScope (2024): 无标签生成的幻觉特征提取

## 相关概念
- [[Hallucination]]
- [[Density Estimation]]
- [[Negative Log-Likelihood]]
- [[Detection-Correction Asymmetry]]
