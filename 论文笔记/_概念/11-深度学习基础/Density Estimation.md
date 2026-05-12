---
type: concept
aliases: [密度估计, 概率密度估计, Density Estimator]
---

# Density Estimation

## 定义
从有限样本中估计随机变量概率密度函数 $P(x)$ 的统计学习方法，是生成模型和异常检测的基础。

## 数学形式

给定样本 $\{x_i\}_{i=1}^N$，学习参数化模型 $P_{\theta}(x)$，使：

$$
\theta^* = \arg\max_{\theta} \sum_{i=1}^N \log P_{\theta}(x_i)
$$

## 核心要点
1. 参数化方法（GMM, Normalizing Flow）vs 非参数化方法（KDE）
2. 可处理密度估计要求精确计算归一化常数
3. [[Probabilistic Circuit]] 提供可处理的精确密度估计
4. 在 LLM 隐空间中用于检测异常（低密度 = 幻觉）

## 代表工作
- [[PCNet]]: 用 PC 对 LLM 隐状态做密度估计检测幻觉
- Normalizing Flows (Rezende & Mohamed 2015)
- Autoregressive Models (PixelCNN, WaveNet)

## 相关概念
- [[Probabilistic Circuit]]
- [[Negative Log-Likelihood]]
- [[Anomaly Detection]]
- [[Kullback-Leibler Divergence]]
