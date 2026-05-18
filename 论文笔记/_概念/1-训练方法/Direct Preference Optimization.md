---
type: concept
aliases: [直接偏好优化, 直接偏好对齐]
---

# Direct Preference Optimization

## 定义
一种无需显式训练奖励模型的偏好对齐方法，通过直接优化策略模型在偏好对上的对数概率比来对齐人类偏好。

## 数学形式
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma \left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

## 核心要点
1. 直接使用偏好对数据 $(y_w, y_l)$，无需训练独立的奖励模型
2. $\beta$ 控制偏离参考策略 $\pi_{\text{ref}}$ 的程度
3. Qwen2.5-VL 在后训练第二阶段使用 DPO，仅处理图文+纯文本偏好数据
4. 每个训练样本仅使用一次（单轮 DPO）

## 代表工作
- [[Qwen2.5-VL]]: SFT + DPO 两阶段后训练
- [[Qwen3-VL]]: 继承 DPO 对齐策略

## 相关概念
- [[Supervised Fine-Tuning]]
- [[RLHF]]
- [[Rejection Sampling]]
- [[Bradley-Terry Model]]
