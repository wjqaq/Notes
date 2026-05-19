---
type: concept
aliases: [TPO Loss, Token Preference Optimization loss]
---

# Token Preference Optimization loss

## 定义
TPO 中提出的训练损失函数，将 token 级视觉锚定奖励融入 DPO 框架。等价于标准 DPO 损失加上 token 级视觉锚定修正项，实现对视觉相关 token 的精细调控。

## 数学形式

$$
\begin{aligned}
\mathcal{L}_{\text{TPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x,v,y_w,y_l)\sim\mathcal{D}} \log \sigma\Bigg( &\beta \sum_{y_{w_i} \in y_w} \Big[ \log p_\theta(y_{w_i}|...) - \log p_{\text{ref}}(y_{w_i}|...) + \log \frac{c_{y_{w_i}}^\theta}{c_{y_{w_i}}^{\text{ref}}} \Big] \\
- &\beta \sum_{y_{l_i} \in y_l} \Big[ \log p_\theta(y_{l_i}|...) - \log p_{\text{ref}}(y_{l_i}|...) + \log \frac{c_{y_{l_i}}^\theta}{c_{y_{l_i}}^{\text{ref}}} \Big] \Bigg)
\end{aligned}
$$

## 核心要点
1. 可分解为 $\mathcal{L}_{\text{DPO}} +$ token 级视觉锚定修正项
2. 修正项 $\log(c_{y_i}^\theta / c_{y_i}^{\text{ref}}) \in (-\log 3, \log 3)$ 有界，训练稳定
3. 正样本修正项鼓励增大，负样本鼓励减小
4. 每次训练步重新计算奖励 $c^\theta$（自校准）

## 代表工作
- [[TPO]]: 提出 TPO 损失

## 相关概念
- [[Direct Preference Optimization]]
- [[Visual-Anchored Rewards]]
- [[Self-Calibrated Rewards]]
- [[Bradley-Terry Model]]
