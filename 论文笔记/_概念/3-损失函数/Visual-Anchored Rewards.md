---
type: concept
aliases: [Visual-Anchored Rewards, 视觉锚定奖励]
---

# Visual-Anchored Rewards

## 定义
TPO 中提出的 token 级奖励机制，通过对比同一 token 在原图和加噪图条件下的 logits 分布差异来量化该 token 对视觉信息的依赖程度，差异越大奖励越高/越低（视正负样本而定）。

## 数学形式

视觉锚定分数:
$$
s_{y_i} = p_{\log}(y_i | x, v, y_{<i}) - p_{\log}(y_i | x, v_c, y_{<i})
$$

自校准奖励:
$$
c_{y_i} = \begin{cases} a + \sigma(s_{y_i}) & \text{if } y_i \in y_w \\ a + 1 - \sigma(s_{y_i}) & \text{if } y_i \in y_l \end{cases}
$$

## 核心要点
1. 无需额外模型或标注，完全基于 LVLM 自身的 logits 差异
2. 能有效区分视觉锚定 token（名词、形容词）和其他 token
3. 加噪图像通过扩散过程生成，噪声步数 500 最优
4. 奖励值范围 $c_{y_i} \in (0.5, 1.5)$，$a=0.5$ 保证无差异时 $c=1$

## 代表工作
- [[TPO]]: 提出自校准视觉锚定奖励

## 相关概念
- [[Self-Calibrated Rewards]]
- [[Visual-Anchored Token]]
- [[Token-Level Rewards]]
- [[TPO]]
