---
type: concept
aliases: [内部对比解码, internal contrastive decoding, SIRA decoding]
---

# Internal Contrastive Decoding

## 定义
SIRA 提出的模型内对比解码机制：在单一 LVLM 内部构造两个分支（全分支和反事实分支），对比其 logits 以抑制[[语言先验]]驱动的 token，增强[[视觉接地]]强的 token。

## 数学形式
$$
z_t^{cd}(v) = (1 + \alpha)z_t^{full}(v) - \alpha z_t^{cf}(v)
$$

## 核心要点
1. 是一种近似基线减法：$z_t^{cf}$ 作为语言先验参考，相减得到每个 token 的增量视觉贡献
2. $\alpha=0.5$ 为最优对比强度
3. 对比在 float32 中计算以避免 bfloat16 半精度 rank reversal
4. 无自适应合理性约束或词汇截断

## 代表工作
- [[SIRA]]: 提出并验证该方法
