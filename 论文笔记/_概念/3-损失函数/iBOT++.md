---
type: concept
aliases: [iBOT plus plus, enhanced masked image modeling]
---

# iBOT++

## 定义
对 iBOT 的升级版本，移除遮蔽指示器 $m_i$，使损失作用于所有 patch token（遮挡和未遮挡），由 TIPSv2（CVPR 2026）提出。

## 数学形式
$$\mathcal{L}_{iBOT++} = -\sum_{i=1}^{N} h_t(f_t(I)_i)^T \log h_s(f_s(I_{mask})_i)$$

与 iBOT 的唯一区别：去掉了 $m_i$ 乘子。

## 核心要点
1. 对可见（未遮挡）token 也施加监督，使其"锚定"到教师表示
2. 可见 token 的 patch 级损失在 iBOT++ 下持续下降（原始 iBOT 不会）
3. 带来 zero-shot ADE150 分割 +14.1 mIoU 的巨大提升
4. 最优 mask ratio 仍为 75%；蒸馏时可将 mask ratio 降至 0
5. 可独立应用于 CLIP 训练（附录验证），具有普适性

## 代表工作
- [[TIPSv2]]: 首次提出 iBOT++，作为核心贡献之一

## 相关概念
- [[iBOT]]
- [[Masked Image Modeling]]
- [[Patch-Text Alignment]]
- [[Knowledge Distillation]]
