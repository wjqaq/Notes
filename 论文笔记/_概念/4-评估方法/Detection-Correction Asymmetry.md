---
type: concept
aliases: [检测-纠正不对称性, DCA]
---

# Detection-Correction Asymmetry

## 定义
LLM 幻觉缓解中的一种现象：隐空间连续状态对于检测幻觉非常有效，但直接编辑它们会破坏流利性和事实一致性，导致原本正确的生成也被腐化。

## 核心要点
1. 实验证实无差别纠正破坏 26%-90% 的正确生成
2. 根源：连续隐空间编辑将激活推离 LLM 预训练流形
3. 解决方案：解耦检测与纠正——隐空间仅用于诊断（检测），纠正路由到离散 token 空间
4. [[PCNet]] 的 PC-LDCD 通过动态门控实现：$\beta_t = \sigma(\mathcal{S}_{\text{NLL}}(z_t) - \tau)$

## 代表工作
- [[PCNet]]: 系统分析 DCA 并通过密度门控解决
- [[Representation Engineering]]: 无差别编辑的典型代表
- [[ITI]]: 腐化率高达 63.5% 的案例

## 相关概念
- [[Hallucination]]
- [[Representation Engineering]]
- [[Contrastive Decoding]]
- [[Anomaly Detection]]
