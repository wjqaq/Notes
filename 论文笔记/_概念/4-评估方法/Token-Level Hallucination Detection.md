---
type: concept
aliases: [Token-Level Hallucination Detection, Token 级幻觉检测]
---

# Token-Level Hallucination Detection

## 定义
从 sentence-level 升级的幻觉检测方式，为每个生成的 token 单独提取跨模态注意力模式并判断是否含幻觉。

## 数学形式
$$\mathbf{A}_{m} = \mathbf{A}^{(l,h)}_{q_m \to n}$$

## 核心要点
1. Sentence-level DHCP 对生成序列所有 token 的注意力取平均，稀释了幻觉 token 信号
2. Token-level 为每步生成的 token 单独提取 $(L, H, N)$ 注意力张量
3. 训练数据获取：LVLM 生成 caption → 检测名词 → 与 CHAIR whitelist 比对 → 标注 token 级幻觉标签
4. 推理时在解码步骤中检测 → 若幻觉则修正注意力 → 重新计算 logits → 采样 token

## 代表工作
- [[MHSA]]: 首次将跨模态注意力幻觉检测/抑制扩展到 token 级别

## 相关概念
- [[Cross-Modal Attention]]
- [[DHCP]]
- [[CHAIR]]
