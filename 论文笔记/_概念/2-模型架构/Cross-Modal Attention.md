---
type: concept
aliases: [跨模态注意力, Cross-Modal Attention]
---

# Cross-Modal Attention

## 定义
LVLM 中文本 token（LLM 层）对视觉 token（视觉编码器输出）的注意力权重分布，反映模型在生成文本时对视觉信息的依赖程度。

## 数学形式
$$\mathbf{A}^{(l,h)}_{q \to n}$$

其中 $q$ 为输出位置，$n$ 为视觉 token 索引，$l$ 为 LLM 层，$h$ 为注意力头。

## 核心要点
1. 幻觉文本与非幻觉文本的跨模态注意力模式存在显著且可区分的差异 (DHCP 发现)
2. 跨模态注意力张量形状为 $(L, H, N)$（层数 × 头数 × 视觉token数）
3. 可通过统一的固定分辨率输入标准化注意力形状
4. 是 MHSA 方法的核心操作对象——通过修正注意力来抑制幻觉

## 代表工作
- [[DHCP]]: 利用跨模态注意力检测幻觉
- [[MHSA]]: 学习修正跨模态注意力以抑制幻觉

## 相关概念
- [[多模态幻觉]]
- [[Spatial Grounding]]
- [[LLM]]
- [[Vision Transformer]]
