---
type: concept
aliases: [Low-Rank Adaptation, 低秩适配]
---

# LoRA (Low-Rank Adaptation)

## 定义
一种参数高效微调方法，通过在预训练权重矩阵上添加低秩分解矩阵来适配下游任务，冻结原有权重仅训练新增的低秩参数。

## 数学形式
对原始权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 添加低秩更新：
$$h = W_0 x + \Delta W x = W_0 x + BA x$$

其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$。

## 核心要点
1. 冻结预训练权重，仅训练低秩矩阵 $A$ 和 $B$
2. 秩 $r$ 通常取 8-128，越大表达力越强但参数越多
3. 可与全参数微调达到相当性能，但训练参数减少 100-1000x
4. 在 [[Re-Align]] 中使用 r=128, alpha=256 对所有模块做 LoRA 微调

## 代表工作
- [[Re-Align]]: r=128 LoRA 对齐微调
- 多数 VLM/LLM 微调工作

## 相关概念
- [[Vision Language Model|VLM]]
- [[Direct Preference Optimization]]
