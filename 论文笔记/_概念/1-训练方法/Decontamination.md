---
type: concept
aliases: [去污染, data decontamination, train-test overlap removal]
---

# Decontamination

## 定义
在训练数据中检测并移除与测试集重叠的样本，防止评测结果因数据泄露而虚高。

## 核心要点
1. Qwen2.5 使用 n-gram 匹配检测：若训练序列和测试序列的最长公共子序列 > 13 tokens 且 > 60% 较短序列长度，则移除
2. 在预训练和后训练阶段均执行
3. 确保基准评测的可靠性

## 代表工作
- [[Qwen2.5]]: 在 18T 预训练数据和 1M+ SFT 数据上执行 LCS 去污染
- [[Qwen2]]: 同样使用 LCS 方法

## 相关概念
- [[Scaling Law|缩放定律]]
- [[Pre-training]]
