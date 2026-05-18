---
type: concept
aliases: [Instruction Tuning, 指令微调]
---

# Instruction Tuning

## 定义
在高质量指令-回答数据上微调预训练模型，使其能遵循人类指令并产生有用、忠实的输出。

## 核心要点
1. 用于幻觉缓解时通过构造高质量数据提高模型忠实度
2. 需大量高质量标注数据和计算资源
3. 与 RLHF 互补：SFT → RLHF 构成完整对齐 pipeline

## 代表工作
- [[LLaVA]]: 视觉指令微调的先驱工作
- Liu et al. (2024): 鲁棒指令微调用于幻觉缓解

## 相关概念
- [[RLHF]]
- [[LVLM]]
