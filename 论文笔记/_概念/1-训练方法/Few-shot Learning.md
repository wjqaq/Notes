---
type: concept
aliases: [Few-shot Learning, In-context Learning, ICL, 少样本学习]
---

# Few-shot Learning

## 定义
模型仅通过少量示例（通常 2-8 个）即可适应新任务的能力，在大模型中体现为上下文学习（in-context learning）。

## 核心要点
1. 无需更新模型参数，仅通过在 prompt 中提供少量示例即可引导模型行为
2. Qwen-VL 使用随机采样的 few-shot exemplars（非 RICES 等精挑方法），在多个 VQA 和 captioning 基准上超越同参数量模型
3. LVLM 的 few-shot 能力取决于预训练阶段是否使用交错图文数据

## 代表工作
- [[Qwen-VL]]: 展现强 in-context learning 能力
- [[Flamingo]]: few-shot LVLM 的代表工作

## 相关概念
- [[In-context Learning]]
- [[LVLM]]
