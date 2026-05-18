---
type: concept
aliases: [监督微调, Instruction Tuning]
---

# Supervised Fine-Tuning

## 定义
在有标注的指令-响应对数据上对预训练模型进行微调，使其能够遵循指令并完成特定下游任务。是 LLM/VLM 后训练的标准第一阶段。

## 核心要点
1. 使用 ChatML 格式结构化指令数据，包括角色标注和视觉嵌入注入
2. Qwen2.5-VL 的 SFT 数据约 2M 条目（50% 纯文本 + 50% 多模态）
3. SFT 阶段通常冻结视觉编码器（ViT），仅训练 LLM 和 Merger
4. 数据覆盖通用 VQA、数学、代码、文档 OCR、定位、视频、Agent 等子集
5. 配合两阶段数据过滤流水线和 Rejection Sampling 提升数据质量

## 代表工作
- [[Qwen2.5-VL]]: SFT + DPO 两阶段后训练
- [[Qwen2-VL]]: 使用 ChatML 格式进行 SFT

## 相关概念
- [[Direct Preference Optimization]]
- [[Rejection Sampling]]
- [[Instruction Tuning]]
- [[Chain-of-Thought]]
