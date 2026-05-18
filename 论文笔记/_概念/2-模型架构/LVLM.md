---
type: concept
aliases: [Large Vision-Language Model, 大型视觉语言模型, VLM]
---

# LVLM (Large Vision-Language Model)

## 定义
集成视觉编码器与大语言模型的统一多模态模型，能够同时理解和推理视觉与语言信息。

## 核心要点
1. 典型架构：视觉编码器（如 ViT/CLIP） + 投影器 + LLM
2. 支持 VQA、图像描述、视觉推理等多模态任务
3. 核心挑战：幻觉（生成与视觉输入不一致的内容）
4. 代表模型：Qwen2.5-VL、InternVL2、LLaVA-v1.5、GPT-4V 等

## 代表工作
- [[CLIP]]: 视觉语言预训练基础
- [[LLaVA]]: 视觉指令微调
- [[MHSA]]: LVLM 幻觉抑制

## 相关概念
- [[多模态幻觉]]
- [[Cross-Modal Attention]]
- [[LLM]]
- [[Vision Transformer]]
