---
type: concept
aliases: [InternVL2-8B, InternVL 2]
---

# InternVL2

## 定义
InternVL 2 系列多模态大模型，由上海 AI 实验室提出，支持动态分辨率视觉编码和多模态理解，在多个视觉-语言基准上表现优异。

## 核心要点
1. 动态分辨率：支持原生可变分辨率图像输入（无需固定 resize）
2. InternVL2-8B 是 MHSA 评估的三个 LVLM backbone 之一（另有 Qwen2.5-VL 和 LLaVA-v1.5）
3. 8B 版本：32 层 LLM + 32 注意力头 + 256 视觉 token，跨模态注意力维度 d=262,144

## 代表工作
- [[MHSA]]: 在 InternVL2-8B 上验证注意力修正幻觉抑制，POPE-COCO F1=94.16

## 相关概念
- [[LVLM]]
- [[Cross-Modal Attention]]
- [[Qwen2.5-VL]]
- [[LLaVA]]
