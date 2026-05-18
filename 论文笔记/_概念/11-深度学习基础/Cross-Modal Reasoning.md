---
type: concept
aliases: [跨模态推理, cross-modal reasoning, audiovisual reasoning]
---

# Cross-Modal Reasoning

## 定义
模型在涉及多个模态（如同时包含音频和视频）的输入上进行逻辑推理的能力。要求模型能整合和关联不同模态的信息以得出结论。

## 核心要点
1. 超越单模态感知，需要建立跨模态语义关联
2. 典型场景: 音视频问答（观看视频+听声音回答因果问题）、语音内容推理
3. Qwen3-Omni Thinking 变体支持全模态推理，包括音频-视频和纯音频推理场景
4. 评估基准: MMAU、MMSU、DailyOmni、VideoHolmes

## 代表工作
- [[Qwen3-Omni]]: Thinking 变体实现全模态推理
- Gemini-2.5-Pro: 闭源跨模态推理标杆

## 相关概念
- [[End-to-End Multimodal Training]]
