---
type: concept
aliases: [ChatML, Chat Markup Language]
---

# ChatML

## 定义
OpenAI 提出的对话标记格式，用特殊 token 标记对话中每条消息的角色和边界，被多个 LVLM 和 LLM 采用。

## 核心要点
1. 使用 `<im_start>` 和 `<im_end>` 标记每条消息的起止
2. 格式：`<im_start>role\ncontent<im_end>`，role 为 user/assistant/system
3. Qwen-VL 的 SFT 阶段使用 ChatML 格式，训练时仅对 assistant 部分计算 loss
4. 支持多图对话：在 `Picture id:` 后嵌入图像 token

## 代表工作
- [[Qwen-VL]]: SFT 阶段使用的对话格式

## 相关概念
- [[Instruction Tuning]]
- [[Supervised Fine-Tuning]]
