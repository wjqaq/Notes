---
type: concept
aliases: [Thinker-Talker 架构, Thinker-Talker]
---

# Thinker-Talker Architecture

## 定义
一种端到端多模态架构，将文本理解和生成（Thinker）与流式语音合成（Talker）分离为两个协同模块，共享高层多模态表征。

## 核心要点
1. Thinker 负责跨模态理解与文本生成（自回归 LLM），Talker 从 Thinker 接收多模态特征进行流式语音合成
2. Thinker 和 Talker 共享完整对话历史，可分别使用不同的系统提示独立控制文本风格和语音风格
3. 支持分块异步 prefill：Thinker 完成当前块后立即 prefill Talker，同时处理下一块
4. Talker 仅依赖多模态特征（非文本 token），使外部模块（RAG、安全过滤）可介入 Thinker 输出
5. 首次提出于 Qwen2.5-Omni，Qwen3-Omni 升级为 MoE 设计

## 代表工作
- [[Qwen2.5-Omni]]: Thinker-Talker 架构首次提出
- [[Qwen3-Omni]]: MoE 升级 + 多码本流式生成 + AuT 编码器

## 相关概念
- [[Mixture-of-Experts]]
- [[Chunked Prefilling]]
- [[Codec-based Speech Generation]]
