---
type: concept
aliases: [Qwen2 LLM, Qwen2 系列]
---

# Qwen2

## 定义
阿里通义千问团队发布的第二代大型语言模型系列，是 Qwen2-VL 的语言骨干网络。

## 数学形式
标准 Transformer 解码器架构，使用 [[RoPE]] 位置编码 + [[RMSNorm]] + [[SwiGLU]] 激活。

## 核心要点
1. 提供 0.5B / 1.5B / 7B / 72B 四个规模
2. 相比 Qwen1 改进了数据质量和训练配方（长上下文预训练 + 高质量数据过滤）
3. Qwen2-VL 直接复用 Qwen2 预训练权重，将 RoPE 替换为 [[MRoPE]]
4. 支持 32K tokens 上下文窗口

## 代表工作
- [[Qwen2 Technical Report]]: Qwen2 系列技术报告，含密集模型与 MoE 模型
- [[Qwen2-VL]]: 以 Qwen2 为 LLM 骨干的多模态模型
- [[Qwen2.5-VL]]: 升级到 [[Qwen2.5 LLM]]
- [[Qwen3]]: 第三代纯语言模型

## 相关概念
- [[RoPE]]
- [[MRoPE]]
- [[LVLM]]
- [[Qwen2.5 LLM]]
