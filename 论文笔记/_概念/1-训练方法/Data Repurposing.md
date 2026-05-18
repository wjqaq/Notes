---
type: concept
aliases: [Data Repurposing, 数据重利用]
---

# Data Repurposing

## 定义
一种自动化数据合成策略：从公开领域的高质量文本（如文学作品、百科条目）出发，由 LLM 生成对应指令，将原文作为响应，构建演示数据。

## 核心要点
1. 针对文学创作等需要专业技能的领域，人工标注困难
2. 从公开领域收集高质量文学作品作为"黄金响应"
3. LLM 根据内容生成不同粒度（详细/简洁）的指令
4. 也可用于角色扮演数据：从 Wikipedia 提取角色档案，生成指令和响应

## 代表工作
- [[Qwen2]]: 后训练数据构建中用于文学创作和角色扮演数据的合成

## 相关概念
- [[Instruction Evolution]]
- [[Supervised Fine-Tuning]]
