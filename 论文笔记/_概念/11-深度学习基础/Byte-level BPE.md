---
type: concept
aliases: [Byte-Pair Encoding, Byte-level BPE, BBPE]
---

# Byte-level BPE

## 定义
字节级 Byte-Pair Encoding，一种在字节序列上应用 BPE 的子词分词算法，无需预定义词汇表即可处理任意文本，由 GPT-2 首次引入大规模使用。

## 数学形式
BBPE 直接在 UTF-8 字节序列上应用 BPE 合并操作，逐次合并出现频率最高的字节对，直到达到预设词汇量。

## 核心要点
1. 基于字节级编码，天然支持所有语言和特殊字符（不会出现 UNK）
2. Qwen3 使用的 tokenizer 词汇量为 151,669
3. 相比字符级 BPE，在处理多语言（尤其非拉丁字符）时更鲁棒

## 代表工作
- [[Qwen3]]: BBPE tokenizer，词汇量 151,669

## 相关概念
- [[Tokenizer]]
- [[SentencePiece]]
