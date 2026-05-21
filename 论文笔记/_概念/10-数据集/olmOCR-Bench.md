---
type: concept
aliases: [olmOCR-Bench, olmocrbench]
---

# olmOCR-Bench

## 定义
基于单元测试的 PDF 内容提取评测框架，旨在解锁万亿级 token 的 PDF 文档处理，强调大规模、多样化文档的 OCR 鲁棒性。

## 核心要点
1. 包含多布局、多语言、多复杂度级别的 PDF 语料库
2. 子任务维度：AI、Old Scans、Tables、Headers & Footers、Multi Column、Long Tiny Text、Math、Overall
3. 单元测试式评估，精确衡量每类文档结构的处理能力
4. Multi Column 和 Long Tiny Text 是最具挑战性的子集
5. 评估指标包括文本识别准确率、布局解析精度、token 级对齐准确性

## 代表工作
- [[RTPrune]]: olmOCR-Bench 上以 84% token 保留 97.88% 精度

## 相关概念
- [[OmniDocBench]]
- [[Ocean-OCR]]
