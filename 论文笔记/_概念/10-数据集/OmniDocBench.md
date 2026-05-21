---
type: concept
aliases: [OmniDocBench, omnidocbench]
---

# OmniDocBench

## 定义
面向多样化 PDF 文档解析的综合评测基准，覆盖多文档类型、多语言、多内容模态，评估实际 OCR 场景性能。

## 核心要点
1. 包含中英文文档，涵盖书籍、幻灯片、财报、学术论文、报纸等类型
2. 评测指标：Text Edit Distance (↓)、Formula CDM (↑)、Table TEDS/TEDS-S (↑)、Read Order Edit Distance (↓)、Overall (↑)
3. Overall 计算公式：$\text{Overall} = \frac{(1-\text{Text Edit}) \times 100 + \text{Table TEDS} + \text{Formula CDM}}{3}$
4. 评估模型的 Markdown 输出与 ground truth 的匹配度
5. v1.5 版本，是目前最广泛使用的文档解析评测之一

## 代表工作
- [[RTPrune]]: OmniDocBench 上 DeepSeek-OCR-Large 达到 99.47% 精度（含 token 剪枝）

## 相关概念
- [[olmOCR-Bench]]
- [[Ocean-OCR]]
