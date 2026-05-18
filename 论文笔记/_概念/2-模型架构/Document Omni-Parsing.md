---
type: concept
aliases: [文档全解析, Universal Document Parsing]
---

# Document Omni-Parsing

## 定义
一种统一的文档理解范式，使用通用模型（而非分离的专用模块流水线）一次性完成文档的版面分析、文本提取、图表解释和格式转换。

## 核心要点
1. Qwen2.5-VL 使用 HTML 格式统一表示所有文档元素
2. 支持的文档元素：段落、表格、图表、公式、化学式、乐谱、自然/合成图像
3. HTML 中包含布局框坐标 (data-bbox) 和阅读顺序信息
4. 实现 OCRBench_v2 上领先 Gemini 1.5 Pro 英文 9.6%、中文 20.6%

## QwenVL HTML 格式示例
```html
<p data-bbox="x1 y1 x2 y2"> text </p>
<table data-bbox="..." class="table{id}"> ... </table>
<div class="chart" data-bbox="..."> <img/> <table> ... </table></div>
<div class="formula" data-bbox="..."> <img/> <div> formula </div></div>
```

## 代表工作
- [[Qwen2.5-VL]]: 引入 QwenVL HTML 格式实现文档全解析

## 相关概念
- [[Visual Grounding]]
- [[Native Dynamic Resolution]]
- [[LVLM]]
