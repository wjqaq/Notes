---
category: 4-评估方法
aliases: [Visual Correlation]
---

# Visual Correlation (视觉相关性)

TPR 中用于 topic 聚类的双重标准之一。利用 VLM 的视觉编码器（如 CLIP）提取文本和图像嵌入，计算语义单元相似度向量之间的 Pearson 相关系数，判断两个语义单元是否在视觉上指向图像的相同区域。即使文本描述相似，如果它们指向图像中不同的实体，会被归类到不同的 topic。

## 代表工作
- [[TPR]]: 使用视觉相关性 + 文本一致性的双重标准进行 topic 聚类
