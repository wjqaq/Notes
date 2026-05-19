---
type: concept
aliases: [Sentence-BERT, SBERT]
---

# SentenceTransformer

## 定义
基于 Siamese BERT 网络的句子嵌入模型，将文本映射到语义向量空间，使语义相似的句子在向量空间中距离更近。

## 数学形式
$$\mathbf{e} = \text{pooling}(\text{BERT}(x))$$

其中 $\mathbf{e}$ 为固定维度的句子嵌入向量。

## 核心要点
1. 在 BERT 基础上使用 siamese/triplet 网络结构微调，使句子级嵌入更具语义意义
2. 常用模型: all-mpnet-base-v2、all-MiniLM-L6-v2
3. [[Re-Align]] 使用 all-mpnet-base-v2 计算 chosen 响应与候选 rejected 响应的余弦相似度
4. 是 RAG、语义搜索、文本聚类等任务的基础组件

## 代表工作
- [[Re-Align]]: 判断候选幻觉与 chosen 响应的语义差异
- 各类语义搜索系统

## 相关概念
- [[Cosine Similarity]]
- [[CLIP]]
