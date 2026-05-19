---
type: concept
aliases: [Facebook AI Similarity Search]
---

# FAISS

## 定义
Facebook AI Research 开发的高效向量相似度搜索库，支持十亿级向量索引和毫秒级检索。

## 数学形式
给定查询向量 $q$ 和数据库向量集 $\{k_i\}$，检索 top-k 最近邻：
$$\text{top-k} = \arg\max_{i}^{(k)} \langle q, k_i \rangle$$

## 核心要点
1. 支持多种索引类型: Flat（暴力）、IVF（倒排）、HNSW（图索引）、PQ（乘积量化）
2. 支持 GPU 加速搜索
3. 在 [[Re-Align]] 中用于从训练集图片中检索语义相似的 top-10 图片
4. 是多数 RAG 和多模态检索系统的核心基础设施

## 代表工作
- [[Re-Align]]: 在偏好数据构建中做图像检索
- 各类 RAG 系统

## 相关概念
- [[Image Retrieval|图像检索]]
- [[CLIP]]
- [[Cosine Similarity]]
