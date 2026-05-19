---
type: concept
aliases: [图像检索, Image Retrieval, 相似图片搜索]
---

# Image Retrieval (图像检索)

## 定义
基于语义相似度从大规模图像数据库中查找与查询图像最相关的图片的技术。

## 数学形式
两幅图像 $v_1, v_2$ 的余弦相似度：
$$s = \left\langle \frac{f_p(v_1)}{\|f_p(v_1)\|}, \frac{f_p(v_2)}{\|f_p(v_2)\|} \right\rangle$$

其中 $f_p(\cdot)$ 为图像编码器（如 [[CLIP]] vision encoder）。

## 核心要点
1. 核心流程: 图像编码 $\to$ 向量索引 $\to$ 相似度搜索 $\to$ top-k 返回
2. 常用编码器: [[CLIP]]-ViT、DINOv2、ResNet
3. 常用向量检索库: [[FAISS]]（Facebook AI Similarity Search）
4. 相似度度量: [[Cosine Similarity|余弦相似度]]、欧氏距离、内积
5. 在 [[Re-Align]] 中用于检索语义相似但细节不同的图片，构建视觉偏好信号

## 代表工作
- [[Re-Align]]: 检索相似图诱导 VLM 幻觉
- MRAG: 多模态检索增强生成

## 相关概念
- [[CLIP]]
- [[FAISS]]
- [[Cosine Similarity]]
- [[Vision Language Model|VLM]]
