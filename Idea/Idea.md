根据OPA-DPO，我们知道模型


根据实验我发现：抑制背景能显著提升判别式任务，抑制前景能显著提升生成式任务。


通过实验分析：
- LM15层能定位到物体（相近的物体也会发现），这里决定了后续是否能正确识别物体;
- LM19层能判断其他语义，如颜色，位置

```python
通过实验分析：
- 图片中存在的物体和问题的物体具有高相似度或一致：
	- 图片在LM 15左右一定会定位到该物体附近;
	否则其整体的注意力全局就比较一样，决策就偏向绝对。
	这一步的决策对后续是否有幻觉有关键性作用。
```

```PlainText
之前 (V1-V3):
  稳定性投票 → vote_counts → 直接用投票数做 gate weights
                 ↑ k=1, 每个 channel 选 1 个 patch

V4:
  稳定性投票(k=7) → 池化选中的 patches → stability-pooled CLS proxy
                  → cosine_sim(proxy, 每个patch) → gate weights
                  ↑ 核心：用 LAST-ViT 的 "更好 CLS" 指导 gating
```
#### 数据集分布
##### LLaVA-Instruct-150K
各个数据集分布：

|数据集名称|样本数|唯一图像数|平均对话轮次|平均Human文本长度|平均GPT文本长度|
|---|---|---|---|---|---|
|llava_instruct_150k|157,712|81,479|2.29|57.3|407.1|
|llava_instruct_80k|80,000|80,000|2.77|57.3|372.1|
|conversation_58k|56,681|56,681|4.53|53.4|291.1|
|detail_23k|23,240|23,240|1.00|44.5|609.3|
|complex_reasoning_77k|76,643|76,643|1.00|74.6|736.0|
|llava_v1_5_mix665k|665,298|349,034|5.18|67.7|80.8|

  数据集特点：
  - detail_23k 和 complex_reasoning_77k：单轮对话，GPT回答较长（600-700字符），适合详细描述和复杂推理任务
  - conversation_58k：多轮对话（平均4.5轮），适合对话式交互训练
  - llava_v1_5_mix665k：最大数据集，包含大量简短回答（中位数仅14字符）
###### POPE
各数据集：

| 数据集/分割             | 样本数   | 唯一图像数 | Yes比例 | No比例  | 唯一物体数 |
| ------------------ | ----- | ----- | ----- | ----- | ----- |
| test (default)     | 9,000 | 500   | 50.0% | 50.0% | 79    |
| Full (all splits)  | 9,000 | 500   | 50.0% | 50.0% | 79    |
| Full - adversarial | 3,000 | 500   | 50.0% | 50.0% | 74    |
| Full - popular     | 3,000 | 500   | 50.0% | 50.0% | 79    |
| Full - random      | 3,000 | 500   | 50.0% | 50.0% | 79    |
  数据集特点：
  - 任务类型：物体幻觉评估
  - 问题格式："Is there a [object] in the image?"
  - 答案类型：二分类                             
  - 数据来源：COCO数据集  
  三个分割的区别：
  - adversarial：负样本来自图像中存在的物体，更具挑战性
  - popular：负样本来自高频物体
  - random：负样本随机选择
##### MME
整体统计：
![](assets/Idea/file-20260508112638554.png#pig_center)
类别分布：
![](assets/Idea/file-20260508112706349.png#pig_center)
  数据集特点：    
  - 任务类型：多模态评估基准
  - 评估维度：感知（perception）和认知（cognition）
  - 问题类型：全部为 Yes/No 二分类问题
  - 答案分布：完全平衡，Yes/No 各占50%  

##### AMBER
各数据集：
![](assets/Idea/file-20260508113402479.png#pig_center)
任务分布：

| 任务类型     | 样本数    | 描述       |
| -------- | ------ | -------- |
| 生成任务     | 1,004  | 图像描述生成   |
| 判别任务-存在性 | ~5,000 | 判断物体是否存在 |
| 判别任务-属性  | ~8,000 | 判断物体属性   |
| 判别任务-关系  | ~1,200 | 判断物体关系   |
  数据集特点：
  - 任务类型：生成任务 + 判别任务
  - 评估维度：存在性、属性、关系三个维度
  - 幻觉检测：标注了真实物体和幻觉物体  


#### 测试
在幻觉检测数据集中测试文本token 和 视觉token 对模型推理时的贡献：



#### Qwen2.5-VL 3B
Sample 3000 
###### POPE
```python
===== POPE Results (peft) =====
        random | Acc: 0.8990 | F1: 0.8900 | P: 0.9769 | R: 0.8173
       popular | Acc: 0.8847 | F1: 0.8763 | P: 0.9445 | R: 0.8173
   adversarial | Acc: 0.8670 | F1: 0.8600 | P: 0.9075 | R: 0.8173
       overall | Acc: 0.8836 | F1: 0.8753 | P: 0.9421 | R: 0.8173    
==================================================
POPE Evaluation (Residual LAST-ViT v2)
==================================================
Total POPE samples: 9000
POPE v2: 100%|██████████████████████████████| 9000/9000 [32:37<00:00,  4.60it/s]
        random | Acc: 0.8947 | F1: 0.8839 | P: 0.9845 | R: 0.8020
       popular | Acc: 0.8823 | F1: 0.8721 | P: 0.9555 | R: 0.8020
   adversarial | Acc: 0.8620 | F1: 0.8531 | P: 0.9120 | R: 0.8013
       overall | Acc: 0.8797 | F1: 0.8695 | P: 0.9497 | R: 0.8018
       
       
======================================================================
POPE Three-Way Comparison
======================================================================
    Category |   Baseline |  LAST-ViT v1 |  Residual v2 |     Δ v1 |     Δ v2
----------------------------------------------------------------------
     overall |     0.8798 |       0.8836 |       0.8797 |  +0.0038 |  -0.0001
      random |     0.8913 |       0.8990 |       0.8947 |  +0.0077 |  +0.0034
     popular |     0.8813 |       0.8847 |       0.8823 |  +0.0034 |  +0.0010
 adversarial |     0.8667 |       0.8670 |       0.8620 |  +0.0003 |  -0.0047
```
##### MME
```python
===== MME Results (baseline) =====
  Overall Accuracy: 0.8538 (2027/2374)
                   OCR: 0.7250 (29/40)
               artwork: 0.8150 (326/400)
             celebrity: 0.7912 (269/340)
        code_reasoning: 0.8750 (35/40)
                 color: 0.9500 (57/60)
  commonsense_reasoning: 0.7429 (104/140)
                 count: 0.8667 (52/60)
             existence: 0.9500 (57/60)
              landmark: 0.9325 (373/400)
  numerical_calculation: 0.7750 (31/40)
              position: 0.8500 (51/60)
               posters: 0.9082 (267/294)
                 scene: 0.8400 (336/400)
      text_translation: 1.0000 (40/40)
===== MME Results (peft) =====
  Overall Accuracy: 0.8433 (2002/2374)
                   OCR: 0.6750 (27/40)
               artwork: 0.7375 (295/400)
             celebrity: 0.7588 (258/340)
        code_reasoning: 0.8750 (35/40)
                 color: 0.9000 (54/60)
  commonsense_reasoning: 0.7786 (109/140)
                 count: 0.8833 (53/60)
             existence: 0.9667 (58/60)
              landmark: 0.9450 (378/400)
  numerical_calculation: 0.8000 (32/40)
              position: 0.8333 (50/60)
               posters: 0.9048 (266/294)
                 scene: 0.8675 (347/400)
      text_translation: 1.0000 (40/40)
      
==================================================
MME Evaluation (Residual LAST-ViT v2)
==================================================
 Overall: 0.8614 (2045/2374)
                   OCR: 0.7500 (30/40)
               artwork: 0.8000 (320/400)
             celebrity: 0.8353 (284/340)
        code_reasoning: 0.8500 (34/40)
                 color: 0.9500 (57/60)
  commonsense_reasoning: 0.7857 (110/140)
                 count: 0.8833 (53/60)
             existence: 0.9500 (57/60)
              landmark: 0.9350 (374/400)
  numerical_calculation: 0.7250 (29/40)
              position: 0.8500 (51/60)
               posters: 0.9082 (267/294)
                 scene: 0.8475 (339/400)
      text_translation: 1.0000 (40/40)
======================================================================
MME Three-Way Comparison
======================================================================
      Metric |   Baseline |  LAST-ViT v1 |  Residual v2 |     Δ v1 |     Δ v2
----------------------------------------------------------------------
     overall |     0.8538 |       0.8433 |       0.8614 |  -0.0105 |  +0.0076
  perception |     0.8595 |       0.8448 |       0.8666 |  -0.0147 |  +0.0071
   cognition |     0.8077 |       0.8308 |       0.8192 |  +0.0231 |  +0.0115
```
##### AMBER
```python
--- Baseline ---
  Generative: CHAIR=0.0070 Cover=0.6116 Hal=0.0372 Cog=0.7872
  Discriminative: Overall=0.8760 Existence=0.9580 Attr=0.8349 Relation=0.9647
  
--- LAST-ViT v1 ---
  Generative: CHAIR=0.0099 Cover=0.4219 Hal=0.0134 Cog=0.7042
  Discriminative: Overall=0.8470 Existence=0.9147 Attr=0.8064 Relation=0.8824
  
--- Residual v2 ---
  Generative: CHAIR=0.0111 Cover=0.4232 Hal=0.0139 Cog=0.7047
  Discriminative: Overall=0.8645 Existence=0.9441 Attr=0.8294 Relation=0.9059
===========================================================================
AMBER Three-Way Comparison
===========================================================================
                      Metric |   Baseline |  LAST-ViT v1 |  Residual v2
---------------------------------------------------------------------------
Generative CHAIR (lower better) |     0.0070 |     0.0099 ★ |     0.0111
            Generative Cover |     0.6116 |     0.4219   |     0.4232 ★
Generative Hal (lower better) |     0.0372 |     0.0134 ★ |     0.0139
              Generative Cog |     0.7872 |     0.7042   |     0.7047 ★
       Discrim Existence Acc |     0.9580 |     0.9147   |     0.9441 ★
       Discrim Attribute Acc |     0.8349 |     0.8064   |     0.8294 ★
        Discrim Relation Acc |     0.9647 |     0.8824   |     0.9059 ★
         Discrim Overall Acc |     0.8760 |     0.8470   |     0.8645 ★
```





```python
toolName: view_files
            
status: success
          
            
filePath: /home/jay/Desktop/code/Project/outputs/cross_modal_match/POPE/adversarial/sample_adversarial_0122/match_summary.txt
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/jay/Desktop/code/Project/outputs/cross_modal_match/POPE/adversarial/sample_adversarial_0122/info.json
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/jay/Desktop/code/Project/outputs/cross_modal_match/POPE/adversarial/sample_adversarial_0122/match_details.json
          
非常好的问题！让我来分析数据揭示的两个关键现象。

---

## 分析：为什么 "spoon" 匹配到了奇怪的区域

### 现象一：Tokenization 分词问题

LLaMA 的 tokenizer 是 **BPE (Byte-Pair Encoding)** ，它不按单词边界分词。`"spoon"` 被拆成了两个 token：

| Token 编号 | Token 文本 | 
|-----------|-----------|
| 10 | `sp` |
| 11 | `oon` |

这不是我们的代码有问题，而是 LLaMA tokenizer 的原生行为。同一个 `sp` 也可能出现在 `spot`, `spider`, `special` 等词中，`oon` 可能出现在 `soon`, `moon`, `balloon`, `afternoon` 中。**子词 token 没有独立的语义**。

### 现象二：两个 token 的最佳匹配图像区域完全不同

从 [match_details.json](file:///home/jay/Desktop/code/Project/outputs/cross_modal_match/POPE/adversarial/sample_adversarial_0122/match_details.json) 的 Text→Image 匹配数据：

| Token | 最佳 Patch 坐标 | 相似度 | 排名 |
|-------|----------------|--------|------|
| `oon` | **(23, 23)** — 右下角 | **0.716** | 第6名 |
| `sp` | **(3, 1)** — 左上角 | **0.402** | 第16名 |

在 `token_overlay.png` 中，我们只画相似度最高的前30个 token。`oon` 排第6，被标注在右下角；`sp` 排第16，被标注在左上角。

### 根本原因分析

这揭示了一个深层问题：**子词 token 在 LLM 隐藏空间中并没有与对应图像的 semantic region 对齐**。

1. **`oon` (sim=0.716 排第6) 为什么在右下角？**
   
   同一个 `oon` 子词还出现在 LLM prompt 模板的 `ASSISTANT:` 中——`ASS` + `IST` + `ANT` 是三个独立的特殊 token（你可以在 `text_tokens` 列表中看到 `"SS"`, `"IST"`, `"ANT"` 在索引 17-19）。LLM 隐藏空间中的 `oon` 表示和其他高频出现的 `oo`/`on` 相关 token 混在一起，它的最佳匹配实际上是被这些高频上下文 token 的分布"劫持"了。右下角（row=23）是 LLM 注意力最集中的区域，几乎所有高频功能词的最佳匹配都落在这里。

2. **`sp` (sim=0.402 排第16) 为什么在左上角？**

   0.402 的相似度已经非常低了（在所有有意义的 token 中排倒数）。这说明 `sp` 这个子词在隐藏空间中的表示和图像 patch 几乎无关联——**它没有独立的视觉语义**。左上角本质上是一个随机的低相似度匹配。

3. **真正的 "spoon" 应该匹配到哪里？**

   从 I→T 方向看，整个 Top-80 匹配中**没有出现 `sp` 和 `oon`**！前80名全被 `in`, `there`, `image`, `the` 等模板功能词占据：
   ```
   Top I→T: 全是 in / there / image / the  — 没有 spoon 相关 token
   ```
   这说明图像中确实存在一个与 `spoon` 概念相关的区域，但它的语义被功能词的"背景噪音"淹没了——**这些功能词在每句话中都出现，它们的隐藏表示已经吸收了大量与图像无关的统计模式**。

### 结论

`token_overlay.png` 中将 `sp`/`oon` 标在左上角/右下角，是因为：

- **子词 token 没有独立视觉语义** → 相似度要么很低（如 `sp` 的 0.402），要么被高频上下文混淆（如 `oon` 的 0.716 落在所有 token 共享的高密度右下角区域）
- **功能词 (`in`, `there`, `the`, `image`) 霸占了所有高相似度匹配**，它们的 semantic headroom 太强，遮盖了内容词

这就是基于 **LM 隐藏状态**做跨模态匹配的固有局限：token 表示在 LM 层已经被语境大量修改，子词的视觉对齐信号被稀释了。如果想看到真正的视觉-语义对齐，需要回到 **视觉编码器输出** 或 **投影器之后、LM 之前** 的表示空间来做分析，而不是在 LM 最后一层。
```