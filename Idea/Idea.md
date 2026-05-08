根据OPA-DPO，我们知道模型


根据实验我发现：抑制背景能显著提升判别式任务，抑制前景能显著提升生成式任务。


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