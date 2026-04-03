IEEE ICME 2025

#### 数据集构建
![](assets/HADPO/file-20260403113750349.png)
- 描述生成；（通过大视觉语言模型进行描述生成）
- 幻觉检测与纠正；（通过GPT-4检测并纠正模型中的幻觉）
- 风格一致数据增强；（通过GPT-4重写样本达到风格一致的数据增强）
- 最后将生成的编好数据集用于后续的HA-DPO模型训练；
#### 方法
##### 多模态幻觉感知DPO （MultiModal Hallucination-Aware DPO）
$$
L_{dpo}(\pi_{\theta};\pi_{ref}) = -E(x_{T},x_{I},y_{pos},y_{neg})\sim D[log \; \sigma(\beta \;log \frac{\pi_{\theta}(y_{pos}|[x_{T},x_{I}])}{\pi_{ref}(y_{pos}|[x_{T},x_{I}])}) - \beta \;log \frac{\pi_{\theta}(y_{neg}|[x_{T},x_{I}])}{\pi_{ref}(y_{neg}|[x_{T},x_{I}])})]
$$

- 其中$x_{T}\;x_{I}$ 表示文本和图像提示；
- $\pi_{ref} \; \pi_{\theta}$ 表示参考模型（奖励模型）和策略模型；
- 该函数目标是让奖励模型偏向于正向响应$y_{pos}$，拒绝响应$y_{neg}$ ;