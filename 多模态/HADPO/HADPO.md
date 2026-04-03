IEEE ICME 2025

#### 数据集构建
![](assets/HADPO/file-20260403113750349.png)
- 描述生成；（我们从VG数据集中随机选择图像，并使用LVLM生成相应的详细描述）
- 幻觉检测与纠正；（将模型生成的描述和原始图像的所有注释信息输入到GPT-4中，并提供详细的提示模板，使GPT-4能够检查生成的描述中是否存在幻觉。如果存在幻觉，则需要提供没有幻觉的纠正后描述。通过这种方式，我们可以获得与图像对应的正向和负向响应）
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
隐式表示奖励模型$\hat r$：
$$
\hat r (x_T,x_{I},y) = \beta \; log \frac{\pi_{\theta}(y|[x_{T},x_{I}])}{\pi_{ref}(y|[x_{T},x_{I}])}
$$
- 最大化奖励边际$\hat r(x_{T},x_{I},y_{pos}) - \hat r(x_{T},x_{I},y_{neg})$，有效发大了$y_{pos}$的对数似然，缩小了$y_{neg}$的对数似。
辅助任务，将监督微调的梯度集成到偏好学习中：
$$
L_{aux} = -\sum log \;P(y|x_{P};\pi_{\theta}),\{x_{P},y\} \sim D_{sft}
$$
- $x_{P} \; y$ 是提示和相关的响应，$D_{sft}$表示sft训练阶段的数据集；
$$
L = L_{dpo} + \lambda L_{aux}
$$
用$\lambda$平衡偏好学习损失和辅助语言建模损失。

