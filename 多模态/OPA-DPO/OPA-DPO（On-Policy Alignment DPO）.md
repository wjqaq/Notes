CVPR 2025
#### 问题
![](assets/OPA-DPO（On-Policy%20Alignment%20DPO）/file-20260403170657832.png)

- 图a左，黑线表示$\pi_{ref}$ 参考模型，可以看到模型本身就倾向于生成错误的答案，

传统的DPO是离线[Offline RL（离线强化学习）](../相关概念.md#Offline%20RL（离线强化学习）)异策略[Off-Policy（异策略）](../相关概念.md#Off-Policy（异策略）)，其行为策略和目标策略的数据集分布往往是偏离的。


#### 方法
