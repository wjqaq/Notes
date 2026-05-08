CVPR 2026

#### 现象&&分析
![](assets/TIPSv2(Advancing%20Vision-Language%20Pretraining%20with%20Enhanced%20Patch-Text%20Alignment)/file-20260505161527166.png#pig_center)
作者发现，老师模型（ViT-g）比学生模型（ViT-L）在图像细节对齐上弱;
通过消融实验发现：
- 掩码比例;
- 网络初始化状态;
#### 方法
掩码图像建模：
![](assets/TIPSv2(Advancing%20Vision-Language%20Pretraining%20with%20Enhanced%20Patch-Text%20Alignment)/file-20260505162109376.png)
传统的iBOT为了去猜测被掩盖的token而忽略了可见的token，提出改进方法iBOT++：
![](assets/TIPSv2(Advancing%20Vision-Language%20Pretraining%20with%20Enhanced%20Patch-Text%20Alignment)/file-20260505162310164.png#pig_center)
不仅要猜测掩码部分，还强制锚定可见部分的特征与教师部分对齐。
但是算法上要监督所有token，那么模型训练的计算量和显存压力会变得非常大，因此提出了工程优化Head-only EMA：
![](assets/TIPSv2(Advancing%20Vision-Language%20Pretraining%20with%20Enhanced%20Patch-Text%20Alignment)/file-20260505162713320.png)
传统的EMA更新需要为教师模型维护一份完整的参数副本，防止模型坍塌;不仅有自监督损失还有图文对比学习损失，其本身就可以防止模型的特征坍塌，仅在轻量级投影头上保留EMA更新。
