---
date: 2025-12-17T16:36:00
---
#### Transformers及相关库
- Transformer：核心库、模型加载、模型训练、流水线等；
- Tokenizer：分词器，对数据进行预处理，文本到Token序列的相互转换；
- Datasets：数据集库，提供了数据集加载、处理等方法；
- Evaluate：评估函数，提供各种指标的计算函数；
- PEFT：高效微调模型的库，提供了几种高效微调的方法，小参数量撬动大模型；
- Accelerate：分布式训练，提供了分布式训练解决方案，包括大模型的加载与推理解决方案；
- Optimum：优化加速库，支持多种后端，如Onnxruntime、Openvino等；
- Gradio：可视化部署库，几行代码可快速实现基于Web交互的算法演示系统。

```python
import torch
print(torch.cuda.is_available())
```