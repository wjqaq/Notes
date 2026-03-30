#### uv
![](assets/vLLM/file-20260319150252836.png)
用rust编写的Python包和项目管理器；
##### 1. Linux安装命令：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- 如果出现提示，需要将其配置环境变量：
```bash
$HOME/.local/bin/.env
```
检查最后一行出现这句没，如果出现了，就source ~/.bashrc即可，没出现先添加进去，再执行source ~/.bashrc
##### 2. 创建uv虚拟环境:
```bash
uv venv --python 3.11 --seed --managed-python
```
- 激活虚拟环境
```bash
source .venv/bin/activate
```
- 安装并指定cuda环境：
```bash
uv pip install vllm --torch-backend=cu128
```


#### 适配vllm cuda pytorch flash-attn
注意其版本适配。
```bash
uv pip install vllm torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0   --index-url https://pypi.tuna.tsinghua.edu.cn/simple   --extra-index-url https://download.pytorch.org/whl/cu128   --index-strategy unsafe-best-match
```
```bash
uv pip install -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
```bash
uv pip install 
-r requirements/metrics.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
```bash
uv pip install 
 bitsandbytes --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
```bash
wget -c https://ghfast.top/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```
```bash
uv pip install flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl 
```



#### 工作流程
以 “介绍法国首都” 为例，完整路径：
1. 用户发送请求→API 层解析→异步加入队列
2. 输入处理：文本→Token ID（如`[1, 234, 567, 890]`）
3. 调度：被选中加入 batch，分配 KV 缓存块
4. 预填充：GPU 计算完整 prompt 的 KV 缓存，生成首步 logits
5. Logits 处理：应用重复惩罚→温度缩放→TopP 过滤
6. 采样：选择下一个 token（如`"巴"`）
7. 后处理：解码为文本，流式推送给用户
8. 解码循环：重复步骤 4-7，生成`"巴黎是法国的首都..."`
9. 触发停止条件→清理资源→返回完整结果