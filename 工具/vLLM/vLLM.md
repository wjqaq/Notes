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
