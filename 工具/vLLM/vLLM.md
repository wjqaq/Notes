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
- 安装并指定cuda环境：
```bash
uv pip install vllm --torch-backend=cu128
```
