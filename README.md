

## 1.基础环境搭建
基础环境核心是pytorch

### 1.1.创建虚拟环境
~~~bash
# 创建虚拟环境
conda create -n my_env python=3.12 -y

# 激活环境
conda activate my_env

# cd 项目根目录
cd my_project

# 初始化当前项目 
uv init . --python 3.12
~~~

### 1.2.编辑uv配置
例如：
~~~
[project]
name = "develope-model"
version = "0.1.0"
description = "AI Environment"
requires-python = ">=3.12"
dependencies = [
  "torch>=2.6.0,<2.7.0",
  "torchvision>=0.21.0,<0.22.0",
  "torchaudio>=2.6.0,<2.7.0",
]

# 默认源
[[tool.uv.index]]
name = "pypi-cn"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

# 指定源
[tool.uv.sources]
torch = [{ index = "pytorch-cu124" }]
torchvision = [{ index = "pytorch-cu124" }]
torchaudio = [{ index = "pytorch-cu124" }]
~~~

### 1.3.安装和验证
~~~bash
# 怎么判断某个包要不要走 pytorch-cu124
# 能列出版本：才考虑把它加到 [tool.uv.sources] 绑定该 index
# 列不出版本：不要绑定，走默认 pypi-cn（或 PyPI）即可
python -m pip index versions <包名> -i https://download.pytorch.org/whl/cu124

# 安装依赖
uv sync

# 验证
python -c "import torch, torchvision, torchaudio; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('torchaudio', torchaudio.__version__); print('cuda build', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
~~~

## 2.可选依赖
~~~bash
# transformers
uv add transformers accelerate safetensors

# 量化
uv add compressed-tensors

# fastapi
uv add fastapi uvicorn
~~~