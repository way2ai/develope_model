## 介绍


## 基础环境搭建
基础环境核心是pytorch

### 1.创建虚拟环境
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

### 2.*编辑uv配置
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
url = "https://mirrors.nju.edu.cn/pytorch/whl/cu124"
explicit = true

# *指定源（保证能稳定拿到 CUDA 版 PyTorch）
# 固定到 PyTorch 的 cu124，因为要用到cuda加速
# 如果不指定可能装到 CPU wheel，或装到非你期望的构建/版本，或某些环境下直接下载失败/版本不全
[tool.uv.sources]
torch = [{ index = "pytorch-cu124" }]
torchvision = [{ index = "pytorch-cu124" }]
torchaudio = [{ index = "pytorch-cu124" }]
~~~

### 3.安装和验证
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

## 可选额外依赖
~~~bash
# transformers
uv add transformers accelerate safetensors

# 量化
uv add compressed-tensors

# fastapi
uv add fastapi uvicorn
~~~

## 环境、工具、依赖解释
cuda：
NVIDIA 推出的并行计算平台和编程模型，让 GPU 能解决复杂的计算问题。

cuda-toolkit:
开发 CUDA 程序所需的工具包，包含编译器 (nvcc)、调试器和基础库。

cudnn:
专门为深度神经网络设计的加速库（卷积、池化等操作的优化实现）

torch:
torch 是深度学习框架。它封装了底层复杂的 CUDA/cuDNN 代码。
用 uv 或者 pip 从 https://download.pytorch.org/whl/cu124 安装的 torch==2.6.0+cu124 这类 CUDA wheel，自带 CUDA 运行时相关库（CUDA runtime / cuDNN / cuBLAS 等一整套依赖）。只需要你的机器上有 NVIDIA 驱动，且版本要足够新，能支持运行对应 CUDA 版本构建的程序

transformers：
一种深度学习模型架构。它是算法设计的蓝图（GPT、BERT 都基于此）。
transformers 库里直接内置了 Llama, Qwen, BERT 等主流模型的架构代码。它还包含 Tokenizer（分词器），负责把人类语言（"你好"）转换成 PyTorch 能看懂的数字（[101, 872, ...]）。

vllm：
专门用于**大模型推理（Serving）**的高性能库。它比 PyTorch 原生推理更快，吞吐量更高。

accelerate：
它能自动帮你搞定**“大模型拆分”**。它会自动把模型切块，一部分放 GPU 1，一部分放 GPU 2，剩下的放 CPU 内存，甚至硬盘。没有它，在单卡上跑大模型非常痛苦。
1. 你传了 device_map="auto"（自动分配显存）。
2. transformers 知道这个功能是 PyTorch 做不到的，必须依赖 accelerate 库。
3. 它内部会尝试 import accelerate。
4. 如果装了：它就把模型切块，自动分配给 GPU 和 CPU。
5. 如果没装：程序直接崩溃，报错：You need to install 'accelerate' to use the device_map argument.

safetensors：
HuggingFace 推出的新标准格式。加载速度比 PyTorch 原生快几十倍（使用内存映射技术），且绝对安全。现在几乎所有开源大模型都用这个格式存储权重。
1. transformers 去文件夹里看，发现有一个 model.safetensors 文件。
2. transformers 内部检查：“用户装了 safetensors 库吗？”
3. 如果装了：它就调用 safetensors 的 C++ 接口极速加载模型（无需你 import）。
4. 如果没装：它会报错，提示你 ImportError:safetensorsnot installed，或者尝试去加载慢速的 .bin 文件。

compressed-tensors：
专门处理模型的稀疏化（Sparsity）和量化（Quantization）。它能让模型体积缩小 4 倍，且保持精度几乎不变，让消费级显卡也能跑大模型。
1. 如果模型的 config.json 里有：quantization_config.quant_method = "compressed-tensors"
transformers 在 AutoModelForCausalLM.from_pretrained(...) 过程中会读取这个配置，然后内部调用量化/解压相关逻辑。
2. 这段内部逻辑会在需要时 动态导入（importlib.import_module(...) / 延迟 import）compressed_tensors。
3. 如果环境里没装，就像你之前那样在加载阶段抛 ImportError；装上后，Transformers 内部能导入到它，你的代码无需再写 import compressed_tensors。

fastapi：
定义了 API 接口

uvicorn：
它是一个高性能的 ASGI 服务器。它监听网络端口（如 8000），接收外部的 HTTP 请求，然后转交给 FastAPI 处理。
