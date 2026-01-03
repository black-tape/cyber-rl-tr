# GPU 版本安装说明

## 系统检测
- GPU: NVIDIA GeForce (12GB 显存)
- CUDA Version: 12.0
- Driver Version: 528.01

## 安装 PyTorch GPU 版本

### 方式 1: 使用 conda (推荐)

```bash
# 激活环境
conda activate mypaper

# 安装 PyTorch GPU 版本 (CUDA 11.8 兼容 CUDA 12.0)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装其他依赖
pip install numpy pandas scikit-learn pyyaml gym
```

### 方式 2: 使用 pip

```bash
# 激活环境
conda activate mypaper

# 安装 PyTorch GPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install numpy pandas scikit-learn pyyaml gym
```

### 方式 3: 使用清华镜像 (国内推荐)

```bash
# 先安装其他依赖
pip install numpy pandas scikit-learn pyyaml gym -i https://pypi.tuna.tsinghua.edu.cn/simple

# 从官方源安装 PyTorch GPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 验证安装

安装完成后，运行以下命令验证 GPU 是否可用：

```python
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

预期输出：
```
PyTorch version: 2.x.x+cu118
CUDA available: True
CUDA version: 11.8
GPU name: NVIDIA GeForce RTX XXXX
```

## 快速安装脚本

我已经为您创建了一个自动安装脚本，运行：

```bash
# Windows (PowerShell 或 CMD)
C:\Users\THUNDEROBOT\miniconda3\envs\mypaper\python.exe install_gpu.py

# 或者直接运行
python install_gpu.py
```

## 注意事项

1. CUDA 11.8 版本的 PyTorch 可以在 CUDA 12.0 环境下运行
2. 如果需要 CUDA 12.x 的原生支持，可以安装：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. 训练时会自动检测并使用 GPU（无需修改代码）
