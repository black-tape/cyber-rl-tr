"""
自动安装 GPU 版本的依赖
适用于 CUDA 12.0 环境
"""

import subprocess
import sys

def run_command(cmd):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"执行: {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("错误信息:", result.stderr)
    return result.returncode

def main():
    print("="*60)
    print("GPU 版本依赖自动安装")
    print("="*60)

    # 检查 Python 版本
    print(f"\nPython 版本: {sys.version}")

    packages = [
        # 其他依赖 (使用清华镜像)
        "pip install numpy pandas scikit-learn pyyaml gym -i https://pypi.tuna.tsinghua.edu.cn/simple",

        # PyTorch GPU 版本 (CUDA 11.8, 兼容 CUDA 12.0)
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    ]

    for i, pkg in enumerate(packages, 1):
        print(f"\n[{i}/{len(packages)}] Installing...")
        ret = run_command(pkg)
        if ret != 0:
            print(f"[X] Installation failed: {pkg}")
            return False
        else:
            print(f"[OK] Installation successful")

    # Verify installation
    print("\n" + "="*60)
    print("Verifying GPU Support")
    print("="*60)

    verify_code = """
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Count: {torch.cuda.device_count()}')
else:
    print('Warning: CUDA not available, using CPU')
"""

    ret = run_command(f'python -c "{verify_code}"')

    if ret == 0:
        print("\n" + "="*60)
        print("[OK] All dependencies installed successfully!")
        print("="*60)
        print("\nYou can start training now:")
        print("  python main.py --mode federated")
        return True
    else:
        print("\n[X] Verification failed")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
