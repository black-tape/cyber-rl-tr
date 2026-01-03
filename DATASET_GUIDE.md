# 数据集获取和使用指南

本项目支持两种数据集：NSL-KDD 和 CICIDS2017

## NSL-KDD 数据集（推荐，易于使用）

### 自动下载和处理

```bash
# 1. 下载 NSL-KDD 数据集
python data/dataset_loader.py --dataset nsl_kdd --download

# 2. 预处理数据并创建 Non-IID 分割
python data/dataset_loader.py --dataset nsl_kdd --preprocess --num_clients 3

# 3. 使用真实数据进行联邦训练
python train/train_real_data.py --dataset nsl_kdd --num_clients 3 --num_rounds 50
```

### 数据说明

- **来源**: NSL-KDD 是 KDD Cup 1999 数据集的改进版本
- **大小**: 约 148,000 条训练记录，22,500 条测试记录
- **特征**: 41 个特征（协议类型、服务类型、连接标志等）
- **标签**: 二分类（正常/攻击）

### 处理后的文件结构

```
data/processed/nsl_kdd/
├── client_0_X.npy          # 客户端 0 的特征
├── client_0_y.npy          # 客户端 0 的标签
├── client_1_X.npy
├── client_1_y.npy
├── client_2_X.npy
├── client_2_y.npy
├── X_test.npy              # 测试集特征
└── y_test.npy              # 测试集标签
```

---

## CICIDS2017 数据集（需手动下载）

### 手动下载步骤

由于 CICIDS2017 数据集较大（约 7GB），需要手动下载：

#### 方法 1: 从官方网站下载

1. 访问 [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
2. 下载 CSV 文件（所有日期的数据）
3. 解压并放置到 `data/raw/cicids2017/` 目录

#### 方法 2: 从 Kaggle 下载（推荐）

```bash
# 1. 安装 Kaggle CLI
pip install kaggle

# 2. 配置 Kaggle API（需要先在 Kaggle 网站获取 API 密钥）
# 将 kaggle.json 放置到 ~/.kaggle/

# 3. 下载数据集
kaggle datasets download -d cicdataset/cicids2017 -p data/raw/

# 4. 解压
unzip data/raw/cicids2017.zip -d data/raw/cicids2017/
```

### 预处理 CICIDS2017

```bash
# 检查数据是否存在
python data/dataset_loader.py --dataset cicids2017

# 预处理数据
python data/dataset_loader.py --dataset cicids2017 --preprocess

# 使用数据训练
python train/train_real_data.py --dataset cicids2017 --num_clients 3
```

### 数据说明

- **来源**: 加拿大网络安全研究所 (CIC)
- **大小**: 约 280 万条记录
- **特征**: 78 个特征（流量统计、时间特征等）
- **攻击类型**: 包含多种真实攻击（DDoS, 端口扫描, Botnet 等）

---

## 快速开始示例

### 完整流程（使用 NSL-KDD）

```bash
# 步骤 1: 安装依赖
pip install -r requirements.txt

# 步骤 2: 下载并预处理数据
python data/dataset_loader.py --dataset nsl_kdd --download --preprocess --num_clients 3

# 步骤 3: 运行联邦训练
python train/train_real_data.py \
    --dataset nsl_kdd \
    --num_clients 3 \
    --num_rounds 50 \
    --episodes_per_round 10
```

### 使用模拟数据（无需下载）

如果只想快速测试系统，可以使用内置的模拟数据生成器：

```bash
# 使用模拟数据进行本地训练
python main.py --mode local

# 使用模拟数据进行联邦训练
python main.py --mode federated
```

---

## 数据集对比

| 特性 | NSL-KDD | CICIDS2017 |
|------|---------|------------|
| 大小 | ~170K 条记录 | ~280 万条记录 |
| 下载方式 | 自动下载 | 手动下载 |
| 特征数量 | 41 | 78 |
| 攻击类型 | 4 大类 | 7+ 种真实攻击 |
| 推荐用途 | 快速实验 | 完整论文实验 |

---

## 常见问题

### Q: NSL-KDD 下载失败怎么办？

A: 可以手动从 GitHub 下载：
```bash
mkdir -p data/raw/nsl_kdd
cd data/raw/nsl_kdd

# 下载训练集
wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt -O train.txt

# 下载测试集
wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt -O test.txt
```

### Q: CICIDS2017 文件太大如何处理？

A: 可以只使用部分日期的数据进行实验，或者在预处理时进行采样：

```python
# 在 dataset_loader.py 中添加采样
df = df.sample(frac=0.1, random_state=42)  # 只使用 10% 的数据
```

### Q: Non-IID 数据分割是如何实现的？

A: 为每个客户端分配不同比例的攻击样本，模拟真实场景中不同节点面临的攻击分布差异：

- 客户端 0: 20% 攻击
- 客户端 1: 40% 攻击
- 客户端 2: 60% 攻击

---

## 数据预处理细节

所有数据集都会经过以下预处理步骤：

1. **特征编码**: 类别特征转换为数值
2. **标准化**: StandardScaler 归一化
3. **缺失值处理**: 填充或删除
4. **标签转换**: 多分类转为二分类（正常/攻击）
5. **Non-IID 分割**: 按攻击比例分配给不同客户端

---

## 数据使用示例代码

```python
import numpy as np
from pathlib import Path

# 加载预处理后的数据
data_dir = Path('data/processed/nsl_kdd')

# 加载客户端 0 的数据
X_train = np.load(data_dir / 'client_0_X.npy')
y_train = np.load(data_dir / 'client_0_y.npy')

# 加载测试数据
X_test = np.load(data_dir / 'X_test.npy')
y_test = np.load(data_dir / 'y_test.npy')

print(f"训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")
print(f"攻击比例: {np.mean(y_train):.2%}")
```

---

## 下一步

1. 完成数据下载和预处理
2. 运行基线实验（单 Agent 训练）
3. 运行联邦学习实验
4. 对比传统 FedAvg 和强化型聚合的性能
