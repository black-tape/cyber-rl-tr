# Multi-Agent Federated Learning for Network Intrusion Detection

**Lightweight multi-agent system with adaptive federated aggregation for robust network security.**

[English](#english) | [中文](#chinese)

---

## <a name="english"></a>English

### Overview

A federated learning system designed for network intrusion detection with:
- **Lightweight Multi-Agent Architecture**: Simple, interpretable agents for anomaly detection, trust management, and adaptive aggregation
- **Robust Baselines**: Includes Krum, Trimmed Mean, and Median aggregation for Byzantine robustness
- **Attack-Resistant**: Tested against label flipping, model scaling, and backdoor attacks
- **Non-IID Data Support**: Dirichlet-based partitioning for extreme data heterogeneity

### Key Features

- Simple statistical agents (no complex RL) - **interpretable and lightweight**
- Dynamic aggregation weights based on anomaly scores and trust
- Comprehensive baselines for fair comparison
- Complete attack simulation framework
- Automatic success criteria validation

### Project Structure

```
mypaper/
├── agent/
│   ├── simple_agents.py           # Lightweight multi-agent system
│   │   ├── AnomalyDetectionAgent  # Statistical anomaly detection
│   │   ├── TrustAgent             # EMA-based reputation tracking
│   │   └── AggregationAgent       # Adaptive weight computation
│   ├── policy_net.py              # [Legacy] Policy networks
│   ├── rl_agent.py                # [Legacy] RL agents
│   └── multi_agent_controller.py  # [Legacy] Multi-agent controller
│
├── baseline/
│   └── robust_aggregation.py      # Robust aggregation methods
│       ├── KrumAggregator         # Nearest-neighbor selection
│       ├── TrimmedMeanAggregator  # Trim extreme values
│       ├── MedianAggregator       # Coordinate-wise median
│       └── FedProxAggregator      # Weighted averaging
│
├── attack/
│   └── poisoning.py               # Attack models
│       ├── LabelFlippingAttack    # Data poisoning
│       ├── ModelScalingAttack     # Gradient manipulation
│       ├── BackdoorAttack         # Backdoor injection
│       └── AttackCoordinator      # Unified attack management
│
├── data/
│   ├── dataset_loader.py          # NSL-KDD & CICIDS2017 loader
│   └── non_iid.py                 # Non-IID partitioning
│       ├── DirichletPartitioner   # Dirichlet distribution
│       ├── LabelSkewPartitioner   # Pathological non-IID
│       └── CombinedPartitioner    # Label + quantity skew
│
├── federated/
│   ├── client.py                  # Federated client
│   └── server.py                  # Federated server
│
├── train/
│   ├── train_mve.py               # [NEW] Minimum Viable Experiment
│   ├── train_local.py             # Local training
│   ├── train_federated.py         # Federated training
│   └── train_real_data.py         # Real dataset training
│
├── env/
│   └── security_env.py            # Security detection environment
│
├── reward/
│   └── reward_fn.py               # Reward function design
│
├── config/
│   └── config.yaml                # Configuration
│
├── CLAUDE.md                      # Project redesign guide
├── EXPERIMENT_REDESIGN.md         # Problem diagnosis & solutions
├── DATASET_GUIDE.md               # Dataset usage guide
├── main.py                        # Main entry point
└── requirements.txt               # Dependencies
```

### Quick Start

#### 1. Install Dependencies

```bash
conda create -n mypaper python=3.10
conda activate mypaper
pip install -r requirements.txt
```

#### 2. Run Minimum Viable Experiment (MVE)

```bash
python train/train_mve.py
```

This runs a focused experiment comparing:
- FedAvg (Baseline)
- Krum (Byzantine-robust)
- Trimmed Mean (Byzantine-robust)
- **Ours (Multi-Agent)** - Anomaly + Trust + Adaptive Aggregation

**Experimental Setup:**
- 5 clients, 40% malicious (2 Byzantine clients)
- Dirichlet α = 0.1 (extreme Non-IID)
- Label Flipping attack (50% labels flipped)
- 50 federated rounds

**Success Criteria:**
- F1 Score improvement ≥ 5% over FedAvg
- Convergence speedup ≥ 20%

#### 3. View Results

Results are saved to:
- `mve_results.png` - Comparison plots
- Console output - Summary table with improvements

### Core Algorithm

#### Multi-Agent Adaptive Aggregation

**Step 1: Anomaly Detection**
```python
anomaly_score = f(L2_norm, cosine_similarity, peer_distance)
```

**Step 2: Trust Management**
```python
trust[i] = α * trust[i] + (1-α) * current_performance
```

**Step 3: Adaptive Weights**
```python
score[i] = trust[i] / (1 + anomaly[i])
weight[i] = softmax(score[i] / temperature)
```

### Configuration

Edit `config/config.yaml`:

```yaml
# Multi-Agent Settings
agent:
  anomaly_threshold: 95  # Percentile for anomaly detection
  trust_alpha: 0.7       # EMA coefficient for trust
  temperature: 1.0       # Softmax temperature

# Federated Settings
federated:
  num_clients: 5
  malicious_ratio: 0.4
  aggregation_method: 'multi_agent'  # or 'fedavg', 'krum', 'trimmed_mean'

# Non-IID Settings
data:
  dirichlet_alpha: 0.1   # Lower = more heterogeneous

# Attack Settings
attack:
  type: 'label_flipping'  # or 'model_scaling', 'backdoor'
  flip_ratio: 0.5
```

### Datasets

**Supported Datasets:**
- **NSL-KDD**: Classic network intrusion detection (auto-download)
- **CICIDS2017**: Realistic attack scenarios (manual download)

**NSL-KDD Auto-Download:**
```python
from data.dataset_loader import NSLKDDProcessor
processor = NSLKDDProcessor()
X_train, y_train, X_test, y_test = processor.preprocess()
```

See [`DATASET_GUIDE.md`](DATASET_GUIDE.md) for CICIDS2017 setup.

### Design Principles

1. **Simplicity over Complexity**: Statistical methods instead of complex RL
2. **Interpretability**: All agent decisions are explainable
3. **Robustness**: Tested against multiple attack types
4. **Reproducibility**: Complete code and experiment scripts
5. **Engineering-Focused**: Practical solutions for real deployments

### Research Contributions

1. **Lightweight Multi-Agent Framework**: Simple agents outperform complex RL
2. **Adaptive Aggregation**: Combines anomaly detection and trust for robust federated learning
3. **Comprehensive Evaluation**: Tests against Krum, Trimmed Mean with multiple attacks
4. **Reproducible Implementation**: Complete open-source codebase

### Citation

```bibtex
@article{multi-agent-fl-ids,
  title={Multi-Agent Federated Learning for Network Intrusion Detection},
  author={Your Name},
  year={2025},
  note={Code: https://github.com/black-tape/cyber-rl-tr}
}
```

### License

MIT License

---

## <a name="chinese"></a>中文

### 概述

用于网络入侵检测的联邦学习系统，具有以下特点：
- **轻量级多 Agent 架构**：简单、可解释的异常检测、信誉管理和自适应聚合 Agent
- **鲁棒基线方法**：包含 Krum、Trimmed Mean、Median 等抗拜占庭攻击方法
- **攻击抵抗能力**：针对标签翻转、模型缩放、后门攻击进行测试
- **Non-IID 数据支持**：基于 Dirichlet 分布的极端数据异构性划分

### 主要特点

- 简单的统计 Agent（无复杂 RL）- **可解释且轻量**
- 基于异常分数和信誉的动态聚合权重
- 完整的基线方法用于公平对比
- 完整的攻击模拟框架
- 自动成功标准验证

### 快速开始

#### 1. 安装依赖

```bash
conda create -n mypaper python=3.10
conda activate mypaper
pip install -r requirements.txt
```

#### 2. 运行最小验证实验（MVE）

```bash
python train/train_mve.py
```

对比以下方法：
- FedAvg（基线）
- Krum（拜占庭鲁棒）
- Trimmed Mean（拜占庭鲁棒）
- **我们的方法（Multi-Agent）** - 异常检测 + 信誉 + 自适应聚合

**实验设置：**
- 5 个客户端，40% 恶意（2 个拜占庭客户端）
- Dirichlet α = 0.1（极度 Non-IID）
- 标签翻转攻击（50% 标签被翻转）
- 50 轮联邦训练

**成功标准：**
- F1 分数相比 FedAvg 提升 ≥ 5%
- 收敛速度加快 ≥ 20%

#### 3. 查看结果

结果保存为：
- `mve_results.png` - 对比图表
- 控制台输出 - 改进摘要表

### 核心算法

#### 多 Agent 自适应聚合

**步骤 1：异常检测**
```python
异常分数 = f(L2范数, 余弦相似度, peer距离)
```

**步骤 2：信誉管理**
```python
trust[i] = α * trust[i] + (1-α) * 当前性能
```

**步骤 3：自适应权重**
```python
score[i] = trust[i] / (1 + anomaly[i])
weight[i] = softmax(score[i] / 温度)
```

### 配置说明

编辑 `config/config.yaml`：

```yaml
# Multi-Agent 设置
agent:
  anomaly_threshold: 95  # 异常检测百分位数
  trust_alpha: 0.7       # 信誉 EMA 系数
  temperature: 1.0       # Softmax 温度

# 联邦学习设置
federated:
  num_clients: 5
  malicious_ratio: 0.4
  aggregation_method: 'multi_agent'  # 或 'fedavg', 'krum', 'trimmed_mean'

# Non-IID 设置
data:
  dirichlet_alpha: 0.1   # 越小越异构

# 攻击设置
attack:
  type: 'label_flipping'  # 或 'model_scaling', 'backdoor'
  flip_ratio: 0.5
```

### 数据集

**支持的数据集：**
- **NSL-KDD**：经典网络入侵检测数据集（自动下载）
- **CICIDS2017**：真实攻击场景数据集（需手动下载）

**NSL-KDD 自动下载：**
```python
from data.dataset_loader import NSLKDDProcessor
processor = NSLKDDProcessor()
X_train, y_train, X_test, y_test = processor.preprocess()
```

CICIDS2017 设置见 [`DATASET_GUIDE.md`](DATASET_GUIDE.md)。

### 设计原则

1. **简单优于复杂**：统计方法而非复杂 RL
2. **可解释性**：所有 Agent 决策都可解释
3. **鲁棒性**：针对多种攻击类型进行测试
4. **可重现性**：完整代码和实验脚本
5. **工程导向**：实际部署的实用解决方案

### 研究贡献

1. **轻量级多 Agent 框架**：简单 Agent 优于复杂 RL
2. **自适应聚合**：结合异常检测和信誉实现鲁棒联邦学习
3. **全面评估**：针对 Krum、Trimmed Mean 和多种攻击进行测试
4. **可复现实现**：完整开源代码库

### 引用

```bibtex
@article{multi-agent-fl-ids,
  title={Multi-Agent Federated Learning for Network Intrusion Detection},
  author={Your Name},
  year={2025},
  note={Code: https://github.com/black-tape/cyber-rl-tr}
}
```

### 许可证

MIT License
