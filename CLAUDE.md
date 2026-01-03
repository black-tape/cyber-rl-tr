# Claude 项目重构指南

## 项目状态：需要重大重构

基于 A-level 审稿人视角的分析，当前项目需要**彻底重构**以确保：
1. 创新点可被验证
2. 实验结果有说服力
3. 论文可被接收

---

## 核心问题诊断

### ❌ 致命问题 1：过度设计
- 使用完整的 DQN/PPO RL Agent
- 引入不必要的复杂性
- 审稿人会质疑："效果来自哪里？RL 还是聚合策略？"

### ❌ 致命问题 2：缺少关键对比
- 只有 FedAvg baseline
- 缺少 Krum, Trimmed Mean 等鲁棒聚合方法
- 无法证明优于现有防御

### ❌ 致命问题 3：实验设置不够"恶劣"
- Non-IID 程度可能不够
- 没有攻击模型（投毒、后门）
- 难以展示在极端场景下的优势

---

## ✅ 重构方案

### 核心创新简化为：三个轻量级 Agent 的协同

```
不再使用强化学习，改为：

1. Anomaly Detection Agent
   - 基于统计特征（L2范数、余弦相似度、peer距离）
   - 输出异常分数 [0, 1]

2. Trust/Reputation Agent
   - 指数移动平均（EMA）
   - 维护历史信誉 [0, 1]

3. Aggregation Agent
   - 自适应权重 = f(trust, anomaly)
   - Softmax 归一化
```

**设计原则：**
- ✅ 简单、可解释
- ✅ 计算开销低
- ✅ 无"黑盒"，审稿人无法攻击

---

## 文件重构计划

### 需要新建的文件

| 文件 | 状态 | 优先级 | 说明 |
|------|------|--------|------|
| `agent/simple_agents.py` | ✅ 已完成 | P0 | 轻量级 Agent 系统 (3个Agent) |
| `baseline/robust_aggregation.py` | ✅ 已完成 | P0 | Krum, Trimmed Mean, Median, FedProx |
| `attack/poisoning.py` | ✅ 已完成 | P0 | Label Flipping, Model Scaling, Backdoor, Gaussian Noise, Sign Flipping |
| `data/non_iid.py` | ✅ 已完成 | P1 | Dirichlet, Label Skew, Quantity Skew, Combined |
| `train/train_mve.py` | ✅ 已完成 | P0 | 最小验证实验脚本 (MVE) |
| `eval/metrics.py` | ⚠️ 待创建 | P2 | 评估指标（MVE中已包含基础指标） |
| `CLAUDE.md` | ✅ 当前文件 | P0 | 项目指南 |

### 需要修改的文件

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `federated/server.py` | 集成 `simple_agents.py` | P0 |
| `config/config.yaml` | 添加新配置项 | P1 |
| `main.py` | 支持新实验模式 | P1 |

### 需要废弃的文件（暂时保留）

| 文件 | 原因 |
|------|------|
| `agent/rl_agent.py` | 过于复杂，不适合当前场景 |
| `agent/policy_net.py` | 不再使用 RL |
| `train/train_federated.py` | 需要重写以支持攻击场景 |

---

## 实现优先级（按周划分）

### 🔥 本周（Week 1）：核心功能

**✅ Day 1-2: Baseline 方法（已完成）**
- ✅ `baseline/robust_aggregation.py` - 实现了 5 种聚合方法：
  - KrumAggregator: 选择最近邻客户端
  - TrimmedMeanAggregator: 裁剪极端值
  - MedianAggregator: 中位数聚合
  - FedProxAggregator: FedAvg 加权平均
  - get_aggregator() 工厂函数

**✅ Day 3-4: 攻击模型（已完成）**
- ✅ `attack/poisoning.py` - 实现了 5 种攻击：
  - LabelFlippingAttack: 标签翻转 (0->1, 1->0)
  - ModelScalingAttack: 模型缩放攻击
  - BackdoorAttack: 后门攻击 (带触发器)
  - GaussianNoiseAttack: 高斯噪声
  - SignFlippingAttack: 符号翻转
  - AttackCoordinator: 协调多种攻击

**✅ Day 4-5: Non-IID 数据划分（已完成）**
- ✅ `data/non_iid.py` - 实现了 4 种划分策略：
  - DirichletPartitioner: Dirichlet 分布 (alpha 控制异构性)
  - LabelSkewPartitioner: 标签倾斜 (每客户端只有部分类别)
  - QuantitySkewPartitioner: 数量倾斜 (幂律分布)
  - CombinedPartitioner: 组合策略
  - analyze_partition() 分析异构性

**✅ Day 6-7: 最小验证实验（MVE）（已完成）**
- ✅ `train/train_mve.py` - 完整的 MVE 脚本：
  - 5 个客户端, 40% 恶意
  - Dirichlet alpha = 0.1 (极度 Non-IID)
  - Label Flipping 攻击 (50% 标签)
  - 对比 4 种方法: FedAvg, Krum, Trimmed Mean, Ours
  - 自动评估成功标准
  - 生成可视化结果

### ⏳ 下周（Week 2）：完整实验

**Day 8-10: 扩展实验矩阵**
```python
# 不同 Non-IID 程度
alphas = [0.1, 0.3, 0.5]

# 不同攻击比例
attack_ratios = [0.1, 0.2, 0.3]

# 不同攻击类型
attacks = ['label_flipping', 'model_scaling', 'backdoor']
```

**Day 11-12: 消融实验**
```python
# 移除不同组件，观察影响
ablations = [
    'full',  # 完整方法
    'no_anomaly',  # 去掉异常检测
    'no_trust',  # 去掉信誉管理
    'fixed_weight'  # 固定权重
]
```

**Day 13-14: 结果可视化**
```python
# 生成关键图表：
# 1. F1 Score vs Attack Ratio
# 2. Convergence Speed
# 3. Ablation Study
# 4. Weight Distribution
```

### 📊 Week 3：论文撰写

**Day 15-17: Introduction + Related Work**
- 使用前面提供的 Prompt 让 Claude 生成初稿

**Day 18-19: Method**
- 系统架构图
- 算法伪代码
- 设计原理说明

**Day 20-21: Experiments**
- 实验设置
- 结果表格
- 结果分析

---

## 最小验证实验（MVE）配置

这是**最关键的实验**，必须先跑出效果再做后续工作。

```yaml
# MVE Configuration
dataset: NSL-KDD
model: MLP
hidden_dims: [64, 32]

# Clients
num_clients: 20
malicious_ratio: 0.2  # 20%

# Non-IID
alpha: 0.1  # 极度不均衡

# Attack
attack_type: label_flipping
flip_ratio: 0.5

# Training
num_rounds: 50
local_epochs: 5
batch_size: 64

# Methods
baselines:
  - FedAvg
  - Krum
  - TrimmedMean

ours:
  - MultiAgent

# Metrics
metrics:
  - accuracy
  - f1_macro
  - convergence_rounds
```

### MVE 成功标准

**必须达到以下结果才继续完整实验：**

```
场景：α=0.1, 20% 恶意客户端, Label Flipping

期望结果：
✅ Ours F1 > FedAvg F1 + 5%
✅ Ours F1 > Krum F1 + 3%
✅ 收敛轮数减少 20%
✅ 对恶意客户端权重 < 0.1

如果达不到 → 立即调整设计
```

---

## 代码示例：如何使用新的 Agent 系统

```python
from agent.simple_agents import MultiAgentFederatedSystem
from baseline.robust_aggregation import KrumAggregator
from attack.poisoning import label_flipping_attack

# 1. 初始化多 Agent 系统
ma_system = MultiAgentFederatedSystem(
    num_clients=20,
    anomaly_threshold=95,
    trust_decay=0.9,
    temperature=1.0
)

# 2. 训练循环
for round in range(num_rounds):
    # 本地训练
    client_updates = {}
    client_performances = {}

    for client_id in range(num_clients):
        # 训练本地模型
        update = train_local(client_id, data[client_id])
        performance = evaluate(client_id, test_data)

        # 如果是恶意客户端，注入攻击
        if client_id in malicious_clients:
            update = label_flipping_attack(update)

        client_updates[client_id] = update
        client_performances[client_id] = performance

    # 3. 多 Agent 协同聚合
    global_model, weights, stats = ma_system.aggregate_updates(
        client_updates,
        global_model,
        client_performances
    )

    # 4. 查看统计
    print(f"Round {round}:")
    print(f"  Anomaly scores: {stats['anomaly_scores']}")
    print(f"  Trust scores: {stats['trust_scores']}")
    print(f"  Aggregation weights: {stats['aggregation_weights']}")
```

---

## 论文撰写 Prompts（提供给 Claude）

### Prompt 1: Introduction

```
Write the Introduction section (approx. 1.5-2 pages) for this paper.

Structure:
1. Background: Federated IDS, privacy concerns
2. Challenges: Non-IID data, malicious clients, limitations of existing methods
3. Key Observation: Client behavior has both short-term anomalies and long-term patterns
4. Our Approach: Multi-agent collaboration (anomaly detection + trust + adaptive aggregation)
5. Contributions: System design, robustness, reproducibility

Style:
- Conservative, engineering-focused
- Avoid "SOTA" or "optimal" claims
- Emphasize realism and robustness
```

### Prompt 2: Method

```
Write the System Model and Method sections.

Include:
- Federated workflow
- Non-IID data (Dirichlet)
- Threat model (partial malicious, no server compromise)
- Multi-Agent Framework:
  * Anomaly Detection Agent (L2 norm, cosine similarity, peer distance)
  * Trust Agent (EMA-based reputation)
  * Aggregation Agent (adaptive weights)
- Design rationale: why soft weighting, not hard filtering

Avoid RL, game theory. Keep it simple and interpretable.
```

### Prompt 3: Experiments

```
Write the Experimental Evaluation section (design only, not results yet).

Include:
- Datasets: NSL-KDD, CIC-IDS2017
- Baselines: FedAvg, Krum, Trimmed Mean
- Settings: 20 clients, α ∈ {0.1, 0.3, 0.5}, attack ratio ∈ {10%, 20%, 30%}
- Attacks: Label flipping, model scaling, backdoor
- Metrics: Accuracy, F1, convergence speed
- Ablation: w/o anomaly agent, w/o trust agent, fixed weights

Emphasize fairness and reproducibility.
```

---

## 下一步行动

我会按以下顺序继续：

1. ✅ **创建 Baseline 方法**（`baseline/robust_aggregation.py`）- 已完成
2. ✅ **创建攻击模型**（`attack/poisoning.py`）- 已完成
3. ✅ **创建 Non-IID 数据划分**（`data/non_iid.py`）- 已完成
4. ✅ **创建 MVE 脚本**（`train/train_mve.py`）- 已完成
5. ⏳ **清理项目文件并上传 GitHub**
6. ⏸️ **运行 MVE 验证效果**
7. ⏸️ **根据结果决定是否继续完整实验**

---

## 📦 已完成的核心模块总结

### 1. Baseline 方法 (`baseline/robust_aggregation.py`)
- **KrumAggregator**: 选择最中心的客户端更新，抵抗 Byzantine 攻击
- **TrimmedMeanAggregator**: 裁剪极端值后平均，可配置裁剪比例
- **MedianAggregator**: 坐标级中位数，最鲁棒但收敛可能较慢
- **FedProxAggregator**: 标准加权平均（作为 FedAvg baseline）
- **get_aggregator()**: 工厂函数，统一接口

### 2. 攻击模型 (`attack/poisoning.py`)
- **LabelFlippingAttack**: 数据投毒 - 翻转训练标签
- **ModelScalingAttack**: 模型投毒 - 放大梯度更新
- **BackdoorAttack**: 后门攻击 - 注入触发器模式
- **GaussianNoiseAttack**: 添加高斯噪声干扰训练
- **SignFlippingAttack**: 符号翻转（最大化损失而非最小化）
- **AttackCoordinator**: 统一管理恶意客户端和攻击应用

### 3. Non-IID 数据划分 (`data/non_iid.py`)
- **DirichletPartitioner**: Dirichlet 分布控制标签异构性 (alpha 参数)
- **LabelSkewPartitioner**: 病理性 Non-IID（每客户端仅部分类别）
- **QuantitySkewPartitioner**: 数量不平衡（幂律分布）
- **CombinedPartitioner**: 组合标签+数量倾斜
- **analyze_partition()**: 分析数据异构性统计

### 4. 最小验证实验 (`train/train_mve.py`)
- **完整实验流程**: 数据加载 → 划分 → 攻击 → 训练 → 评估
- **SimpleMLP**: 轻量级模型避免容量掩盖差异
- **FederatedTrainer**: 联邦训练管理器
- **run_mve_experiment()**: 单方法实验执行
- **自动评估**: 成功标准检查 (F1 提升 ≥5%, 收敛加速 ≥20%)
- **可视化**: 自动生成对比图表
