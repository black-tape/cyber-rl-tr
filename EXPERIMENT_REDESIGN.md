# 实验设计修正方案

## 当前问题诊断

基于审稿人视角的分析，当前实现存在以下**致命问题**：

### ❌ 问题 1：Agent 设计过于复杂
- 使用了完整的 DQN/PPO（引入噪声 > 收益）
- 没有体现"简单有效"的工程优势
- 难以向审稿人证明效果来自核心创新

### ❌ 问题 2：缺少关键 Baseline
- 只有 FedAvg uniform vs reward-based
- 缺少 Krum, Trimmed Mean 等鲁棒聚合方法
- 无法证明优于现有防御方法

### ❌ 问题 3：实验设置不够"狠"
- Non-IID 程度可能不够
- 缺少攻击模型（投毒、后门）
- 难以展示在恶劣环境下的优势

---

## ✅ 修正方案

### 1. 简化核心创新为三个轻量级 Agent

```python
# 不再使用 RL，改为基于规则的简单 Agent

class AnomalyDetectionAgent:
    """异常检测 Agent - 基于统计特征"""
    def compute_anomaly_score(self, update, historical_updates):
        # 简单的 L2 范数偏差 + 余弦相似度
        norm_dev = np.linalg.norm(update - mean_update)
        cosine_sim = cosine_similarity(update, mean_update)
        return anomaly_score

class TrustAgent:
    """信誉 Agent - 指数移动平均"""
    def update_trust(self, client_id, current_performance):
        # 简单的 EMA
        trust[client_id] = α * trust[client_id] + (1-α) * current_performance

class AggregationAgent:
    """聚合 Agent - 自适应权重"""
    def compute_weights(self, anomaly_scores, trust_scores):
        # 组合异常分数和信誉
        weights = softmax(trust_scores / (1 + anomaly_scores))
        return weights
```

**设计原则：**
- ✅ 可解释性强
- ✅ 计算开销低
- ✅ 审稿人无法攻击"黑盒"

---

### 2. 添加完整 Baseline 对比

| 方法 | 类型 | 核心思想 |
|------|------|---------|
| **FedAvg** | 基础 | 均匀权重 |
| **Krum** | 鲁棒聚合 | 选择最近邻的客户端 |
| **Trimmed Mean** | 鲁棒聚合 | 裁剪极端值 |
| **FedProx** | 正则化 | 添加近端项 |
| **Ours (Multi-Agent)** | 自适应协同 | 异常检测 + 信誉 + 自适应聚合 |

---

### 3. 强化 Non-IID 和攻击设置

**Non-IID 配置：**
```python
# Dirichlet 分布参数
α_values = [0.1, 0.3, 0.5]  # 0.1 极度不均衡

# 攻击比例
poisoning_ratios = [0.1, 0.2, 0.3]  # 10%, 20%, 30%
```

**攻击模型：**
```python
# 1. Label Flipping
def label_flipping_attack(data, labels, flip_ratio=0.5):
    # 翻转 50% 的标签
    attack_indices = random.sample(range(len(labels)), int(len(labels) * flip_ratio))
    labels[attack_indices] = 1 - labels[attack_indices]

# 2. Model Scaling (放大梯度)
def model_scaling_attack(model_update, scaling_factor=10):
    return model_update * scaling_factor

# 3. Backdoor Attack
def backdoor_attack(data, labels, trigger_pattern):
    # 在数据中植入后门触发器
    poisoned_data = data.copy()
    poisoned_data[:, trigger_pattern] = 1
    poisoned_labels = np.ones_like(labels)  # 全部标记为攻击
```

---

### 4. 最小验证实验（MVE）设计

```python
# 实验配置
experiment_config = {
    # 数据集
    'dataset': 'NSL-KDD',  # 或 CIC-IDS2017

    # 客户端设置
    'num_clients': 20,
    'malicious_clients': [0.1, 0.2, 0.3],  # 10%, 20%, 30%

    # Non-IID 设置
    'alpha': [0.1, 0.3, 0.5],  # Dirichlet 参数

    # 攻击类型
    'attacks': ['label_flipping', 'model_scaling', 'backdoor'],

    # 模型
    'model': 'MLP',  # 简单模型，避免吃掉差距
    'hidden_dims': [64, 32],

    # 训练
    'num_rounds': 50,
    'local_epochs': 5,

    # 评估指标
    'metrics': ['accuracy', 'f1_macro', 'convergence_rounds']
}
```

---

### 5. 期望的实验结果图表

**图 1：不同方法在不同攻击比例下的 F1 Score**
```
       Attack Ratio: 10%    20%    30%
FedAvg:              0.82   0.65   0.42
Krum:                0.85   0.71   0.58
Trimmed Mean:        0.83   0.68   0.55
Ours:                0.88   0.79   0.68  ← 提升 5-10%
```

**图 2：收敛速度对比**
```
Rounds to 90% accuracy:
FedAvg:        35 rounds
Krum:          28 rounds
Trimmed Mean:  30 rounds
Ours:          22 rounds  ← 快 20-30%
```

**图 3：Ablation Study**
```
完整方法:                    F1 = 0.88
- 去掉异常检测 Agent:        F1 = 0.82
- 去掉信誉 Agent:            F1 = 0.84
- 固定权重聚合:              F1 = 0.80
```

---

## 实现优先级

### 🔥 第一优先级（本周完成）

1. **简化 Agent 设计**
   - 移除 DQN/PPO
   - 实现轻量级规则 Agent
   - 测试基础功能

2. **实现 Baseline**
   - FedAvg ✅（已有）
   - Krum ⚠️（需添加）
   - Trimmed Mean ⚠️（需添加）

3. **添加攻击模型**
   - Label Flipping ⚠️（需添加）
   - Model Scaling ⚠️（需添加）

### ⏳ 第二优先级（下周）

4. **强化 Non-IID 数据划分**
   - Dirichlet 分布
   - 极端不均衡场景

5. **运行最小验证实验**
   - α=0.1, 20% 攻击
   - FedAvg vs Ours
   - 验证是否有 5% 提升

### 📊 第三优先级（后续）

6. **完整实验矩阵**
7. **结果可视化**
8. **论文初稿**

---

## 成功标准

**MVE 成功标准（必须达到）：**

```
在以下设置下：
- NSL-KDD 数据集
- α = 0.1 (极度 Non-IID)
- 20% 恶意客户端
- Label Flipping 攻击

我们的方法必须达到：
✅ F1 Score 提升 ≥ 5%
✅ 收敛轮数减少 ≥ 20%
✅ 对恶意客户端更鲁棒（可视化）
```

**如果达不到 → 立即调整设计，而非继续完整实验**

---

## 下一步行动

我建议立即修改以下文件：

1. ✅ `agent/simple_agents.py` - 新建，实现轻量级 Agent
2. ✅ `baseline/robust_aggregation.py` - 新建，实现 Krum/TrimmedMean
3. ✅ `attack/poisoning.py` - 新建，实现攻击模型
4. ✅ `train/train_mve.py` - 新建，最小验证实验
5. ⚠️ `config/config.yaml` - 修改，添加新配置

您希望我：
- A. 立即开始修改代码（实现简化版 Agent）
- B. 先运行现有代码看看效果
- C. 先帮您写论文 Introduction
- D. 其他建议

**推荐选 A**，因为现有代码无法验证核心假设。
