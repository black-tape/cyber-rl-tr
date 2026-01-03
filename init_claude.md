# 多 Agent 强化学习 + 强化型联邦学习安全防御系统

## 项目核心目标

构建一个基于多 Agent 强化学习和强化型联邦学习的网络安全入侵检测防御系统，通过个性化小模型实现分布式、自适应的安全防御能力。

## 核心创新点

1. **多 Agent 强化学习架构**
   - 每个客户端/终端拥有独立的 RL Agent（轻量级小模型）
   - 每个 Agent 根据本地攻击分布独立学习策略
   - 适应 Non-IID 数据场景，提供个性化防御

2. **强化型联邦学习聚合**
   - 超越传统 FedAvg 简单平均
   - 根据 Agent 局部 Reward 动态调整聚合权重
   - 性能好的 Agent 获得更高权重，加速全局收敛

3. **个性化轻量级模型**
   - 小型 MLP 网络（2-3 层），适合边缘设备/IoT 场景
   - 保持本地隐私，仅共享模型参数
   - 可扩展到大规模分布式环境

## 项目架构设计

```
project_root/
│
├── data/                         # 数据目录
│   ├── raw/                      # 原始数据集
│   ├── processed/                # 预处理后数据
│
├── env/                          # 强化学习环境
│   └── security_env.py           # 多 Agent 安全环境（支持多智能体交互）
│
├── agent/                        # Agent 模块
│   ├── rl_agent.py               # 单个 RL Agent 实现
│   ├── policy_net.py             # 策略网络（小型 MLP）
│   └── multi_agent_controller.py # 多 Agent 管理器
│
├── federated/                    # 联邦学习模块
│   ├── client.py                 # 联邦客户端（每个 Agent 对应一个客户端）
│   └── server.py                 # 联邦服务器（强化型聚合策略）
│
├── reward/                       # 奖励函数
│   └── reward_fn.py              # 奖励函数设计
│
├── baseline/                     # 基线方法
│   ├── rule_based.py             # 静态规则基线
│   └── single_agent.py           # 单 Agent RL 基线
│
├── train/                        # 训练脚本
│   ├── train_local.py            # 单客户端本地训练
│   └── train_federated.py        # 多 Agent 联邦训练
│
├── eval/                         # 评估模块
│   ├── metrics.py                # 评估指标
│   └── evaluate.py               # 评估脚本
│
├── config/                       # 配置文件
│   └── config.yaml               # 项目配置
│
└── main.py                       # 项目入口
```

## 核心组件设计

### 1. 强化学习环境（security_env.py）

**State（状态）**：
- 当前网络流量特征向量（例如：协议类型、包大小、时间戳等）
- 最近 K 次历史行为统计（例如：连接频率、异常计数）

**Action（动作）**：
- 0 = ALLOW（允许通过）
- 1 = ALERT（警告但放行）
- 2 = BLOCK（阻断）

**Reward（奖励）**：
- 攻击被成功阻断：+1.0
- 攻击被误放行：-1.0
- 正常流量被误阻断（误报）：-0.3
- 正常流量正确放行：+0.2

**多 Agent 支持**：
- 支持多个 Agent 同时与环境交互
- 提供 reset() 和 step() 接口
- 支持分布式状态管理

### 2. RL Agent 设计（rl_agent.py + policy_net.py）

**网络结构**：
- 输入层：State 维度
- 隐藏层：2-3 层 MLP（64-128 维）
- 输出层：Action 数量（3）

**学习算法**：
- 主算法：DQN 或 PPO
- 支持经验回放（Replay Buffer）
- Epsilon-greedy 探索策略

**核心接口**：
```python
class RLAgent:
    def select_action(state) -> action
    def update(replay_buffer) -> loss
    def get_model_weights() -> dict
    def set_model_weights(weights) -> None
```

### 3. 多 Agent 控制器（multi_agent_controller.py）

**功能**：
- 管理所有 Agent 列表
- 收集各 Agent 的本地 Reward 统计
- 调度联邦聚合流程
- 协调 Agent 与环境的交互

**核心接口**：
```python
class MultiAgentController:
    def register_agent(agent_id, agent) -> None
    def collect_local_rewards() -> dict
    def trigger_federated_aggregation() -> None
    def broadcast_global_model(weights) -> None
```

### 4. 强化型联邦学习（server.py）

**传统 FedAvg 问题**：
- 简单平均所有客户端模型
- 忽略各客户端的性能差异
- 收敛速度慢，易受低质量客户端影响

**强化型聚合策略**：
```
聚合权重 w_i = softmax(avg_reward_i / temperature)

其中：
- avg_reward_i：Agent i 的局部平均 Reward
- temperature：温度参数，控制权重分布平滑度
- Reward 高的 Agent 获得更高聚合权重
```

**聚合流程**：
1. 各客户端本地训练并上传模型权重
2. Server 收集各客户端的 Reward 统计
3. 根据 Reward 计算动态聚合权重
4. 加权聚合生成全局模型
5. 广播全局模型回客户端

### 5. 奖励函数（reward_fn.py）

**设计原则**：
- 优先保证系统安全（重罚漏报）
- 控制误报率（轻罚误报）
- 引导 Agent 学习平衡策略

**具体设计**：
```python
def compute_reward(action, true_label, is_attack):
    if is_attack:
        if action == BLOCK:
            return +1.0  # 成功阻断攻击
        else:
            return -1.0  # 攻击漏报（严重）
    else:
        if action == BLOCK:
            return -0.3  # 误报（轻微）
        else:
            return +0.2  # 正确放行
```

## 数据集与分布设置

**数据集选择**：
- CICIDS2017（推荐）：包含多种真实网络攻击类型
- NSL-KDD（备选）：经典入侵检测数据集

**Non-IID 数据分布模拟**：
- 将数据集按攻击类型划分
- 每个客户端分配不同攻击类型子集
- 模拟真实场景下各节点面临的攻击分布差异

**数据预处理**：
- 特征归一化
- 类别编码
- 处理缺失值
- 划分训练/验证/测试集

## 训练流程设计

### 本地训练流程（train_local.py）

```
1. 初始化单个 Agent
2. 加载本地数据子集
3. For each episode:
   - Reset 环境
   - For each step:
     - Agent 选择 action
     - 环境返回 reward
     - 存入 replay buffer
   - 更新 Agent 策略
4. 保存模型
```

### 联邦训练流程（train_federated.py）

```
1. 初始化多个 Agent（对应多个客户端）
2. For each federated round:

   # 本地训练阶段
   For each client:
     - 加载全局模型
     - 本地训练 E 个 epochs
     - 收集本地 Reward 统计
     - 上传模型权重和 Reward

   # 强化型聚合阶段
   Server:
     - 根据各客户端 Reward 计算聚合权重
     - 加权聚合模型参数
     - 广播全局模型

3. 全局模型评估
```

## 基线方法（Baseline）

### 1. 静态规则基线（rule_based.py）
- 基于阈值/黑名单的静态规则
- 例如：IP 黑名单、端口扫描检测规则
- 无学习能力，仅作为基础对比

### 2. 单 Agent RL 基线（single_agent.py）
- 单客户端集中式 RL 训练
- 使用完整数据集，不考虑 Non-IID
- 对比分布式学习的优势

### 3. 传统 FedAvg 基线
- 多 Agent RL + 传统简单平均聚合
- 对比强化型聚合的优势

## 评估指标（metrics.py）

### 核心指标

1. **Detection Rate（检测率）**
   ```
   DR = TP / (TP + FN)
   TP：真阳性（攻击被正确阻断）
   FN：假阴性（攻击漏报）
   ```

2. **False Positive Rate（误报率）**
   ```
   FPR = FP / (FP + TN)
   FP：假阳性（正常流量被误阻断）
   TN：真阴性（正常流量正确放行）
   ```

3. **False Negative Rate（漏报率）**
   ```
   FNR = FN / (TP + FN)
   ```

4. **F1-Score**
   ```
   F1 = 2 * Precision * Recall / (Precision + Recall)
   ```

### 收敛性指标

- 收敛轮数（Rounds to Convergence）
- 训练时间
- 平均 Reward 曲线

### 联邦学习特定指标

- 通信成本（Communication Rounds）
- 客户端本地训练时间
- 模型参数传输量

## 实验设计

### 实验配置

| 参数 | 配置 |
|------|------|
| 客户端数量 | 3-5 |
| 数据集 | CICIDS2017 / NSL-KDD |
| Agent 模型 | 2-3 层 MLP（64-128 维） |
| RL 算法 | DQN / PPO |
| 联邦轮数 | 50-100 |
| 本地训练 Epochs | 5-10 |
| Batch Size | 64-128 |
| Learning Rate | 0.001 |
| 温度参数 | 1.0-2.0 |

### 对比实验

1. **强化型 FL vs 传统 FedAvg**
   - 对比收敛速度
   - 对比最终检测性能

2. **多 Agent RL vs 单 Agent RL**
   - 对比 Non-IID 场景下的鲁棒性
   - 对比可扩展性

3. **RL 方法 vs 静态规则**
   - 对比自适应能力
   - 对比新攻击检测能力

### Ablation Study（消融实验）

- 移除动态聚合权重的影响
- 不同客户端数量的影响
- 不同 Non-IID 程度的影响
- 不同 RL 算法（DQN vs PPO）的影响

## 预期结果与论文贡献

### 预期结果

1. **性能提升**
   - Detection Rate > 95%
   - FPR < 5%
   - 强化型 FL 比传统 FedAvg 收敛快 20-30%

2. **可扩展性**
   - 支持 3-5 个客户端同时训练
   - 模型轻量，适合边缘部署

3. **鲁棒性**
   - 在 Non-IID 数据分布下表现稳定
   - 对新攻击类型有一定泛化能力

### 论文贡献点

1. **技术创新**
   - 首次将多 Agent RL 与强化型联邦学习结合用于安全防御
   - 提出基于 Reward 的动态聚合策略

2. **实验验证**
   - 在公开数据集上完整实验
   - 完整的 Baseline 对比和消融实验

3. **实用价值**
   - 可复现代码和实验
   - 适合边缘设备/IoT 场景
   - 保护本地隐私

## 项目执行原则

### 开发原则

1. **模块化设计**
   - 每个模块独立可测试
   - 清晰的接口定义
   - 便于替换不同算法

2. **最小可行产品（MVP）**
   - 先实现基础功能
   - 确保可运行
   - 逐步优化

3. **可复现性**
   - 固定随机种子
   - 详细记录配置
   - 提供完整运行脚本

### 代码规范

- 使用 Python 3.8+
- 主要依赖：PyTorch, NumPy, Pandas, scikit-learn
- 清晰的注释和文档
- 类型提示（Type Hints）

### 版本控制

- 及时保存关键版本
- 记录实验结果
- 备份数据和模型

## 关键技术细节

### 状态表示（State Representation）

**特征选择**：
- 网络流量特征（协议、端口、包大小、时长等）
- 统计特征（连接频率、字节数等）
- 时序特征（最近 K 步行为）

**特征工程**：
- 归一化/标准化
- 特征选择（去除冗余特征）
- 维度控制（避免高维灾难）

### 探索-利用平衡（Exploration-Exploitation）

**Epsilon-Greedy 策略**：
```python
epsilon = max(epsilon_min, epsilon_start * decay_rate ^ episode)

if random() < epsilon:
    action = random_action()  # 探索
else:
    action = policy_network(state)  # 利用
```

### 稳定性技巧

1. **Target Network**（DQN）
   - 使用目标网络稳定训练
   - 定期更新目标网络

2. **Reward Clipping**
   - 限制 Reward 范围，避免梯度爆炸

3. **Gradient Clipping**
   - 梯度裁剪，防止训练不稳定

## 风险与应对

### 潜在问题

1. **收敛困难**
   - 应对：调整学习率、Reward 设计
   - 应对：增加训练轮数

2. **客户端性能差异大**
   - 应对：调整温度参数，避免过度依赖单一客户端
   - 应对：设置最小权重阈值

3. **数据集不平衡**
   - 应对：类别加权、过采样/欠采样
   - 应对：调整 Reward 函数

### 调试策略

1. **单步调试**
   - 先验证单 Agent 可训练
   - 再验证联邦聚合流程
   - 最后整体调试

2. **可视化**
   - 实时绘制 Reward 曲线
   - 监控各客户端性能
   - 检查聚合权重分布

3. **日志记录**
   - 详细记录训练过程
   - 保存中间模型
   - 记录异常情况

## 项目里程碑

### Phase 1：基础框架（1-2 周）
- 搭建项目结构
- 实现基础 RL Agent
- 实现简单环境

### Phase 2：核心功能（2-3 周）
- 实现多 Agent 控制器
- 实现联邦学习框架
- 实现强化型聚合

### Phase 3：实验验证（2-3 周）
- 数据集准备
- Baseline 实现
- 完整实验流程

### Phase 4：优化与分析（1-2 周）
- 性能优化
- 结果分析
- 论文撰写

## 参考资源

### 数据集
- CICIDS2017：https://www.unb.ca/cic/datasets/ids-2017.html
- NSL-KDD：https://www.unb.ca/cic/datasets/nsl.html

### 相关论文
- Federated Learning（FedAvg）
- Multi-Agent Reinforcement Learning
- Intrusion Detection using RL

### 技术文档
- PyTorch 官方文档
- OpenAI Gym 环境设计
- 联邦学习框架（如 Flower）

---

## 注意事项

1. **保持项目聚焦**
   - 不偏离核心目标
   - 避免过度复杂化
   - 以可运行、可复现为第一原则

2. **及时验证假设**
   - 每完成一个模块立即测试
   - 出现问题及时调整
   - 不要堆积技术债

3. **文档先行**
   - 设计先于实现
   - 接口定义清晰
   - 代码注释完整

4. **迭代开发**
   - 从简单到复杂
   - 小步快跑
   - 持续改进

---

**本文档是项目的北极星，任何开发和调整都应以此为基准，确保项目不偏移核心目标。**
