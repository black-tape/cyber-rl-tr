# 多 Agent 强化学习 + 强化型联邦学习安全防御系统

基于多 Agent 强化学习和强化型联邦学习的网络安全入侵检测防御系统。

## 项目特点

- **多 Agent 强化学习**: 每个客户端拥有独立的 RL Agent，适应 Non-IID 数据分布
- **强化型联邦学习**: 基于 Reward 的动态聚合策略，优于传统 FedAvg
- **轻量级模型**: 2-3 层 MLP，适合边缘设备和 IoT 场景
- **隐私保护**: 仅共享模型参数，保护本地数据隐私

## 项目结构

```
mypaper/
├── agent/                      # Agent 模块
│   ├── policy_net.py          # 策略网络 (DQN/PPO)
│   ├── rl_agent.py            # RL Agent 实现
│   └── multi_agent_controller.py  # 多 Agent 控制器
│
├── env/                       # 环境模块
│   └── security_env.py        # 安全检测环境
│
├── federated/                 # 联邦学习模块
│   ├── client.py             # 联邦客户端
│   └── server.py             # 联邦服务器 (强化型聚合)
│
├── reward/                    # 奖励函数
│   └── reward_fn.py          # 奖励函数设计
│
├── train/                     # 训练脚本
│   ├── train_local.py        # 本地训练
│   └── train_federated.py    # 联邦训练
│
├── config/                    # 配置文件
│   └── config.yaml           # 项目配置
│
├── main.py                    # 主入口
└── requirements.txt           # 依赖项
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行联邦训练

```bash
python main.py --mode federated
```

### 3. 运行本地训练（单 Agent）

```bash
python main.py --mode local
```

## 配置说明

编辑 `config/config.yaml` 来调整训练参数:

- `agent.type`: 选择 'dqn' 或 'ppo'
- `federated.num_clients`: 客户端数量
- `federated.aggregation_method`: 聚合方法
  - `'reward_based'`: 强化型聚合（推荐）
  - `'uniform'`: 传统 FedAvg
- `federated.temperature`: 温度参数（控制权重分布）

## 核心算法

### 强化型联邦聚合

传统 FedAvg 使用简单平均:
```
w_global = (1/N) * Σ w_i
```

强化型聚合基于 Reward:
```
α_i = softmax(R_i / T)
w_global = Σ α_i * w_i
```

其中:
- `R_i`: Agent i 的局部平均 Reward
- `T`: 温度参数
- `α_i`: 动态聚合权重

### 奖励函数设计

| 场景 | 动作 | 奖励 |
|------|------|------|
| 攻击流量 | BLOCK | +1.0 (成功阻断) |
| 攻击流量 | ALLOW | -1.0 (漏报) |
| 正常流量 | BLOCK | -0.3 (误报) |
| 正常流量 | ALLOW | +0.2 (正确放行) |

## 评估指标

- **Detection Rate (DR)**: 攻击检测率
- **False Positive Rate (FPR)**: 误报率
- **False Negative Rate (FNR)**: 漏报率
- **F1 Score**: 综合指标
- **Average Reward**: 平均奖励

## 实验结果

预期结果:
- Detection Rate > 95%
- False Positive Rate < 5%
- 强化型聚合比传统 FedAvg 收敛快 20-30%

## 技术栈

- **PyTorch**: 深度学习框架
- **NumPy**: 数值计算
- **PyYAML**: 配置管理
- **Python 3.8+**

## 数据集

当前使用模拟数据进行测试。支持以下真实数据集:
- CICIDS2017
- NSL-KDD

## 论文贡献点

1. 首次将多 Agent RL 与强化型联邦学习结合用于安全防御
2. 提出基于 Reward 的动态聚合策略
3. 在 Non-IID 数据分布下验证有效性
4. 提供完整可复现的实现

## 后续工作

- [ ] 集成真实数据集 (CICIDS2017)
- [ ] 实现基线方法对比
- [ ] 添加可视化工具
- [ ] 性能优化

## 引用

如果您使用此代码,请引用:

```
@article{multi-agent-fl-security,
  title={Multi-Agent Reinforcement Learning with Reward-based Federated Aggregation for Network Intrusion Detection},
  author={Your Name},
  year={2025}
}
```

## 许可证

MIT License
