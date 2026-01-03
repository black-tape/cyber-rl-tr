"""
使用真实数据集进行联邦训练
支持 NSL-KDD 和 CICIDS2017 数据集
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from env.security_env import SecurityEnv
from agent.rl_agent import DQNAgent
from reward.reward_fn import RewardFunction
from federated.client import FederatedClient
from federated.server import FederatedServer


def load_nsl_kdd_data(data_dir: str = 'data/processed/nsl_kdd', num_clients: int = 3):
    """
    加载 NSL-KDD 数据集

    Args:
        data_dir: 数据目录
        num_clients: 客户端数量

    Returns:
        (datasets, labels_list, X_test, y_test)
    """
    data_path = Path(data_dir)

    print(f"从 {data_path} 加载 NSL-KDD 数据...")

    # 加载客户端数据
    datasets = []
    labels_list = []

    for i in range(num_clients):
        X = np.load(data_path / f'client_{i}_X.npy')
        y = np.load(data_path / f'client_{i}_y.npy')
        datasets.append(X)
        labels_list.append(y)
        print(f"客户端 {i}: {len(X)} 条样本, 攻击比例: {np.mean(y):.2%}")

    # 加载测试数据
    X_test = np.load(data_path / 'X_test.npy')
    y_test = np.load(data_path / 'y_test.npy')
    print(f"测试集: {len(X_test)} 条样本, 攻击比例: {np.mean(y_test):.2%}")

    return datasets, labels_list, X_test, y_test


def train_with_real_data(
    dataset_name: str = 'nsl_kdd',
    num_clients: int = 3,
    num_rounds: int = 50,
    episodes_per_round: int = 10
):
    """
    使用真实数据集进行联邦训练

    Args:
        dataset_name: 数据集名称 ('nsl_kdd' 或 'cicids2017')
        num_clients: 客户端数量
        num_rounds: 联邦轮次
        episodes_per_round: 每轮的 episodes 数量
    """
    print("=" * 70)
    print(f"使用真实数据集进行联邦训练: {dataset_name.upper()}")
    print("=" * 70)

    # 加载数据
    if dataset_name == 'nsl_kdd':
        datasets, labels_list, X_test, y_test = load_nsl_kdd_data(
            num_clients=num_clients
        )
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    # 获取特征维度
    feature_dim = datasets[0].shape[1]
    history_length = 5
    state_dim = feature_dim + history_length * 3  # 特征 + 历史动作编码

    print(f"\n特征维度: {feature_dim}")
    print(f"状态维度: {state_dim}")

    # 创建奖励函数
    reward_fn = RewardFunction(
        attack_blocked_reward=1.0,
        attack_missed_penalty=-1.0,
        false_positive_penalty=-0.3,
        normal_allowed_reward=0.2
    )

    # 为每个客户端创建环境
    print("\n创建环境...")
    envs = []
    for i in range(num_clients):
        env = SecurityEnv(
            data=datasets[i],
            labels=labels_list[i],
            state_dim=state_dim,
            history_length=history_length
        )
        envs.append(env)

    # 创建联邦客户端
    print("\n创建联邦客户端...")
    clients = []
    for i in range(num_clients):
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=3,
            hidden_dims=(128, 64),
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995
        )

        client = FederatedClient(
            client_id=i,
            agent=agent,
            local_epochs=5
        )
        clients.append(client)

    # 创建联邦服务器
    print("\n创建联邦服务器...")
    server = FederatedServer(
        aggregation_method='reward_based',
        temperature=1.0,
        min_weight=0.01
    )

    # 创建模型保存目录
    model_dir = Path('models/real_data')
    model_dir.mkdir(parents=True, exist_ok=True)

    # 联邦训练
    print("\n开始联邦训练...")
    print("=" * 70)

    round_rewards = []

    for round_num in range(num_rounds):
        print(f"\n--- 联邦轮次 {round_num + 1}/{num_rounds} ---")

        # 客户端本地训练
        client_weights = {}
        client_rewards = {}
        client_statistics = {}

        for i, client in enumerate(clients):
            print(f"[客户端 {i}] 本地训练...")

            train_stats = client.local_train(
                env=envs[i],
                num_episodes=episodes_per_round,
                batch_size=64,
                update_freq=1
            )

            client_weights[i] = client.get_model_weights()
            client_rewards[i] = train_stats['avg_reward']
            client_statistics[i] = train_stats

            # 打印统计
            if train_stats['episode_statistics']:
                last_stats = train_stats['episode_statistics'][-1]
                print(f"  Reward: {train_stats['avg_reward']:.2f} | "
                      f"DR: {last_stats['Detection_Rate']:.3f} | "
                      f"FPR: {last_stats['False_Positive_Rate']:.3f} | "
                      f"F1: {last_stats['F1_Score']:.3f}")

        # 服务器聚合
        print("\n[服务器] 强化型聚合...")
        global_weights = server.aggregate(client_weights, client_rewards)

        # 显示聚合权重
        agg_summary = server.get_aggregation_weights_summary()
        print("聚合权重:")
        for cid, weight in agg_summary['latest_weights'].items():
            print(f"  客户端 {cid}: {weight:.4f} (Reward: {client_rewards[cid]:.2f})")

        # 广播全局模型
        for client in clients:
            client.set_model_weights(global_weights)

        # 记录统计
        server.record_round_statistics(round_num, client_statistics)

        avg_reward = np.mean(list(client_rewards.values()))
        round_rewards.append(avg_reward)

        print(f"\n[轮次 {round_num + 1}] 全局平均 Reward: {avg_reward:.2f}")

        # 定期保存
        if (round_num + 1) % 10 == 0:
            checkpoint_path = model_dir / f'federated_{dataset_name}_round_{round_num + 1}.pt'
            server.save_global_model(str(checkpoint_path))
            print(f"检查点已保存: {checkpoint_path}")

    # 保存最终模型
    final_model_path = model_dir / f'federated_{dataset_name}_final.pt'
    server.save_global_model(str(final_model_path))
    print(f"\n最终模型已保存: {final_model_path}")

    # 在测试集上评估
    print("\n" + "=" * 70)
    print("测试集评估")
    print("=" * 70)

    # 创建测试环境
    test_env = SecurityEnv(
        data=X_test,
        labels=y_test,
        state_dim=state_dim,
        history_length=history_length
    )

    # 使用第一个客户端的 Agent 进行测试
    print("\n使用全局模型进行测试...")
    test_agent = clients[0].agent

    eval_rewards = []
    eval_stats_list = []

    for episode in range(20):
        state = test_env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = test_agent.select_action(state, training=False)
            next_state, reward, done, info = test_env.step(action)
            episode_reward += reward
            state = next_state

        eval_rewards.append(episode_reward)
        eval_stats_list.append(test_env.get_episode_statistics())

    # 计算平均指标
    avg_reward = np.mean(eval_rewards)
    avg_dr = np.mean([s['Detection_Rate'] for s in eval_stats_list])
    avg_fpr = np.mean([s['False_Positive_Rate'] for s in eval_stats_list])
    avg_f1 = np.mean([s['F1_Score'] for s in eval_stats_list])

    print(f"\n测试集结果:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Detection Rate: {avg_dr:.3f}")
    print(f"  False Positive Rate: {avg_fpr:.3f}")
    print(f"  F1 Score: {avg_f1:.3f}")

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)

    return server, clients, round_rewards


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='使用真实数据集进行联邦训练')
    parser.add_argument('--dataset', type=str, default='nsl_kdd',
                        choices=['nsl_kdd', 'cicids2017'],
                        help='数据集名称')
    parser.add_argument('--num_clients', type=int, default=3,
                        help='客户端数量')
    parser.add_argument('--num_rounds', type=int, default=50,
                        help='联邦轮次')
    parser.add_argument('--episodes_per_round', type=int, default=10,
                        help='每轮的 episodes 数量')

    args = parser.parse_args()

    train_with_real_data(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        episodes_per_round=args.episodes_per_round
    )
