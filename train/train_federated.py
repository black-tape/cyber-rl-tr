"""
联邦训练脚本
多 Agent 强化型联邦学习训练
"""

import sys
import os
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from env.security_env import SecurityEnv, MockDataGenerator
from agent.rl_agent import DQNAgent, PPOAgent
from reward.reward_fn import RewardFunction
from federated.client import FederatedClient
from federated.server import FederatedServer


def load_config(config_path: str = '../config/config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_agent(config: dict, state_dim: int):
    """创建 Agent"""
    agent_type = config['agent']['type']

    if agent_type == 'dqn':
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=config['environment']['action_dim'],
            hidden_dims=tuple(config['agent']['hidden_dims']),
            learning_rate=config['agent']['learning_rate'],
            gamma=config['agent']['gamma'],
            epsilon_start=config['dqn']['epsilon_start'],
            epsilon_end=config['dqn']['epsilon_end'],
            epsilon_decay=config['dqn']['epsilon_decay'],
            target_update_freq=config['dqn']['target_update_freq'],
            use_dueling=config['dqn']['use_dueling']
        )
    elif agent_type == 'ppo':
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=config['environment']['action_dim'],
            hidden_dims=tuple(config['agent']['hidden_dims']),
            learning_rate=config['agent']['learning_rate'],
            gamma=config['agent']['gamma'],
            gae_lambda=config['ppo']['gae_lambda'],
            clip_epsilon=config['ppo']['clip_epsilon'],
            value_coef=config['ppo']['value_coef'],
            entropy_coef=config['ppo']['entropy_coef']
        )
    else:
        raise ValueError(f"不支持的 Agent 类型: {agent_type}")

    return agent


def train_federated(config_path: str = '../config/config.yaml'):
    """联邦训练主函数"""
    # 加载配置
    config = load_config(config_path)

    # 设置随机种子
    set_seed(config['training']['seed'])

    # 创建目录
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(config['model']['save_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("联邦训练 - 多 Agent 强化型联邦学习")
    print("=" * 60)

    # 生成 Non-IID 数据
    print("\n[1/5] 生成 Non-IID 训练数据...")
    num_clients = config['federated']['num_clients']

    datasets, labels_list = MockDataGenerator.generate_multi_agent_data(
        num_agents=num_clients,
        num_samples_per_agent=config['data']['num_samples_per_client'],
        feature_dim=config['data']['feature_dim'],
        attack_ratios=config['data']['attack_ratios']
    )

    for i, labels in enumerate(labels_list):
        print(f"客户端 {i}: {len(labels)} 个样本, 攻击比例: {np.mean(labels):.2%}")

    # 创建环境
    print("\n[2/5] 为每个客户端创建环境...")
    reward_fn = RewardFunction(
        attack_blocked_reward=config['reward']['attack_blocked_reward'],
        attack_missed_penalty=config['reward']['attack_missed_penalty'],
        false_positive_penalty=config['reward']['false_positive_penalty'],
        normal_allowed_reward=config['reward']['normal_allowed_reward'],
        alert_reward=config['reward']['alert_reward']
    )

    envs = []
    for i in range(num_clients):
        env = SecurityEnv(
            data=datasets[i],
            labels=labels_list[i],
            state_dim=config['environment']['state_dim'],
            history_length=config['environment']['history_length']
        )
        envs.append(env)
    print(f"环境创建完成: 状态维度 = {envs[0].state_dim}")

    # 创建联邦客户端
    print("\n[3/5] 创建联邦客户端和 Agents...")
    clients = []
    for i in range(num_clients):
        agent = create_agent(config, envs[i].state_dim)
        client = FederatedClient(
            client_id=i,
            agent=agent,
            local_epochs=config['federated']['local_epochs']
        )
        clients.append(client)
    print(f"创建了 {num_clients} 个客户端")

    # 创建联邦服务器
    print("\n[4/5] 创建联邦服务器...")
    server = FederatedServer(
        aggregation_method=config['federated']['aggregation_method'],
        temperature=config['federated']['temperature'],
        min_weight=config['federated']['min_weight']
    )
    print(f"聚合方法: {config['federated']['aggregation_method']}")
    print(f"温度参数: {config['federated']['temperature']}")

    # 联邦训练
    print("\n[5/5] 开始联邦训练...")
    num_rounds = config['federated']['num_rounds']
    episodes_per_epoch = config['federated']['episodes_per_epoch']

    round_rewards = []

    for round_num in range(num_rounds):
        print(f"\n{'='*60}")
        print(f"联邦轮次 {round_num + 1}/{num_rounds}")
        print(f"{'='*60}")

        # 客户端本地训练
        client_weights = {}
        client_rewards = {}
        client_statistics = {}

        for i, client in enumerate(clients):
            print(f"\n[客户端 {i}] 本地训练...")

            # 本地训练
            train_stats = client.local_train(
                env=envs[i],
                num_episodes=episodes_per_epoch,
                batch_size=config['training']['batch_size'],
                update_freq=config['training']['update_freq']
            )

            # 收集模型权重和 Reward
            client_weights[i] = client.get_model_weights()
            client_rewards[i] = train_stats['avg_reward']
            client_statistics[i] = train_stats

            print(f"[客户端 {i}] 平均 Reward: {train_stats['avg_reward']:.2f}")

            # 打印最后一个 episode 的统计
            if train_stats['episode_statistics']:
                last_stats = train_stats['episode_statistics'][-1]
                print(f"[客户端 {i}] DR: {last_stats['Detection_Rate']:.3f}, "
                      f"FPR: {last_stats['False_Positive_Rate']:.3f}, "
                      f"F1: {last_stats['F1_Score']:.3f}")

        # 服务器聚合
        print(f"\n[服务器] 执行强化型聚合...")
        global_weights = server.aggregate(client_weights, client_rewards)

        # 计算并显示聚合权重
        agg_weights_summary = server.get_aggregation_weights_summary()
        print(f"[服务器] 聚合权重:")
        for cid, weight in agg_weights_summary['latest_weights'].items():
            print(f"  客户端 {cid}: {weight:.4f}")

        # 广播全局模型
        print(f"[服务器] 广播全局模型到所有客户端...")
        for client in clients:
            client.set_model_weights(global_weights)

        # 记录统计
        server.record_round_statistics(round_num, client_statistics)

        # 轮次统计
        avg_reward = np.mean(list(client_rewards.values()))
        round_rewards.append(avg_reward)

        print(f"\n[轮次 {round_num + 1}] 全局平均 Reward: {avg_reward:.2f}")
        print(f"[轮次 {round_num + 1}] 最佳客户端: {server.get_best_client()}")

        # 保存检查点
        if (round_num + 1) % config['training']['save_freq'] == 0:
            checkpoint_path = model_dir / f"federated_round_{round_num + 1}.pt"
            server.save_global_model(str(checkpoint_path))
            print(f"[检查点] 已保存至: {checkpoint_path}")

    # 保存最终模型
    print("\n" + "=" * 60)
    print("保存最终模型...")
    final_model_path = model_dir / f"federated_final_{config['agent']['type']}.pt"
    server.save_global_model(str(final_model_path))
    print(f"最终模型已保存至: {final_model_path}")

    # 最终评估
    print("\n" + "=" * 60)
    print("最终评估")
    print("=" * 60)

    for i, client in enumerate(clients):
        print(f"\n[客户端 {i}] 评估...")
        eval_stats = evaluate_client(client, envs[i], num_eval_episodes=20)
        print(f"Detection Rate: {eval_stats['avg_detection_rate']:.3f}")
        print(f"False Positive Rate: {eval_stats['avg_fpr']:.3f}")
        print(f"F1 Score: {eval_stats['avg_f1']:.3f}")
        print(f"Average Reward: {eval_stats['avg_reward']:.2f}")

    # 全局统计
    print("\n" + "=" * 60)
    print("全局统计")
    print("=" * 60)
    server_stats = server.get_statistics()
    print(f"总轮次: {server_stats['num_rounds']}")
    print(f"聚合方法: {server_stats['aggregation_method']}")
    print(f"平均 Reward 趋势: {round_rewards[:5]} ... {round_rewards[-5:]}")

    return server, clients, round_rewards


def evaluate_client(client: FederatedClient, env, num_eval_episodes: int = 10):
    """评估客户端"""
    eval_rewards = []
    eval_stats = []

    for _ in range(num_eval_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = client.agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state

        eval_rewards.append(episode_reward)
        eval_stats.append(env.get_episode_statistics())

    # 计算平均统计
    avg_stats = {
        'avg_reward': np.mean(eval_rewards),
        'avg_detection_rate': np.mean([s['Detection_Rate'] for s in eval_stats]),
        'avg_fpr': np.mean([s['False_Positive_Rate'] for s in eval_stats]),
        'avg_f1': np.mean([s['F1_Score'] for s in eval_stats])
    }

    return avg_stats


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    train_federated(config_path)
