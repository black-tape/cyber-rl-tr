"""
本地训练脚本
单 Agent 本地训练（不使用联邦学习）
"""

import sys
import os
import yaml
import numpy as np
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from env.security_env import SecurityEnv, MockDataGenerator
from agent.rl_agent import DQNAgent, PPOAgent
from reward.reward_fn import RewardFunction


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


def train_local(config_path: str = '../config/config.yaml'):
    """本地训练主函数"""
    # 加载配置
    config = load_config(config_path)

    # 设置随机种子
    set_seed(config['training']['seed'])

    # 创建日志目录
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    # 创建模型保存目录
    model_dir = Path(config['model']['save_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("本地训练 - 单 Agent RL")
    print("=" * 50)

    # 生成模拟数据
    print("\n[1/4] 生成训练数据...")
    features, labels = MockDataGenerator.generate_mock_data(
        num_samples=config['data']['num_samples_per_client'],
        feature_dim=config['data']['feature_dim'],
        attack_ratio=0.3
    )
    print(f"数据生成完成: {len(features)} 个样本, 攻击比例: {np.mean(labels):.2%}")

    # 创建环境
    print("\n[2/4] 创建环境...")
    reward_fn = RewardFunction(
        attack_blocked_reward=config['reward']['attack_blocked_reward'],
        attack_missed_penalty=config['reward']['attack_missed_penalty'],
        false_positive_penalty=config['reward']['false_positive_penalty'],
        normal_allowed_reward=config['reward']['normal_allowed_reward'],
        alert_reward=config['reward']['alert_reward']
    )

    env = SecurityEnv(
        data=features,
        labels=labels,
        state_dim=config['environment']['state_dim'],
        history_length=config['environment']['history_length']
    )
    print(f"环境创建完成: 状态维度 = {env.state_dim}")

    # 创建 Agent
    print("\n[3/4] 创建 Agent...")
    agent = create_agent(config, env.state_dim)
    print(f"Agent 类型: {config['agent']['type'].upper()}")

    # 训练
    print("\n[4/4] 开始训练...")
    num_episodes = config['training']['num_episodes']
    batch_size = config['training']['batch_size']
    update_freq = config['training']['update_freq']
    eval_freq = config['training']['eval_freq']

    episode_rewards = []
    episode_statistics = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        step = 0
        losses = []

        while not done:
            # 选择动作
            action = agent.select_action(state, training=True)

            # 执行动作
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            # 存储经验
            if hasattr(agent, 'store_transition'):
                # DQN
                agent.store_transition(state, action, reward, next_state, done)

                # 更新网络
                if step % update_freq == 0:
                    loss = agent.update(batch_size)
                    if loss is not None:
                        losses.append(loss)
            else:
                # PPO
                agent.store_reward(reward, done)

            state = next_state
            step += 1

        # PPO 在 episode 结束时更新
        if hasattr(agent, 'clear_buffer'):
            update_info = agent.update()
            if update_info:
                losses.append(update_info.get('total_loss', 0.0))

        # 重置 episode
        agent.reset_episode()

        # 记录统计
        episode_rewards.append(episode_reward)
        stats = env.get_episode_statistics()
        episode_statistics.append(stats)

        # 打印进度
        if (episode + 1) % config['logging']['print_freq'] == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_loss = np.mean(losses) if losses else 0.0
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward (10): {avg_reward:.2f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"DR: {stats['Detection_Rate']:.3f} | "
                  f"FPR: {stats['False_Positive_Rate']:.3f}")

        # 评估
        if (episode + 1) % eval_freq == 0:
            print(f"\n--- 评估 (Episode {episode + 1}) ---")
            eval_stats = evaluate_agent(agent, env, num_eval_episodes=5)
            print(f"Detection Rate: {eval_stats['avg_detection_rate']:.3f}")
            print(f"False Positive Rate: {eval_stats['avg_fpr']:.3f}")
            print(f"F1 Score: {eval_stats['avg_f1']:.3f}")
            print(f"Average Reward: {eval_stats['avg_reward']:.2f}\n")

    # 保存模型
    print("\n保存模型...")
    model_path = model_dir / f"local_agent_{config['agent']['type']}.pt"
    torch.save(agent.get_model_weights(), model_path)
    print(f"模型已保存至: {model_path}")

    # 最终评估
    print("\n" + "=" * 50)
    print("最终评估")
    print("=" * 50)
    final_stats = evaluate_agent(agent, env, num_eval_episodes=20)
    print(f"Detection Rate: {final_stats['avg_detection_rate']:.3f}")
    print(f"False Positive Rate: {final_stats['avg_fpr']:.3f}")
    print(f"F1 Score: {final_stats['avg_f1']:.3f}")
    print(f"Average Reward: {final_stats['avg_reward']:.2f}")

    return agent, episode_rewards, episode_statistics


def evaluate_agent(agent, env, num_eval_episodes: int = 10):
    """评估 Agent"""
    eval_rewards = []
    eval_stats = []

    for _ in range(num_eval_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
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
    train_local(config_path)
