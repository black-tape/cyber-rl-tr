"""
联邦学习客户端模块
"""

import numpy as np
from typing import Dict, Optional, Tuple
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.rl_agent import DQNAgent, PPOAgent


class FederatedClient:
    """联邦学习客户端"""

    def __init__(
        self,
        client_id: int,
        agent: any,
        local_epochs: int = 5
    ):
        """
        初始化联邦客户端

        Args:
            client_id: 客户端 ID
            agent: RL Agent 实例
            local_epochs: 本地训练轮数
        """
        self.client_id = client_id
        self.agent = agent
        self.local_epochs = local_epochs

        # 统计信息
        self.local_reward_history = []
        self.training_history = []

    def local_train(
        self,
        env: any,
        num_episodes: int,
        batch_size: int = 64,
        update_freq: int = 1
    ) -> Dict:
        """
        本地训练

        Args:
            env: 环境实例
            num_episodes: 训练 episodes
            batch_size: 批大小
            update_freq: 更新频率

        Returns:
            训练统计信息
        """
        episode_rewards = []
        episode_statistics = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            done = False
            step = 0

            while not done:
                # 选择动作
                action = self.agent.select_action(state, training=True)

                # 执行动作
                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                # 存储经验
                if hasattr(self.agent, 'store_transition'):
                    # DQN
                    self.agent.store_transition(state, action, reward, next_state, done)

                    # 更新网络
                    if step % update_freq == 0:
                        loss = self.agent.update(batch_size)
                else:
                    # PPO
                    self.agent.store_reward(reward, done)

                state = next_state
                step += 1

            # PPO 在 episode 结束时更新
            if hasattr(self.agent, 'clear_buffer'):
                self.agent.update()

            # 重置 episode
            self.agent.reset_episode()

            # 记录统计
            episode_rewards.append(episode_reward)
            if hasattr(env, 'get_episode_statistics'):
                episode_statistics.append(env.get_episode_statistics())

        # 计算平均统计
        avg_reward = np.mean(episode_rewards)
        self.local_reward_history.append(avg_reward)

        training_stats = {
            'client_id': self.client_id,
            'avg_reward': avg_reward,
            'episode_rewards': episode_rewards,
            'episode_statistics': episode_statistics,
            'num_episodes': num_episodes
        }

        self.training_history.append(training_stats)

        return training_stats

    def get_model_weights(self) -> Dict:
        """获取模型权重"""
        return self.agent.get_model_weights()

    def set_model_weights(self, weights: Dict):
        """设置模型权重"""
        self.agent.set_model_weights(weights)

    def get_local_reward(self) -> float:
        """获取局部平均 Reward"""
        if not self.local_reward_history:
            return 0.0
        return np.mean(self.local_reward_history[-10:])  # 最近 10 次的平均

    def get_statistics(self) -> Dict:
        """获取客户端统计信息"""
        return {
            'client_id': self.client_id,
            'local_reward': self.get_local_reward(),
            'reward_history': self.local_reward_history,
            'agent_stats': self.agent.get_statistics()
        }

    def reset_statistics(self):
        """重置统计信息"""
        self.local_reward_history = []
        self.training_history = []
