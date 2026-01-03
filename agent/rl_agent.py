"""
强化学习 Agent 模块
支持 DQN 和 PPO 算法
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple, Optional, List
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.policy_net import PolicyNetwork, DuelingPolicyNetwork, ActorCriticNetwork


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 10000):
        """
        初始化回放缓冲区

        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 64),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 10,
        use_dueling: bool = False,
        device: Optional[str] = None
    ):
        """
        初始化 DQN Agent

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最小探索率
            epsilon_decay: 探索率衰减
            target_update_freq: 目标网络更新频率
            use_dueling: 是否使用 Dueling DQN
            device: 设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 网络
        if use_dueling:
            self.policy_net = DuelingPolicyNetwork(
                state_dim, action_dim, hidden_dims
            ).to(self.device)
            self.target_net = DuelingPolicyNetwork(
                state_dim, action_dim, hidden_dims
            ).to(self.device)
        else:
            self.policy_net = PolicyNetwork(
                state_dim, action_dim, hidden_dims
            ).to(self.device)
            self.target_net = PolicyNetwork(
                state_dim, action_dim, hidden_dims
            ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # 经验回放
        self.replay_buffer = ReplayBuffer()

        # 统计
        self.total_rewards = 0.0
        self.episode_rewards = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作

        Args:
            state: 状态
            training: 是否训练模式

        Returns:
            动作
        """
        if training and random.random() < self.epsilon:
            # 探索
            return random.randint(0, self.action_dim - 1)
        else:
            # 利用
            state_tensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor.unsqueeze(0))
                return q_values.argmax(dim=1).item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """存储经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_rewards += reward

    def update(self, batch_size: int = 64) -> Optional[float]:
        """
        更新网络

        Args:
            batch_size: 批大小

        Returns:
            损失值
        """
        if len(self.replay_buffer) < batch_size:
            return None

        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 计算当前 Q 值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # 计算目标 Q 值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # 更新 epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def get_model_weights(self) -> Dict:
        """获取模型权重"""
        return {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict()
        }

    def set_model_weights(self, weights: Dict):
        """设置模型权重"""
        if 'policy_net' in weights:
            self.policy_net.load_state_dict(weights['policy_net'])
        if 'target_net' in weights:
            self.target_net.load_state_dict(weights['target_net'])

    def reset_episode(self):
        """重置 episode"""
        if self.total_rewards > 0:
            self.episode_rewards.append(self.total_rewards)
        self.total_rewards = 0.0

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'epsilon': self.epsilon,
            'total_rewards': self.total_rewards,
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            'buffer_size': len(self.replay_buffer)
        }


class PPOAgent:
    """PPO Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 64),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: Optional[str] = None
    ):
        """
        初始化 PPO Agent

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            learning_rate: 学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda
            clip_epsilon: PPO 裁剪参数
            value_coef: 价值损失系数
            entropy_coef: 熵奖励系数
            device: 设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 网络
        self.actor_critic = ActorCriticNetwork(
            state_dim, action_dim, hidden_dims
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # 经验缓冲
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        # 统计
        self.total_rewards = 0.0
        self.episode_rewards = []

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> int:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        action, log_prob, value = self.actor_critic.get_action(state_tensor, deterministic)

        # 存储
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())

        return action

    def store_reward(self, reward: float, done: bool):
        """存储奖励"""
        self.rewards.append(reward)
        self.dones.append(done)
        self.total_rewards += reward

    def update(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.states) == 0:
            return {}

        # 计算优势和回报
        advantages, returns = self._compute_gae()

        # 转换为张量
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # PPO 更新
        log_probs, values, entropy = self.actor_critic.evaluate_actions(states, actions)

        # 计算损失
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.MSELoss()(values, returns)
        entropy_loss = -entropy.mean()

        loss = actor_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
        self.optimizer.step()

        # 清空缓冲
        self.clear_buffer()

        return {
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'total_loss': loss.item()
        }

    def _compute_gae(self) -> Tuple[List[float], List[float]]:
        """计算 GAE"""
        advantages = []
        returns = []
        gae = 0
        next_value = 0

        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                next_value = 0
                gae = 0

            delta = self.rewards[t] + self.gamma * next_value - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
            next_value = self.values[t]

        # 标准化优势
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages.tolist(), returns

    def clear_buffer(self):
        """清空缓冲"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def reset_episode(self):
        """重置 episode"""
        if self.total_rewards > 0:
            self.episode_rewards.append(self.total_rewards)
        self.total_rewards = 0.0

    def get_model_weights(self) -> Dict:
        """获取模型权重"""
        return {'actor_critic': self.actor_critic.state_dict()}

    def set_model_weights(self, weights: Dict):
        """设置模型权重"""
        if 'actor_critic' in weights:
            self.actor_critic.load_state_dict(weights['actor_critic'])

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_rewards': self.total_rewards,
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            'buffer_size': len(self.states)
        }
