"""
多 Agent 控制器模块
管理所有 Agent，协调联邦学习流程
"""

import numpy as np
from typing import Dict, List, Optional, Any
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.rl_agent import DQNAgent, PPOAgent


class MultiAgentController:
    """多 Agent 控制器"""

    def __init__(self, agent_type: str = 'dqn'):
        """
        初始化多 Agent 控制器

        Args:
            agent_type: Agent 类型 ('dqn' 或 'ppo')
        """
        self.agent_type = agent_type
        self.agents: Dict[int, Any] = {}
        self.agent_rewards: Dict[int, List[float]] = {}
        self.agent_statistics: Dict[int, Dict] = {}

    def register_agent(
        self,
        agent_id: int,
        agent: Any
    ):
        """
        注册 Agent

        Args:
            agent_id: Agent ID
            agent: Agent 实例
        """
        self.agents[agent_id] = agent
        self.agent_rewards[agent_id] = []
        self.agent_statistics[agent_id] = {}

    def create_agent(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int = 3,
        **kwargs
    ):
        """
        创建并注册 Agent

        Args:
            agent_id: Agent ID
            state_dim: 状态维度
            action_dim: 动作维度
            **kwargs: 其他参数
        """
        if self.agent_type == 'dqn':
            agent = DQNAgent(state_dim, action_dim, **kwargs)
        elif self.agent_type == 'ppo':
            agent = PPOAgent(state_dim, action_dim, **kwargs)
        else:
            raise ValueError(f"不支持的 Agent 类型: {self.agent_type}")

        self.register_agent(agent_id, agent)

    def get_agent(self, agent_id: int):
        """获取指定 Agent"""
        return self.agents.get(agent_id)

    def collect_local_rewards(self) -> Dict[int, float]:
        """
        收集各 Agent 的局部平均 Reward

        Returns:
            {agent_id: avg_reward}
        """
        local_rewards = {}

        for agent_id, agent in self.agents.items():
            stats = agent.get_statistics()
            avg_reward = stats.get('avg_episode_reward', 0.0)
            local_rewards[agent_id] = avg_reward

        return local_rewards

    def collect_model_weights(self) -> Dict[int, Dict]:
        """
        收集各 Agent 的模型权重

        Returns:
            {agent_id: weights}
        """
        model_weights = {}

        for agent_id, agent in self.agents.items():
            model_weights[agent_id] = agent.get_model_weights()

        return model_weights

    def broadcast_global_model(self, global_weights: Dict):
        """
        广播全局模型到所有 Agent

        Args:
            global_weights: 全局模型权重
        """
        for agent in self.agents.values():
            agent.set_model_weights(global_weights)

    def update_agent_statistics(self, agent_id: int, statistics: Dict):
        """
        更新 Agent 统计信息

        Args:
            agent_id: Agent ID
            statistics: 统计字典
        """
        self.agent_statistics[agent_id] = statistics

    def get_all_statistics(self) -> Dict[int, Dict]:
        """获取所有 Agent 的统计信息"""
        all_stats = {}

        for agent_id, agent in self.agents.items():
            agent_stats = agent.get_statistics()
            agent_stats.update(self.agent_statistics.get(agent_id, {}))
            all_stats[agent_id] = agent_stats

        return all_stats

    def reset_all_episodes(self):
        """重置所有 Agent 的 episode"""
        for agent in self.agents.values():
            agent.reset_episode()

    def get_num_agents(self) -> int:
        """获取 Agent 数量"""
        return len(self.agents)

    def compute_aggregation_weights(
        self,
        temperature: float = 1.0,
        method: str = 'reward_based'
    ) -> Dict[int, float]:
        """
        计算聚合权重

        Args:
            temperature: 温度参数
            method: 聚合方法 ('reward_based', 'uniform')

        Returns:
            {agent_id: weight}
        """
        if method == 'uniform':
            # 均匀权重
            num_agents = len(self.agents)
            return {agent_id: 1.0 / num_agents for agent_id in self.agents.keys()}

        elif method == 'reward_based':
            # 基于 Reward 的动态权重
            local_rewards = self.collect_local_rewards()

            # 计算 softmax 权重
            reward_values = np.array(list(local_rewards.values()))

            # 避免数值不稳定
            reward_values = reward_values / temperature
            reward_values = reward_values - np.max(reward_values)

            exp_rewards = np.exp(reward_values)
            softmax_weights = exp_rewards / np.sum(exp_rewards)

            # 构建权重字典
            weights = {}
            for i, agent_id in enumerate(local_rewards.keys()):
                weights[agent_id] = float(softmax_weights[i])

            return weights

        else:
            raise ValueError(f"不支持的聚合方法: {method}")

    def trigger_federated_aggregation(
        self,
        aggregation_weights: Optional[Dict[int, float]] = None
    ) -> Dict:
        """
        触发联邦聚合

        Args:
            aggregation_weights: 聚合权重（None=使用均匀权重）

        Returns:
            全局模型权重
        """
        # 收集模型权重
        model_weights = self.collect_model_weights()

        if aggregation_weights is None:
            # 使用均匀权重
            aggregation_weights = self.compute_aggregation_weights(method='uniform')

        # 执行加权聚合
        global_weights = self._aggregate_weights(model_weights, aggregation_weights)

        return global_weights

    def _aggregate_weights(
        self,
        model_weights: Dict[int, Dict],
        aggregation_weights: Dict[int, float]
    ) -> Dict:
        """
        加权聚合模型权重

        Args:
            model_weights: {agent_id: weights}
            aggregation_weights: {agent_id: weight}

        Returns:
            全局模型权重
        """
        # 获取第一个 Agent 的权重作为模板
        first_agent_id = next(iter(model_weights))
        global_weights = {}

        # 遍历所有网络
        for network_name in model_weights[first_agent_id].keys():
            network_weights = {}

            # 遍历所有层
            for param_name in model_weights[first_agent_id][network_name].keys():
                # 加权平均
                weighted_sum = None

                for agent_id, weights in model_weights.items():
                    param = weights[network_name][param_name]
                    weight = aggregation_weights[agent_id]

                    if weighted_sum is None:
                        weighted_sum = param * weight
                    else:
                        weighted_sum += param * weight

                network_weights[param_name] = weighted_sum

            global_weights[network_name] = network_weights

        return global_weights

    def save_agents(self, save_dir: str):
        """
        保存所有 Agent

        Args:
            save_dir: 保存目录
        """
        import torch
        import os

        os.makedirs(save_dir, exist_ok=True)

        for agent_id, agent in self.agents.items():
            weights = agent.get_model_weights()
            save_path = os.path.join(save_dir, f'agent_{agent_id}.pt')
            torch.save(weights, save_path)

    def load_agents(self, load_dir: str):
        """
        加载所有 Agent

        Args:
            load_dir: 加载目录
        """
        import torch
        import os

        for agent_id, agent in self.agents.items():
            load_path = os.path.join(load_dir, f'agent_{agent_id}.pt')
            if os.path.exists(load_path):
                weights = torch.load(load_path)
                agent.set_model_weights(weights)

    def get_summary(self) -> Dict:
        """
        获取控制器摘要信息

        Returns:
            摘要字典
        """
        all_stats = self.get_all_statistics()
        local_rewards = self.collect_local_rewards()

        summary = {
            'num_agents': self.get_num_agents(),
            'agent_type': self.agent_type,
            'local_rewards': local_rewards,
            'avg_reward_across_agents': np.mean(list(local_rewards.values())) if local_rewards else 0.0,
            'agent_statistics': all_stats
        }

        return summary
