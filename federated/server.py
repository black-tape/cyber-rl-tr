"""
联邦学习服务器模块
实现强化型聚合策略
"""

import numpy as np
from typing import Dict, List, Optional
import torch


class FederatedServer:
    """联邦学习服务器（强化型聚合）"""

    def __init__(
        self,
        aggregation_method: str = 'reward_based',
        temperature: float = 1.0,
        min_weight: float = 0.01
    ):
        """
        初始化联邦服务器

        Args:
            aggregation_method: 聚合方法 ('reward_based', 'uniform', 'weighted_avg')
            temperature: 温度参数（控制权重分布平滑度）
            min_weight: 最小权重阈值
        """
        self.aggregation_method = aggregation_method
        self.temperature = temperature
        self.min_weight = min_weight

        # 全局模型
        self.global_weights = None

        # 聚合历史
        self.aggregation_history = []
        self.round_statistics = []

    def compute_aggregation_weights(
        self,
        client_rewards: Dict[int, float]
    ) -> Dict[int, float]:
        """
        计算聚合权重

        Args:
            client_rewards: {client_id: avg_reward}

        Returns:
            {client_id: weight}
        """
        num_clients = len(client_rewards)

        if self.aggregation_method == 'uniform':
            # 传统 FedAvg：均匀权重
            return {cid: 1.0 / num_clients for cid in client_rewards.keys()}

        elif self.aggregation_method == 'reward_based':
            # 强化型聚合：基于 Reward 的动态权重
            # 权重 w_i = softmax(avg_reward_i / temperature)

            client_ids = list(client_rewards.keys())
            rewards = np.array([client_rewards[cid] for cid in client_ids])

            # Softmax 计算
            rewards_scaled = rewards / self.temperature
            rewards_scaled = rewards_scaled - np.max(rewards_scaled)  # 数值稳定性

            exp_rewards = np.exp(rewards_scaled)
            softmax_weights = exp_rewards / np.sum(exp_rewards)

            # 应用最小权重阈值
            softmax_weights = np.maximum(softmax_weights, self.min_weight)
            softmax_weights = softmax_weights / np.sum(softmax_weights)  # 重新归一化

            # 构建权重字典
            aggregation_weights = {
                client_ids[i]: float(softmax_weights[i])
                for i in range(num_clients)
            }

            return aggregation_weights

        elif self.aggregation_method == 'weighted_avg':
            # 加权平均（基于数据量等）
            # 这里简化为基于 reward 的线性权重
            total_reward = sum(client_rewards.values())
            if total_reward <= 0:
                return {cid: 1.0 / num_clients for cid in client_rewards.keys()}

            return {
                cid: max(reward / total_reward, self.min_weight)
                for cid, reward in client_rewards.items()
            }

        else:
            raise ValueError(f"不支持的聚合方法: {self.aggregation_method}")

    def aggregate(
        self,
        client_weights: Dict[int, Dict],
        client_rewards: Dict[int, float]
    ) -> Dict:
        """
        聚合客户端模型

        Args:
            client_weights: {client_id: model_weights}
            client_rewards: {client_id: avg_reward}

        Returns:
            全局模型权重
        """
        # 计算聚合权重
        aggregation_weights = self.compute_aggregation_weights(client_rewards)

        # 记录聚合信息
        self.aggregation_history.append({
            'client_rewards': client_rewards.copy(),
            'aggregation_weights': aggregation_weights.copy()
        })

        # 执行加权聚合
        global_weights = self._weighted_aggregate(client_weights, aggregation_weights)

        self.global_weights = global_weights

        return global_weights

    def _weighted_aggregate(
        self,
        client_weights: Dict[int, Dict],
        aggregation_weights: Dict[int, float]
    ) -> Dict:
        """
        加权聚合模型参数

        Args:
            client_weights: {client_id: model_weights}
            aggregation_weights: {client_id: weight}

        Returns:
            全局模型权重
        """
        # 获取第一个客户端的权重作为模板
        first_client_id = next(iter(client_weights))
        global_weights = {}

        # 遍历所有网络（如 policy_net, target_net 等）
        for network_name in client_weights[first_client_id].keys():
            network_weights = {}

            # 遍历所有层参数
            for param_name in client_weights[first_client_id][network_name].keys():
                # 加权平均
                weighted_param = None

                for client_id, weights in client_weights.items():
                    param = weights[network_name][param_name]
                    weight = aggregation_weights[client_id]

                    if weighted_param is None:
                        weighted_param = param.clone() * weight
                    else:
                        weighted_param += param * weight

                network_weights[param_name] = weighted_param

            global_weights[network_name] = network_weights

        return global_weights

    def get_global_weights(self) -> Optional[Dict]:
        """获取全局模型权重"""
        return self.global_weights

    def set_global_weights(self, weights: Dict):
        """设置全局模型权重"""
        self.global_weights = weights

    def record_round_statistics(
        self,
        round_num: int,
        client_statistics: Dict[int, Dict]
    ):
        """
        记录联邦轮次统计

        Args:
            round_num: 轮次编号
            client_statistics: {client_id: statistics}
        """
        # 计算全局统计
        all_rewards = [stats.get('avg_reward', 0.0) for stats in client_statistics.values()]

        round_stats = {
            'round': round_num,
            'avg_reward': np.mean(all_rewards),
            'min_reward': np.min(all_rewards),
            'max_reward': np.max(all_rewards),
            'std_reward': np.std(all_rewards),
            'client_statistics': client_statistics
        }

        self.round_statistics.append(round_stats)

    def get_statistics(self) -> Dict:
        """获取服务器统计信息"""
        return {
            'aggregation_method': self.aggregation_method,
            'temperature': self.temperature,
            'num_rounds': len(self.round_statistics),
            'aggregation_history': self.aggregation_history,
            'round_statistics': self.round_statistics
        }

    def save_global_model(self, save_path: str):
        """
        保存全局模型

        Args:
            save_path: 保存路径
        """
        if self.global_weights is None:
            raise ValueError("全局模型尚未初始化")

        torch.save(self.global_weights, save_path)

    def load_global_model(self, load_path: str):
        """
        加载全局模型

        Args:
            load_path: 加载路径
        """
        self.global_weights = torch.load(load_path)

    def get_best_client(self) -> Optional[int]:
        """获取表现最好的客户端 ID"""
        if not self.aggregation_history:
            return None

        latest_rewards = self.aggregation_history[-1]['client_rewards']
        best_client = max(latest_rewards, key=latest_rewards.get)

        return best_client

    def get_aggregation_weights_summary(self) -> Dict:
        """获取聚合权重摘要"""
        if not self.aggregation_history:
            return {}

        latest = self.aggregation_history[-1]

        return {
            'latest_rewards': latest['client_rewards'],
            'latest_weights': latest['aggregation_weights'],
            'weight_distribution': self._compute_weight_distribution(
                latest['aggregation_weights']
            )
        }

    def _compute_weight_distribution(self, weights: Dict[int, float]) -> Dict:
        """计算权重分布统计"""
        weight_values = list(weights.values())

        return {
            'mean': np.mean(weight_values),
            'std': np.std(weight_values),
            'min': np.min(weight_values),
            'max': np.max(weight_values),
            'entropy': self._compute_entropy(weight_values)
        }

    def _compute_entropy(self, weights: List[float]) -> float:
        """计算权重分布的熵"""
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # 归一化

        # 避免 log(0)
        weights = weights[weights > 0]

        entropy = -np.sum(weights * np.log(weights))

        return float(entropy)
