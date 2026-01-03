"""
轻量级多 Agent 联邦学习系统
不使用强化学习，基于简单有效的规则和统计方法
适合网络安全入侵检测场景
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class AnomalyDetectionAgent:
    """
    异常检测 Agent
    使用简单统计方法检测客户端更新中的异常
    """

    def __init__(self, threshold_percentile: float = 95):
        """
        初始化异常检测 Agent

        Args:
            threshold_percentile: 异常阈值百分位数
        """
        self.threshold_percentile = threshold_percentile
        self.historical_updates = []

    def compute_anomaly_score(
        self,
        client_update: Dict,
        global_model: Dict,
        all_updates: List[Dict]
    ) -> float:
        """
        计算客户端更新的异常分数

        Args:
            client_update: 客户端模型更新
            global_model: 全局模型
            all_updates: 所有客户端更新列表

        Returns:
            异常分数 (0-1，越高越异常)
        """
        # 1. 计算更新的 L2 范数
        update_norm = self._compute_update_norm(client_update)

        # 2. 计算与全局模型的余弦相似度
        cosine_sim = self._compute_cosine_similarity(client_update, global_model)

        # 3. 计算与其他客户端更新的距离
        peer_distance = self._compute_peer_distance(client_update, all_updates)

        # 4. 组合异常分数
        # 范数过大 + 相似度过低 + 与peers距离过大 = 异常
        norm_score = self._normalize_score(update_norm, [self._compute_update_norm(u) for u in all_updates])
        similarity_score = 1 - cosine_sim  # 相似度低 = 异常高
        distance_score = self._normalize_score(peer_distance, [self._compute_peer_distance(u, all_updates) for u in all_updates])

        # 加权组合
        anomaly_score = 0.3 * norm_score + 0.4 * similarity_score + 0.3 * distance_score

        return float(np.clip(anomaly_score, 0, 1))

    def _compute_update_norm(self, update: Dict) -> float:
        """计算更新的 L2 范数"""
        norm = 0.0
        for param_name, param in update.items():
            norm += np.sum(param.cpu().numpy() ** 2)
        return np.sqrt(norm)

    def _compute_cosine_similarity(self, update1: Dict, update2: Dict) -> float:
        """计算两个更新之间的余弦相似度"""
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0

        for param_name in update1.keys():
            p1 = update1[param_name].cpu().numpy().flatten()
            p2 = update2[param_name].cpu().numpy().flatten()

            dot_product += np.dot(p1, p2)
            norm1 += np.sum(p1 ** 2)
            norm2 += np.sum(p2 ** 2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (np.sqrt(norm1) * np.sqrt(norm2)))

    def _compute_peer_distance(self, update: Dict, all_updates: List[Dict]) -> float:
        """计算与其他客户端的平均距离"""
        if len(all_updates) <= 1:
            return 0.0

        distances = []
        for other_update in all_updates:
            if other_update is update:
                continue

            # 计算 L2 距离
            dist = 0.0
            for param_name in update.keys():
                p1 = update[param_name].cpu().numpy().flatten()
                p2 = other_update[param_name].cpu().numpy().flatten()
                dist += np.sum((p1 - p2) ** 2)

            distances.append(np.sqrt(dist))

        return float(np.mean(distances))

    def _normalize_score(self, value: float, all_values: List[float]) -> float:
        """将分数归一化到 [0, 1]"""
        if len(all_values) == 0:
            return 0.0

        min_val = np.min(all_values)
        max_val = np.max(all_values)

        if max_val == min_val:
            return 0.0

        return (value - min_val) / (max_val - min_val)


class TrustAgent:
    """
    信誉/信任 Agent
    维护客户端的历史信誉分数
    """

    def __init__(self, decay_rate: float = 0.9, initial_trust: float = 1.0):
        """
        初始化信誉 Agent

        Args:
            decay_rate: 信誉衰减率（指数移动平均的 alpha）
            initial_trust: 初始信誉值
        """
        self.decay_rate = decay_rate
        self.initial_trust = initial_trust
        self.trust_scores: Dict[int, float] = defaultdict(lambda: initial_trust)
        self.performance_history: Dict[int, List[float]] = defaultdict(list)

    def update_trust(
        self,
        client_id: int,
        current_performance: float,
        anomaly_score: float
    ):
        """
        更新客户端信誉分数

        Args:
            client_id: 客户端 ID
            current_performance: 当前性能指标 (如 accuracy, f1)
            anomaly_score: 当前异常分数
        """
        # 性能好且异常分数低 -> 信誉上升
        # 性能差或异常分数高 -> 信誉下降
        performance_factor = current_performance
        anomaly_penalty = 1 - anomaly_score

        # 综合评分
        current_score = performance_factor * anomaly_penalty

        # 指数移动平均更新信誉
        old_trust = self.trust_scores[client_id]
        new_trust = self.decay_rate * old_trust + (1 - self.decay_rate) * current_score

        self.trust_scores[client_id] = float(np.clip(new_trust, 0, 1))
        self.performance_history[client_id].append(current_performance)

    def get_trust(self, client_id: int) -> float:
        """获取客户端信誉分数"""
        return self.trust_scores[client_id]

    def get_all_trust_scores(self) -> Dict[int, float]:
        """获取所有客户端的信誉分数"""
        return dict(self.trust_scores)

    def reset_trust(self, client_id: int):
        """重置客户端信誉"""
        self.trust_scores[client_id] = self.initial_trust


class AggregationAgent:
    """
    聚合 Agent
    基于异常分数和信誉计算自适应聚合权重
    """

    def __init__(self, temperature: float = 1.0, min_weight: float = 0.01):
        """
        初始化聚合 Agent

        Args:
            temperature: 温度参数（控制权重分布的平滑度）
            min_weight: 最小权重阈值
        """
        self.temperature = temperature
        self.min_weight = min_weight

    def compute_aggregation_weights(
        self,
        anomaly_scores: Dict[int, float],
        trust_scores: Dict[int, float]
    ) -> Dict[int, float]:
        """
        计算聚合权重

        Args:
            anomaly_scores: {client_id: anomaly_score}
            trust_scores: {client_id: trust_score}

        Returns:
            {client_id: aggregation_weight}
        """
        client_ids = list(anomaly_scores.keys())

        # 计算每个客户端的综合分数
        # 高信誉 + 低异常 = 高权重
        scores = {}
        for cid in client_ids:
            trust = trust_scores.get(cid, 0.5)
            anomaly = anomaly_scores.get(cid, 0.5)

            # 综合分数：信誉 / (1 + 异常)
            score = trust / (1.0 + anomaly)
            scores[cid] = score

        # Softmax 归一化
        weights = self._softmax_weights(scores, self.temperature)

        # 应用最小权重阈值
        weights = {cid: max(w, self.min_weight) for cid, w in weights.items()}

        # 重新归一化
        total_weight = sum(weights.values())
        weights = {cid: w / total_weight for cid, w in weights.items()}

        return weights

    def _softmax_weights(
        self,
        scores: Dict[int, float],
        temperature: float
    ) -> Dict[int, float]:
        """Softmax 归一化"""
        client_ids = list(scores.keys())
        score_values = np.array([scores[cid] for cid in client_ids])

        # 温度缩放
        scaled_scores = score_values / temperature

        # 数值稳定性
        scaled_scores = scaled_scores - np.max(scaled_scores)

        # Softmax
        exp_scores = np.exp(scaled_scores)
        softmax_weights = exp_scores / np.sum(exp_scores)

        return {client_ids[i]: float(softmax_weights[i]) for i in range(len(client_ids))}


class MultiAgentFederatedSystem:
    """
    多 Agent 联邦学习系统
    集成异常检测、信誉管理和自适应聚合
    """

    def __init__(
        self,
        num_clients: int,
        anomaly_threshold: float = 95,
        trust_decay: float = 0.9,
        temperature: float = 1.0
    ):
        """
        初始化多 Agent 系统

        Args:
            num_clients: 客户端数量
            anomaly_threshold: 异常检测阈值
            trust_decay: 信誉衰减率
            temperature: 聚合温度参数
        """
        self.num_clients = num_clients

        # 初始化三个 Agent
        self.anomaly_agent = AnomalyDetectionAgent(anomaly_threshold)
        self.trust_agent = TrustAgent(trust_decay)
        self.aggregation_agent = AggregationAgent(temperature)

        # 统计信息
        self.round_history = []

    def aggregate_updates(
        self,
        client_updates: Dict[int, Dict],
        global_model: Dict,
        client_performances: Dict[int, float]
    ) -> Tuple[Dict, Dict[int, float], Dict]:
        """
        多 Agent 协同聚合

        Args:
            client_updates: {client_id: model_update}
            global_model: 当前全局模型
            client_performances: {client_id: performance_metric}

        Returns:
            (aggregated_model, aggregation_weights, statistics)
        """
        client_ids = list(client_updates.keys())
        all_updates = list(client_updates.values())

        # Step 1: 异常检测 Agent 计算异常分数
        anomaly_scores = {}
        for cid in client_ids:
            score = self.anomaly_agent.compute_anomaly_score(
                client_updates[cid],
                global_model,
                all_updates
            )
            anomaly_scores[cid] = score

        # Step 2: 信誉 Agent 更新信誉分数
        for cid in client_ids:
            self.trust_agent.update_trust(
                cid,
                client_performances.get(cid, 0.5),
                anomaly_scores[cid]
            )

        trust_scores = self.trust_agent.get_all_trust_scores()

        # Step 3: 聚合 Agent 计算权重
        aggregation_weights = self.aggregation_agent.compute_aggregation_weights(
            anomaly_scores,
            trust_scores
        )

        # Step 4: 加权聚合模型
        aggregated_model = self._weighted_aggregate(client_updates, aggregation_weights)

        # 统计信息
        statistics = {
            'anomaly_scores': anomaly_scores,
            'trust_scores': trust_scores,
            'aggregation_weights': aggregation_weights
        }

        self.round_history.append(statistics)

        return aggregated_model, aggregation_weights, statistics

    def _weighted_aggregate(
        self,
        client_updates: Dict[int, Dict],
        weights: Dict[int, float]
    ) -> Dict:
        """加权聚合模型参数"""
        # 获取第一个客户端的模型结构
        first_client = next(iter(client_updates))
        aggregated_model = {}

        # 遍历所有参数
        for param_name in client_updates[first_client].keys():
            weighted_param = None

            for client_id, update in client_updates.items():
                param = update[param_name]
                weight = weights[client_id]

                if weighted_param is None:
                    weighted_param = param.clone() * weight
                else:
                    weighted_param += param * weight

            aggregated_model[param_name] = weighted_param

        return aggregated_model

    def get_statistics_summary(self) -> Dict:
        """获取统计摘要"""
        if not self.round_history:
            return {}

        latest = self.round_history[-1]

        return {
            'latest_anomaly_scores': latest['anomaly_scores'],
            'latest_trust_scores': latest['trust_scores'],
            'latest_weights': latest['aggregation_weights'],
            'num_rounds': len(self.round_history)
        }
