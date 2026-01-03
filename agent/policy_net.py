"""
策略网络模块
实现轻量级 MLP 网络，适合边缘设备/IoT 场景
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PolicyNetwork(nn.Module):
    """策略网络 - 轻量级 MLP"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,  # ALLOW, ALERT, BLOCK
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.1
    ):
        """
        初始化策略网络

        Args:
            state_dim: 状态维度
            action_dim: 动作维度（默认3个动作）
            hidden_dims: 隐藏层维度元组
            dropout: Dropout 概率
        """
        super(PolicyNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        # 构建网络层
        layers = []
        prev_dim = state_dim

        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            state: 状态张量 [batch_size, state_dim]

        Returns:
            Q值 [batch_size, action_dim]
        """
        return self.network(state)

    def get_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0
    ) -> Tuple[int, torch.Tensor]:
        """
        选择动作（支持 epsilon-greedy）

        Args:
            state: 状态张量 [state_dim]
            epsilon: 探索概率

        Returns:
            (action, q_values)
        """
        if torch.rand(1).item() < epsilon:
            # 探索：随机选择动作
            action = torch.randint(0, self.action_dim, (1,)).item()
        else:
            # 利用：选择最优动作
            with torch.no_grad():
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                q_values = self.forward(state)
                action = q_values.argmax(dim=1).item()

        return action, self.forward(state.unsqueeze(0) if state.dim() == 1 else state)

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """获取 Q 值"""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            return self.forward(state)


class DuelingPolicyNetwork(nn.Module):
    """Dueling DQN 策略网络"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.1
    ):
        """
        初始化 Dueling 网络

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度元组
            dropout: Dropout 概率
        """
        super(DuelingPolicyNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 共享特征提取层
        feature_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            feature_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.feature_layer = nn.Sequential(*feature_layers)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            state: 状态张量 [batch_size, state_dim]

        Returns:
            Q值 [batch_size, action_dim]
        """
        features = self.feature_layer(state)

        # 计算 Value 和 Advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # 组合成 Q 值: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def get_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0
    ) -> Tuple[int, torch.Tensor]:
        """选择动作"""
        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, self.action_dim, (1,)).item()
        else:
            with torch.no_grad():
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                q_values = self.forward(state)
                action = q_values.argmax(dim=1).item()

        return action, self.forward(state.unsqueeze(0) if state.dim() == 1 else state)

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """获取 Q 值"""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            return self.forward(state)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic 网络（用于 PPO）"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.1
    ):
        """
        初始化 Actor-Critic 网络

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度元组
            dropout: Dropout 概率
        """
        super(ActorCriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 共享特征提取层
        feature_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            feature_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.feature_layer = nn.Sequential(*feature_layers)

        # Actor（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )

        # Critic（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            state: 状态张量 [batch_size, state_dim]

        Returns:
            (action_probs, state_value)
        """
        features = self.feature_layer(state)

        # Actor 输出动作概率分布
        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits, dim=-1)

        # Critic 输出状态价值
        state_value = self.critic(features)

        return action_probs, state_value

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        选择动作

        Args:
            state: 状态张量
            deterministic: 是否确定性选择（用于评估）

        Returns:
            (action, log_prob, state_value)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        action_probs, state_value = self.forward(state)

        if deterministic:
            action = action_probs.argmax(dim=-1).item()
            log_prob = torch.log(action_probs[0, action])
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action))

        return action, log_prob, state_value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作（用于训练）

        Args:
            states: 状态张量 [batch_size, state_dim]
            actions: 动作张量 [batch_size]

        Returns:
            (log_probs, state_values, entropy)
        """
        action_probs, state_values = self.forward(states)

        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, state_values.squeeze(-1), entropy
