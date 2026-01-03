"""
奖励函数模块
设计原则：
- 优先保证系统安全（重罚漏报）
- 控制误报率（轻罚误报）
- 引导 Agent 学习平衡策略
"""

import numpy as np
from typing import Dict, Tuple


class RewardFunction:
    """奖励函数类"""

    # 动作定义
    ALLOW = 0   # 允许通过
    ALERT = 1   # 警告但放行
    BLOCK = 2   # 阻断

    def __init__(
        self,
        attack_blocked_reward: float = 1.0,
        attack_missed_penalty: float = -1.0,
        false_positive_penalty: float = -0.3,
        normal_allowed_reward: float = 0.2,
        alert_reward: float = 0.1
    ):
        """
        初始化奖励函数

        Args:
            attack_blocked_reward: 成功阻断攻击的奖励
            attack_missed_penalty: 攻击漏报的惩罚
            false_positive_penalty: 误报的惩罚
            normal_allowed_reward: 正确放行正常流量的奖励
            alert_reward: 警告行为的基础奖励
        """
        self.attack_blocked_reward = attack_blocked_reward
        self.attack_missed_penalty = attack_missed_penalty
        self.false_positive_penalty = false_positive_penalty
        self.normal_allowed_reward = normal_allowed_reward
        self.alert_reward = alert_reward

    def compute_reward(self, action: int, is_attack: bool) -> float:
        """
        计算奖励值

        Args:
            action: Agent 采取的动作 (0=ALLOW, 1=ALERT, 2=BLOCK)
            is_attack: 是否为攻击流量

        Returns:
            奖励值
        """
        if is_attack:
            # 攻击流量情况
            if action == self.BLOCK:
                # 成功阻断攻击
                return self.attack_blocked_reward
            elif action == self.ALERT:
                # 警告但未阻断，部分奖励
                return self.alert_reward
            else:  # ALLOW
                # 攻击漏报（严重）
                return self.attack_missed_penalty
        else:
            # 正常流量情况
            if action == self.BLOCK:
                # 误报（轻微）
                return self.false_positive_penalty
            elif action == self.ALERT:
                # 警告正常流量，轻微惩罚
                return self.alert_reward * 0.5
            else:  # ALLOW
                # 正确放行
                return self.normal_allowed_reward

    def compute_batch_rewards(
        self,
        actions: np.ndarray,
        is_attacks: np.ndarray
    ) -> np.ndarray:
        """
        批量计算奖励

        Args:
            actions: 动作数组
            is_attacks: 是否为攻击的布尔数组

        Returns:
            奖励数组
        """
        rewards = np.zeros_like(actions, dtype=np.float32)

        for i in range(len(actions)):
            rewards[i] = self.compute_reward(actions[i], is_attacks[i])

        return rewards

    def get_statistics(
        self,
        actions: np.ndarray,
        is_attacks: np.ndarray
    ) -> Dict[str, float]:
        """
        计算统计指标

        Args:
            actions: 动作数组
            is_attacks: 是否为攻击的布尔数组

        Returns:
            统计字典，包含 TP, TN, FP, FN, DR, FPR, FNR, F1
        """
        # 计算混淆矩阵
        tp = np.sum((is_attacks == True) & (actions == self.BLOCK))  # 真阳性
        tn = np.sum((is_attacks == False) & (actions == self.ALLOW))  # 真阴性
        fp = np.sum((is_attacks == False) & (actions == self.BLOCK))  # 假阳性
        fn = np.sum((is_attacks == True) & (actions != self.BLOCK))  # 假阴性

        # 计算指标
        dr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Detection Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # False Negative Rate

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = dr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'Detection_Rate': float(dr),
            'False_Positive_Rate': float(fpr),
            'False_Negative_Rate': float(fnr),
            'Precision': float(precision),
            'Recall': float(recall),
            'F1_Score': float(f1)
        }


class AdaptiveRewardFunction(RewardFunction):
    """自适应奖励函数，可根据系统状态动态调整奖励权重"""

    def __init__(
        self,
        attack_blocked_reward: float = 1.0,
        attack_missed_penalty: float = -1.0,
        false_positive_penalty: float = -0.3,
        normal_allowed_reward: float = 0.2,
        alert_reward: float = 0.1,
        adaptive_rate: float = 0.01
    ):
        super().__init__(
            attack_blocked_reward,
            attack_missed_penalty,
            false_positive_penalty,
            normal_allowed_reward,
            alert_reward
        )
        self.adaptive_rate = adaptive_rate
        self.false_positive_count = 0
        self.false_negative_count = 0

    def compute_reward(self, action: int, is_attack: bool) -> float:
        """计算奖励并更新统计"""
        reward = super().compute_reward(action, is_attack)

        # 更新统计
        if not is_attack and action == self.BLOCK:
            self.false_positive_count += 1
        elif is_attack and action != self.BLOCK:
            self.false_negative_count += 1

        return reward

    def adapt_penalties(self):
        """根据累积的错误调整惩罚权重"""
        # 如果误报过多，降低误报惩罚
        if self.false_positive_count > 100:
            self.false_positive_penalty *= (1 - self.adaptive_rate)

        # 如果漏报过多，增加漏报惩罚
        if self.false_negative_count > 50:
            self.attack_missed_penalty *= (1 + self.adaptive_rate)

    def reset_statistics(self):
        """重置统计计数器"""
        self.false_positive_count = 0
        self.false_negative_count = 0


# 预定义的奖励函数配置
CONSERVATIVE_REWARD = RewardFunction(
    attack_blocked_reward=1.0,
    attack_missed_penalty=-2.0,  # 更严格的漏报惩罚
    false_positive_penalty=-0.2,  # 更宽松的误报惩罚
    normal_allowed_reward=0.1
)

BALANCED_REWARD = RewardFunction(
    attack_blocked_reward=1.0,
    attack_missed_penalty=-1.0,
    false_positive_penalty=-0.3,
    normal_allowed_reward=0.2
)

AGGRESSIVE_REWARD = RewardFunction(
    attack_blocked_reward=1.0,
    attack_missed_penalty=-0.8,
    false_positive_penalty=-0.5,  # 更严格的误报惩罚
    normal_allowed_reward=0.3
)
