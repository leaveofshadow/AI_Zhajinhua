"""Actor-Critic 双头网络。

输入 ~204 维特征向量，输出策略分布和预期价值。
约 200K 参数，适合消费级 GPU 训练。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .features import FEATURE_DIM

__all__ = ["ZhaJinHuaNet"]

# 动作空间大小: fold(0) + call(1) + raise_2x~6x(2-6) + look(7) = 8 个基础动作
# compare 动作在推理时单独处理（选择目标后组合）
POLICY_DIM = 8


class ZhaJinHuaNet(nn.Module):
    """炸金花 Actor-Critic 网络。

    Architecture:
        Shared: Linear(input_dim, 256) → ReLU → Linear(256, 128) → ReLU
        Actor:  Linear(128, 64) → ReLU → Linear(64, policy_dim) → Softmax
        Critic: Linear(128, 64) → ReLU → Linear(64, 1) → Tanh
    """

    def __init__(self, input_dim: int = FEATURE_DIM, policy_dim: int = POLICY_DIM) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.policy_dim = policy_dim

        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # 策略头 (Actor)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, policy_dim),
        )

        # 价值头 (Critic)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # 最后一层用较小的初始化
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-2].weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            x: shape (batch, input_dim) 或 (input_dim,)

        Returns:
            (policy_logits, value)
            policy_logits: shape (batch, policy_dim) 或 (policy_dim,)
            value: shape (batch, 1) 或 (1,)
        """
        shared_features = self.shared(x)
        policy_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return policy_logits, value

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """获取动作概率分布。"""
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """获取预期价值。"""
        _, value = self.forward(x)
        return value.squeeze(-1)
