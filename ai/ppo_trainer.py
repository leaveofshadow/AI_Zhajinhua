"""PPO 训练循环。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .model import ZhaJinHuaNet

__all__ = ["PPOTrainer", "Experience", "ReplayBuffer"]


@dataclass(slots=True)
class Experience:
    """单步经验数据。"""
    state: torch.Tensor       # 特征向量
    action_index: int         # 动作索引
    log_prob: float           # 旧策略的对数概率
    reward: float             # 奖励
    value: float              # 旧价值估计
    done: bool                # 是否局结束


class ReplayBuffer:
    """经验回放缓冲区。"""

    def __init__(self, capacity: int = 100000) -> None:
        self.capacity = capacity
        self.buffer: List[Experience] = []

    def push(self, exp: Experience) -> None:
        self.buffer.append(exp)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]

    def extend(self, experiences: List[Experience]) -> None:
        self.buffer.extend(experiences)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]

    def sample(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """采样一个 batch 并计算 GAE 优势。"""
        if len(self.buffer) < batch_size:
            return None

        import random
        samples = random.sample(self.buffer, batch_size)

        states = torch.stack([s.state for s in samples])
        actions = torch.tensor([s.action_index for s in samples], dtype=torch.long)
        old_log_probs = torch.tensor([s.log_prob for s in samples], dtype=torch.float32)
        rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32)
        old_values = torch.tensor([s.value for s in samples], dtype=torch.float32)
        dones = torch.tensor([s.done for s in samples], dtype=torch.float32)

        return {
            "states": states,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "rewards": rewards,
            "old_values": old_values,
            "dones": dones,
        }

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class PPOTrainer:
    """PPO 训练器。"""

    def __init__(
        self,
        model: ZhaJinHuaNet,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

    def train_step(self, batch_size: int = 256) -> Optional[Dict[str, float]]:
        """执行一个 PPO 训练步。"""
        batch = self.buffer.sample(batch_size)
        if batch is None:
            return None

        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["old_log_probs"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        old_values = batch["old_values"].to(self.device)
        dones = batch["dones"].to(self.device)

        # 计算 returns 和 advantages (简单版本：无 GAE)
        returns = rewards  # 局末奖励直接作为 return
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 前向传播
        logits, values = self.model(states)
        values = values.squeeze(-1)

        # 新的策略分布
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # PPO clipped surrogate
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.MSELoss()(values, returns)

        # 总 loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
        }

    def save(self, path: str) -> None:
        """保存模型。"""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """加载模型。"""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
