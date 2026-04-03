"""PPO 训练循环。

支持：
- GAE (Generalized Advantage Estimation) 优势计算
- 多 epoch 重复更新同一批数据
- 正确的 episode-based 批处理
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .model import ZhaJinHuaNet

__all__ = ["PPOTrainer", "Experience", "RolloutBuffer"]


@dataclass(slots=True)
class Experience:
    """单步经验数据。"""
    state: torch.Tensor       # 特征向量
    action_index: int         # 动作索引
    log_prob: float           # 旧策略的对数概率
    reward: float             # 奖励
    value: float              # 旧价值估计
    done: bool                # 是否局结束


class RolloutBuffer:
    """用于收集一批 episode 经验并计算 GAE。

    PPO 标准做法：收集一批经验 → 计算 GAE → 多 epoch 更新 → 清空。
    """

    def __init__(self) -> None:
        self.states: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def push(self, exp: Experience) -> None:
        self.states.append(exp.state)
        self.actions.append(exp.action_index)
        self.log_probs.append(exp.log_prob)
        self.rewards.append(exp.reward)
        self.values.append(exp.value)
        self.dones.append(exp.done)

    def extend(self, experiences: List[Experience]) -> None:
        for exp in experiences:
            self.push(exp)

    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
        """计算 GAE 优势和回报。

        Returns:
            (advantages, returns) 各为 shape (N,) 的 tensor
        """
        n = len(self.rewards)
        advantages = [0.0] * n
        returns = [0.0] * n
        last_gae = 0.0

        for t in reversed(range(n)):
            if self.dones[t]:
                # 局结束：无 bootstrap
                next_value = 0.0
                last_gae = 0.0
            else:
                next_value = self.values[t + 1] if t + 1 < n else 0.0

            delta = self.rewards[t] + gamma * next_value - self.values[t]
            last_gae = delta + gamma * lam * (0.0 if self.dones[t] else last_gae)
            advantages[t] = last_gae
            returns[t] = last_gae + self.values[t]

        return (
            torch.tensor(advantages, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32),
        )

    def get_tensors(self) -> Dict[str, torch.Tensor]:
        """获取所有数据的 tensor 字典。"""
        return {
            "states": torch.stack(self.states),
            "actions": torch.tensor(self.actions, dtype=torch.long),
            "old_log_probs": torch.tensor(self.log_probs, dtype=torch.float32),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32),
            "old_values": torch.tensor(self.values, dtype=torch.float32),
            "dones": torch.tensor(self.dones, dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.states)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


class PPOTrainer:
    """PPO 训练器。

    标准流程：
    1. 收集一批经验到 buffer
    2. 计算 GAE
    3. 多 epoch 小批量更新
    4. 清空 buffer
    """

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
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
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
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def train_on_buffer(self) -> Dict[str, float]:
        """在当前 buffer 上执行多 epoch PPO 更新。

        Returns:
            平均训练指标
        """
        if len(self.buffer) < self.mini_batch_size:
            return {}

        # 计算 GAE
        advantages, returns = self.buffer.compute_gae(self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 获取原始数据
        data = self.buffer.get_tensors()
        states = data["states"].to(self.device)
        actions = data["actions"].to(self.device)
        old_log_probs = data["old_log_probs"].to(self.device)

        n_samples = len(self.buffer)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            # 随机打乱索引
            indices = torch.randperm(n_samples)

            for start in range(0, n_samples, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n_samples)
                batch_idx = indices[start:end]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx].to(self.device)
                b_returns = returns[batch_idx].to(self.device)

                # 前向传播
                logits, values = self.model(b_states)
                values = values.squeeze(-1)

                # 新策略分布
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # PPO clipped surrogate
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values, b_returns)

                # 总 loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_loss += loss.item()
                num_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(1, num_updates),
            "value_loss": total_value_loss / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
            "total_loss": total_loss / max(1, num_updates),
            "num_updates": num_updates,
        }

    def save(self, path: str) -> None:
        """保存模型。"""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """加载模型。"""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
