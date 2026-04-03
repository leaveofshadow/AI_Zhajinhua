"""观察空间 → 特征向量编码。

将 game.get_observation() 返回的 dict 编码为固定长度的 tensor（~246 维）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from engine.cards import Card

__all__ = ["encode_observation", "FEATURE_DIM"]

# 特征维度
# 手牌: 52 * 3 = 156 (每张牌52维one-hot，最多3张)
# 公共信息: 6 维 (my_chips_norm, pot_norm, current_bet_norm, active_players_norm, round_count_norm, has_looked)
# 位置编码: 6 * 2 = 12 维 (sin/cos for each of 6 possible positions)
# 对手信息: 6 * 5 = 30 维 (6个对手各5维: is_active, is_looked, total_bet_norm, chip_norm, last_action_code)
FEATURE_DIM = 156 + 6 + 12 + 30  # = 204


def encode_observation(obs: Dict[str, Any]) -> torch.Tensor:
    """将观察字典编码为特征向量。

    Args:
        obs: game.get_observation() 返回的字典

    Returns:
        shape (FEATURE_DIM,) 的 float32 tensor
    """
    features: List[float] = []

    # --- 手牌编码 (156 维) ---
    card_vec = [0.0] * 156  # 3 * 52
    cards = obs.get("my_cards", [])
    if cards:
        for i, card in enumerate(cards[:3]):
            if isinstance(card, Card):
                card_vec[i * 52 + card.id] = 1.0
    features.extend(card_vec)

    # --- 公共信息 (6 维) ---
    my_chips = obs.get("my_chips", 0)
    pot = obs.get("pot", 0)
    current_bet = obs.get("current_bet", 1)
    active_players = obs.get("active_players", 1)
    round_count = obs.get("round_count", 0)
    has_looked = 1.0 if obs.get("has_looked", False) else 0.0

    max_chips = 10000.0  # 归一化常数
    features.extend([
        my_chips / max_chips,
        pot / max_chips,
        current_bet / max_chips,
        active_players / 6.0,
        round_count / 50.0,
        has_looked,
    ])

    # --- 位置编码 (12 维: 6 positions * sin/cos) ---
    position = obs.get("my_position", 0)
    for i in range(6):
        angle = position * 2 * math.pi / 6
        features.append(math.sin(angle + i * math.pi / 6))
        features.append(math.cos(angle + i * math.pi / 6))

    # --- 对手信息 (30 维: 6 players * 5 features) ---
    player_states = obs.get("player_states", [])
    action_map = {"": 0, "fold": 1, "call": 2, "look": 3, "raise": 4, "compare": 5}
    for i in range(6):
        if i < len(player_states):
            ps = player_states[i]
            last_action = str(ps.get("last_action", ""))
            # 提取动作前缀
            action_code = 0
            for key in action_map:
                if last_action.startswith(key) and key:
                    action_code = action_map[key]
                    break
            features.extend([
                1.0 if ps.get("is_active", False) else 0.0,
                1.0 if ps.get("is_looked", False) else 0.0,
                ps.get("total_bet", 0) / max_chips,
                ps.get("chip_count", 0) / max_chips,
                action_code / 5.0,
            ])
        else:
            features.extend([0.0] * 5)

    return torch.tensor(features, dtype=torch.float32)


def encode_batch(observations: List[Dict[str, Any]]) -> torch.Tensor:
    """批量编码观察。"""
    return torch.stack([encode_observation(obs) for obs in observations])
