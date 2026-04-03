"""推理封装：给定状态 → 选动作。"""

from __future__ import annotations

import random
from typing import List, Optional

import torch

from engine.actions import Action, ActionType
from .features import encode_observation
from .model import ZhaJinHuaNet, POLICY_DIM

__all__ = ["Agent"]


class Agent:
    """AI 推理封装。

    支持 epsilon-greedy 探索（训练时）和 greedy 推理（对战时）。
    """

    def __init__(
        self,
        model: Optional[ZhaJinHuaNet] = None,
        model_path: Optional[str] = None,
        device: str = "cpu",
        epsilon: float = 0.0,
    ) -> None:
        self.device = torch.device(device)
        self.epsilon = epsilon

        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None:
            self.model = ZhaJinHuaNet().to(self.device)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            self.model = ZhaJinHuaNet().to(self.device)

        self.model.eval()

    @torch.no_grad()
    def act(
        self,
        observation: dict,
        valid_actions: List[Action],
    ) -> Action:
        """根据观察选择动作。

        Args:
            observation: game.get_observation() 返回的字典
            valid_actions: 当前合法动作列表

        Returns:
            选中的 Action
        """
        if not valid_actions:
            return Action(ActionType.FOLD)

        # Epsilon-greedy 探索
        if self.epsilon > 0 and random.random() < self.epsilon:
            return random.choice(valid_actions)

        # 将合法动作映射到策略网络输出索引
        valid_indices = [self._action_to_index(a) for a in valid_actions]

        # 网络推理
        features = encode_observation(observation).unsqueeze(0).to(self.device)
        logits, _ = self.model(features)
        logits = logits.squeeze(0)

        # 屏蔽非法动作（设为 -inf）
        mask = torch.full_like(logits, float("-inf"))
        for idx in valid_indices:
            mask[idx] = 0.0
        masked_logits = logits + mask

        # 选概率最大的动作 (greedy)
        best_idx = masked_logits.argmax().item()

        # 映射回 Action
        return self._index_to_action(best_idx, valid_actions)

    @torch.no_grad()
    def act_batch(
        self,
        observations: list,
        valid_actions_list: List[List[Action]],
    ) -> List[Action]:
        """批量推理。"""
        results = []
        for obs, valid in zip(observations, valid_actions_list):
            results.append(self.act(obs, valid))
        return results

    def set_epsilon(self, epsilon: float) -> None:
        """设置探索率。"""
        self.epsilon = epsilon

    @staticmethod
    def _action_to_index(action: Action) -> int:
        """将 Action 映射到策略网络输出索引 (0-7)。"""
        at = action.action_type
        if at == ActionType.FOLD:
            return 0
        elif at == ActionType.CALL:
            return 1
        elif ActionType.RAISE_2X <= at <= ActionType.RAISE_6X:
            return int(at)  # 2-6
        elif at == ActionType.LOOK:
            return 7
        elif at == ActionType.COMPARE:
            return 0  # compare 用 fold 位置近似，后续由 valid_actions 约束
        return 0

    @staticmethod
    def _index_to_action(index: int, valid_actions: List[Action]) -> Action:
        """将策略网络输出索引映射回最佳匹配的合法 Action。"""
        # 优先匹配基础动作类型
        for action in valid_actions:
            at = action.action_type
            if at == ActionType.FOLD and index == 0:
                return action
            elif at == ActionType.CALL and index == 1:
                return action
            elif ActionType.RAISE_2X <= at <= ActionType.RAISE_6X and index == int(at):
                return action
            elif at == ActionType.LOOK and index == 7:
                return action
            elif at == ActionType.COMPARE and index == 0:
                return action
        # 兜底：返回第一个合法动作
        return valid_actions[0]
