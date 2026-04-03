"""动作系统：定义炸金花所有可用动作及其验证规则。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .game import GameState

__all__ = ["ActionType", "Action", "ActionValidator"]


class ActionType(IntEnum):
    """动作类型枚举。"""
    FOLD = 0       # 弃牌
    CALL = 1       # 跟注
    RAISE_2X = 2   # 加注到 2 倍
    RAISE_3X = 3   # 加注到 3 倍
    RAISE_4X = 4   # 加注到 4 倍
    RAISE_5X = 5   # 加注到 5 倍
    RAISE_6X = 6   # 加注到 6 倍
    LOOK = 7       # 看牌
    COMPARE = 8    # 比牌（需指定目标）


@dataclass(slots=True)
class Action:
    """一个玩家动作。"""
    action_type: ActionType
    target: int = -1  # 比牌目标座位号（仅 COMPARE 时使用）

    @property
    def raise_multiplier(self) -> int:
        """加注倍数（仅 RAISE 系列动作有效）。"""
        if ActionType.RAISE_2X <= self.action_type <= ActionType.RAISE_6X:
            return int(self.action_type)
        return 0

    def __repr__(self) -> str:
        if self.action_type == ActionType.COMPARE:
            return f"Action(COMPARE, target={self.target})"
        return f"Action({self.action_type.name})"


class ActionValidator:
    """动作验证器：检查动作合法性并生成合法动作列表。"""

    @staticmethod
    def validate(action: Action, player_id: int, game_state: GameState) -> bool:
        """验证动作是否合法。"""
        player = game_state.players[player_id]

        # 已弃牌或已出局的玩家不能操作
        if not player.is_active:
            return False

        at = action.action_type

        # 弃牌：始终合法
        if at == ActionType.FOLD:
            return True

        # 看牌：必须尚未看牌
        if at == ActionType.LOOK:
            return not player.has_looked

        # 跟注：筹码足够
        if at == ActionType.CALL:
            call_amount = ActionValidator._call_amount(player, game_state)
            return player.chips >= call_amount

        # 加注：基于底注的倍数，且必须高于当前注
        if ActionType.RAISE_2X <= at <= ActionType.RAISE_6X:
            multiplier = int(at)
            raise_amount = game_state.min_bet * multiplier
            if raise_amount <= game_state.current_bet:
                return False  # 加注不能低于当前注
            pay = raise_amount if player.has_looked else max(1, raise_amount // 2)
            return player.chips >= pay

        # 比牌
        if at == ActionType.COMPARE:
            # 必须指定有效目标
            if action.target < 0 or action.target >= game_state.num_players:
                return False
            if action.target == player_id:
                return False
            target_player = game_state.players[action.target]
            if not target_player.is_active:
                return False
            # 至少经过 min_compare_round 轮
            if game_state.round_count < game_state.min_compare_round:
                return False
            # 筹码足够付比牌费（2 倍当前注）
            compare_cost = game_state.current_bet * 2
            return player.chips >= compare_cost

        return False

    @staticmethod
    def get_valid_actions(
        player_id: int, game_state: GameState
    ) -> List[Action]:
        """获取玩家的所有合法动作。"""
        actions: List[Action] = []
        player = game_state.players[player_id]

        if not player.is_active:
            return actions

        # 弃牌
        actions.append(Action(ActionType.FOLD))

        # 看牌
        if not player.has_looked:
            actions.append(Action(ActionType.LOOK))

        # 跟注
        call_amount = ActionValidator._call_amount(player, game_state)
        if player.chips >= call_amount:
            actions.append(Action(ActionType.CALL))

        # 加注（2x-6x，基于底注倍数）
        for mult in range(2, 7):
            at = ActionType(mult)
            raise_amount = game_state.min_bet * mult
            if raise_amount <= game_state.current_bet:
                continue  # 加注不能低于当前注
            pay = raise_amount if player.has_looked else max(1, raise_amount // 2)
            if player.chips >= pay:
                actions.append(Action(at))

        # 比牌
        if game_state.round_count >= game_state.min_compare_round:
            compare_cost = game_state.current_bet * 2
            if player.chips >= compare_cost:
                for target_id in range(game_state.num_players):
                    if target_id != player_id and game_state.players[target_id].is_active:
                        actions.append(Action(ActionType.COMPARE, target=target_id))

        return actions

    @staticmethod
    def _call_amount(player, game_state: GameState) -> int:
        """计算玩家跟注需要支付的金额。

        暗牌玩家付明牌价的一半。
        """
        if player.has_looked:
            return game_state.current_bet
        else:
            return max(1, game_state.current_bet // 2)
