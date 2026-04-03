"""牌局状态机：完整的炸金花牌局流程管理。

支持 3-6 人对局、暗牌/明牌切换、比牌/弃牌/加注/看牌等全部动作，
以及向量化并行环境用于训练时批量模拟。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from .actions import Action, ActionType, ActionValidator
from .cards import Card, Deck
from .hand_evaluator import HandRank, compare, evaluate

__all__ = [
    "GamePhase",
    "PlayerState",
    "GameState",
    "Game",
    "GameResult",
    "ActionRecord",
    "VectorizedGameEnv",
]


class GamePhase(Enum):
    """牌局阶段。"""
    WAITING = auto()
    PLAYING = auto()
    FINISHED = auto()


@dataclass(slots=True)
class PlayerState:
    """玩家运行时状态。"""
    cards: List[Card] = field(default_factory=list)
    chips: int = 0
    is_active: bool = True
    has_looked: bool = False
    total_bet: int = 0       # 本局累计下注
    last_action: str = ""     # 上一个动作名称


@dataclass(slots=True)
class ActionRecord:
    """动作记录，用于回放。"""
    round: int
    player: int
    action: Action
    amount: int = 0           # 本次动作涉及金额
    result: str = ""          # 比牌结果等


@dataclass(slots=True)
class GameResult:
    """一局游戏的结果。"""
    winner: int = -1
    rankings: List[int] = field(default_factory=list)  # 名次列表: rankings[i] = 玩家i的名次(0-based)
    chip_changes: List[int] = field(default_factory=list)
    final_hands: List[Optional[HandRank]] = field(default_factory=list)


class GameState:
    """牌局配置与运行时状态的容器。"""

    __slots__ = (
        "num_players", "initial_chips", "min_bet",
        "min_compare_round", "max_rounds",
        "player_chips", "eliminated",
        "phase", "players", "pot", "current_bet",
        "current_player_idx", "round_count", "action_history",
        "_deck", "_round_action_count", "_starting_player_idx",
    )

    def __init__(
        self,
        num_players: int = 4,
        initial_chips: int = 1000,
        min_bet: int = 10,
        min_compare_round: int = 2,
        max_rounds: int = 50,
        player_chips: Optional[List[int]] = None,
        eliminated: Optional[List[bool]] = None,
    ) -> None:
        if not (3 <= num_players <= 6):
            raise ValueError(f"num_players must be 3-6, got {num_players}")

        self.num_players: int = num_players
        self.initial_chips: int = initial_chips
        self.min_bet: int = min_bet
        self.min_compare_round: int = min_compare_round
        self.max_rounds: int = max_rounds
        self.player_chips: Optional[List[int]] = player_chips
        self.eliminated: Optional[List[bool]] = eliminated

        self.phase: GamePhase = GamePhase.WAITING
        self.players: List[PlayerState] = []
        self.pot: int = 0
        self.current_bet: int = min_bet
        self.current_player_idx: int = 0
        self.round_count: int = 0
        self.action_history: List[ActionRecord] = []
        self._deck: Optional[Deck] = None
        self._round_action_count: int = 0
        self._starting_player_idx: int = 0


class Game:
    """炸金花牌局管理器。

    使用方法:
        game = Game(num_players=4)
        game.start()
        while not game.is_finished():
            actions = game.get_valid_actions(game.state.current_player_idx)
            game.step(actions[0])  # 选择一个合法动作
        result = game.get_result()
    """

    def __init__(
        self,
        num_players: int = 4,
        initial_chips: int = 1000,
        min_bet: int = 10,
        min_compare_round: int = 2,
        max_rounds: int = 50,
        seed: Optional[int] = None,
        player_chips: Optional[List[int]] = None,
        eliminated: Optional[List[bool]] = None,
    ) -> None:
        self.state = GameState(
            num_players=num_players,
            initial_chips=initial_chips,
            min_bet=min_bet,
            min_compare_round=min_compare_round,
            max_rounds=max_rounds,
            player_chips=player_chips,
            eliminated=eliminated,
        )
        self._seed = seed

    def start(self) -> None:
        """发牌并开始牌局。"""
        state = self.state

        # 初始化玩家（支持自定义筹码）
        if state.player_chips:
            state.players = [PlayerState(chips=c) for c in state.player_chips]
        else:
            state.players = [
                PlayerState(chips=state.initial_chips)
                for _ in range(state.num_players)
            ]

        # 标记被淘汰的玩家
        if state.eliminated:
            for i, elim in enumerate(state.eliminated):
                if elim and i < len(state.players):
                    state.players[i].is_active = False

        # 洗牌发牌（只给活跃玩家发牌）
        state._deck = Deck(seed=self._seed)
        for p in state.players:
            if p.is_active:
                p.cards = state._deck.deal(3)

        # 底注：只从活跃玩家收取
        for p in state.players:
            if p.is_active:
                ante = min(state.min_bet, p.chips)
                p.chips -= ante
                p.total_bet += ante
                state.pot += ante

        state.current_bet = state.min_bet
        # 找到第一个活跃玩家作为起始玩家
        first_active = 0
        for i, p in enumerate(state.players):
            if p.is_active:
                first_active = i
                break
        state.current_player_idx = first_active
        state.round_count = 0
        state._round_action_count = 0
        state._starting_player_idx = first_active
        state.action_history = []
        state.phase = GamePhase.PLAYING

    def step(self, action: Action) -> bool:
        """执行一个动作。

        Args:
            action: 玩家动作

        Returns:
            True 表示本局结束
        """
        state = self.state
        if state.phase != GamePhase.PLAYING:
            raise RuntimeError("Game is not in PLAYING phase")

        pid = state.current_player_idx
        player = state.players[pid]

        # 验证动作合法性
        if not ActionValidator.validate(action, pid, state):
            raise ValueError(f"Invalid action {action} for player {pid}")

        amount = 0
        result_str = ""

        at = action.action_type

        if at == ActionType.FOLD:
            player.is_active = False
            player.last_action = "fold"
            result_str = "fold"

        elif at == ActionType.LOOK:
            player.has_looked = True
            player.last_action = "look"
            result_str = "look"

        elif at == ActionType.CALL:
            amount = self._call_amount(pid)
            self._deduct_chips(pid, amount)
            player.last_action = f"call({amount})"

        elif ActionType.RAISE_2X <= at <= ActionType.RAISE_6X:
            multiplier = int(at)
            # 加注基于底注倍数，防止指数暴涨
            new_bet = state.min_bet * multiplier
            if new_bet <= state.current_bet:
                # 加注不能低于当前注，降级为跟注
                amount = self._call_amount(pid)
                self._deduct_chips(pid, amount)
                player.last_action = f"call({amount})"
            else:
                amount = new_bet if player.has_looked else max(1, new_bet // 2)
                self._deduct_chips(pid, amount)
                state.current_bet = new_bet
                player.last_action = f"raise_{multiplier}x({amount})"

        elif at == ActionType.COMPARE:
            target_id = action.target
            compare_cost = state.current_bet * 2
            amount = compare_cost
            self._deduct_chips(pid, amount)

            # 比牌
            attacker_hand = evaluate(player.cards)
            defender_hand = evaluate(state.players[target_id].cards)
            cmp_result = compare(attacker_hand, defender_hand)

            if cmp_result > 0:
                # 发起者赢
                state.players[target_id].is_active = False
                result_str = "win"
            else:
                # 平局或发起者输 → 发起者出局
                player.is_active = False
                result_str = "lose" if cmp_result < 0 else "lose(draw)"

            player.last_action = f"compare({result_str} vs {target_id})"

        # 记录动作
        state.action_history.append(ActionRecord(
            round=state.round_count,
            player=pid,
            action=action,
            amount=amount,
            result=result_str,
        ))

        # 检查是否局结束
        if self._check_finished():
            self._finish_game()
            return True

        # 推进到下一个活跃玩家
        self._advance_player()
        return False

    def get_observation(self, player_id: int) -> Dict[str, Any]:
        """获取玩家视角的不完全信息观察。"""
        state = self.state
        player = state.players[player_id]

        # 私有信息
        my_cards: List[Card] = []
        if player.has_looked:
            my_cards = list(player.cards)

        # 对手公开信息
        player_states = []
        for i, p in enumerate(state.players):
            player_states.append({
                "is_active": p.is_active,
                "is_looked": p.has_looked,
                "total_bet": p.total_bet,
                "last_action": p.last_action,
                "chip_count": p.chips,
            })

        return {
            "my_cards": my_cards,
            "my_chips": player.chips,
            "has_looked": player.has_looked,
            "pot": state.pot,
            "current_bet": state.current_bet,
            "round_count": state.round_count,
            "my_position": player_id,
            "active_players": sum(1 for p in state.players if p.is_active),
            "player_states": player_states,
        }

    def get_valid_actions(self, player_id: int) -> List[Action]:
        """获取指定玩家的所有合法动作。"""
        return ActionValidator.get_valid_actions(player_id, self.state)

    def is_finished(self) -> bool:
        """牌局是否结束。"""
        return self.state.phase == GamePhase.FINISHED

    def get_result(self) -> GameResult:
        """获取牌局结果（仅在 FINISHED 阶段有效）。"""
        state = self.state
        if state.phase != GamePhase.FINISHED:
            raise RuntimeError("Game is not finished yet")

        result = GameResult()

        # 找出赢家
        active = [i for i, p in enumerate(state.players) if p.is_active]
        if len(active) == 1:
            result.winner = active[0]

        # 最终手牌
        for p in state.players:
            if p.cards:
                result.final_hands.append(evaluate(p.cards))
            else:
                result.final_hands.append(None)

        # 筹码变化
        for p in state.players:
            result.chip_changes.append(p.chips - state.initial_chips + p.total_bet)
            # 实际变化 = 当前筹码 - 初始筹码 （底注已扣除）
        result.chip_changes = [p.chips - (state.initial_chips - p.total_bet) for p in state.players]
        # 更正: chip_change = 最终筹码 - 初始筹码
        result.chip_changes = [p.chips - state.initial_chips for p in state.players]

        # 名次（简化：赢家第0名，其余按出局顺序或手牌大小排列）
        result.rankings = self._compute_rankings()

        return result

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _call_amount(self, player_id: int) -> int:
        """计算跟注金额。"""
        player = self.state.players[player_id]
        if player.has_looked:
            return self.state.current_bet
        else:
            return max(1, self.state.current_bet // 2)

    def _deduct_chips(self, player_id: int, amount: int) -> None:
        """从玩家扣除筹码并加入底池。"""
        player = self.state.players[player_id]
        actual = min(amount, player.chips)
        player.chips -= actual
        player.total_bet += actual
        self.state.pot += actual

    def _advance_player(self) -> None:
        """推进到下一个活跃玩家。"""
        state = self.state
        state._round_action_count += 1

        active_count = sum(1 for p in state.players if p.is_active)

        # 如果本轮活跃玩家都已行动过，进入下一轮
        if state._round_action_count >= active_count:
            state._round_action_count = 0
            state.round_count += 1
            # 起始玩家移到下一个活跃玩家
            state._starting_player_idx = self._next_active_player(
                state._starting_player_idx
            )

        # 找下一个活跃玩家
        state.current_player_idx = self._next_active_player(
            state.current_player_idx
        )

    def _next_active_player(self, current: int) -> int:
        """从 current 的下一个位置开始找第一个活跃玩家。"""
        state = self.state
        for i in range(1, state.num_players + 1):
            idx = (current + i) % state.num_players
            if state.players[idx].is_active:
                return idx
        return current

    def _check_finished(self) -> bool:
        """检查牌局是否结束。"""
        state = self.state
        active = sum(1 for p in state.players if p.is_active)
        if active <= 1:
            return True
        if state.round_count >= state.max_rounds:
            return True
        return False

    def _finish_game(self) -> None:
        """结算牌局。"""
        state = self.state
        state.phase = GamePhase.FINISHED

        active = [i for i, p in enumerate(state.players) if p.is_active]

        if len(active) == 1:
            # 唯一幸存者赢得底池
            winner = active[0]
            state.players[winner].chips += state.pot
        elif len(active) > 1:
            # 超过最大轮数，强制比牌决出胜负
            self._force_compare(active)

    def _force_compare(self, active_players: List[int]) -> None:
        """强制比牌：剩余活跃玩家按顺序两两比牌。"""
        state = self.state
        remaining = list(active_players)

        while len(remaining) > 1:
            p1 = remaining[0]
            p2 = remaining[1]

            hand1 = evaluate(state.players[p1].cards)
            hand2 = evaluate(state.players[p2].cards)
            cmp = compare(hand1, hand2)

            if cmp > 0:
                state.players[p2].is_active = False
                remaining.remove(p2)
            elif cmp < 0:
                state.players[p1].is_active = False
                remaining.remove(p1)
            else:
                # 平局：双方都比，p1 先出局（发起者规则延伸）
                state.players[p1].is_active = False
                remaining.remove(p1)

        if remaining:
            state.players[remaining[0]].chips += state.pot

    def _compute_rankings(self) -> List[int]:
        """计算每个玩家的名次。"""
        state = self.state
        # 按筹码变化排序
        player_chip_changes = [
            (i, p.chips - state.initial_chips)
            for i, p in enumerate(state.players)
        ]
        player_chip_changes.sort(key=lambda x: x[1], reverse=True)

        rankings = [0] * state.num_players
        for rank, (pid, _) in enumerate(player_chip_changes):
            rankings[pid] = rank
        return rankings


class VectorizedGameEnv:
    """并行运行 N 个 Game 实例，支持批量操作。

    用于训练时高效生成经验数据。
    """

    def __init__(
        self,
        num_envs: int,
        num_players: int = 4,
        initial_chips: int = 1000,
        min_bet: int = 10,
        min_compare_round: int = 2,
        max_rounds: int = 50,
    ) -> None:
        self.num_envs = num_envs
        self.num_players = num_players
        self._configs = {
            "initial_chips": initial_chips,
            "min_bet": min_bet,
            "min_compare_round": min_compare_round,
            "max_rounds": max_rounds,
        }
        self._games: List[Game] = []

    def reset_all(self) -> List[Dict[str, Any]]:
        """重置所有环境并开始新一局。返回所有环境的观察。"""
        self._games = [
            Game(num_players=self.num_players, **self._configs)
            for _ in range(self.num_envs)
        ]
        for g in self._games:
            g.start()
        return self.get_all_observations()

    def reset_envs(self, env_ids: List[int]) -> List[Dict[str, Any]]:
        """重置指定环境（已完成的环境）。"""
        observations = []
        for eid in env_ids:
            g = Game(num_players=self.num_players, **self._configs)
            g.start()
            self._games[eid] = g
            observations.append(g.get_observation(0))
        return observations

    def step_all(
        self, actions: List[Optional[Action]]
    ) -> List[Dict[str, Any]]:
        """对所有环境执行动作。

        actions[i] 为 None 表示该环境已结束，跳过。
        返回所有环境的观察。
        """
        results = []
        for i, action in enumerate(actions):
            if action is None or self._games[i].is_finished():
                results.append(None)
                continue
            self._games[i].step(action)
            results.append(self._games[i].get_observation(
                self._games[i].state.current_player_idx
            ))
        return results

    def get_all_observations(self) -> List[Dict[str, Any]]:
        """获取所有环境的当前观察。"""
        return [
            g.get_observation(g.state.current_player_idx)
            if not g.is_finished() else None
            for g in self._games
        ]

    def get_all_valid_actions(self) -> List[List[Action]]:
        """获取所有环境的合法动作。"""
        actions = []
        for g in self._games:
            if g.is_finished():
                actions.append([])
            else:
                actions.append(g.get_valid_actions(g.state.current_player_idx))
        return actions

    def get_finished_env_ids(self) -> List[int]:
        """获取所有已完成环境的 ID。"""
        return [i for i, g in enumerate(self._games) if g.is_finished()]

    def get_all_results(self) -> List[Optional[GameResult]]:
        """获取所有环境的结果。"""
        return [
            g.get_result() if g.is_finished() else None
            for g in self._games
        ]

    def run_random_game(self) -> GameResult:
        """运行一局随机策略游戏（用于快速测试和基准测试）。"""
        game = Game(num_players=self.num_players, **self._configs)
        game.start()

        max_steps = 1000
        for _ in range(max_steps):
            if game.is_finished():
                break
            pid = game.state.current_player_idx
            valid = game.get_valid_actions(pid)
            if not valid:
                break
            import random as _rng
            action = _rng.choice(valid)
            game.step(action)

        return game.get_result()

    def benchmark(self, num_games: int = 10000) -> float:
        """运行性能基准测试，返回每秒可模拟的局数。"""
        import time as _time
        start = _time.perf_counter()
        for _ in range(num_games):
            self.run_random_game()
        elapsed = _time.perf_counter() - start
        return num_games / elapsed
