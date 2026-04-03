"""对局调度器：统一处理 AI 和 Human 混合对局。"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any, Dict, List, Optional

from engine.actions import Action, ActionType
from engine.game import Game, GamePhase
from server.schemas import SeatConfig
from server.services.replay_store import ReplayStore

logger = logging.getLogger(__name__)


class GameRunner:
    """对局调度器。"""

    def __init__(self, replay_store: ReplayStore) -> None:
        self.replay_store = replay_store
        self._action_records: Dict[str, List[Dict]] = {}  # room_id -> records
        self._agents: Dict[str, Any] = {}  # model_path -> Agent

    def start_game(
        self,
        room_id: str,
        num_players: int,
        initial_chips: int,
        min_bet: int,
        seats: List[SeatConfig],
        player_chips: Optional[List[int]] = None,
        eliminated: Optional[List[bool]] = None,
    ) -> Game:
        """创建并开始一局游戏。"""
        game = Game(
            num_players=num_players,
            initial_chips=initial_chips,
            min_bet=min_bet,
            player_chips=player_chips,
            eliminated=eliminated,
        )
        game.start()
        self._action_records[room_id] = []
        return game

    async def handle_ai_turn(
        self,
        game: Game,
        player_id: int,
        seats: List[SeatConfig],
    ) -> Optional[Action]:
        """处理 AI 回合。

        如果座位指定了模型路径且文件存在，使用训练好的 Agent；
        否则使用加权随机策略。
        """
        valid_actions = game.get_valid_actions(player_id)
        if not valid_actions:
            return None

        # 尝试使用训练好的模型
        seat = seats[player_id] if player_id < len(seats) else None
        if seat and seat.ai_model:
            agent = self._get_agent(seat.ai_model)
            if agent is not None:
                obs = game.get_observation(player_id)
                action = agent.act(obs, valid_actions)
                return action

        # 回退到加权随机策略
        return await self._random_ai_turn(game, player_id, valid_actions)

    def _get_agent(self, model_path: str) -> Any:
        """获取或创建 AI Agent（带缓存）。"""
        if model_path in self._agents:
            return self._agents[model_path]

        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return None

        try:
            from ai.agent import Agent as AIAgent
            agent = AIAgent(model_path=model_path, epsilon=0.0)
            self._agents[model_path] = agent
            logger.info(f"Loaded AI model: {model_path}")
            return agent
        except Exception as e:
            logger.error(f"Failed to load AI model {model_path}: {e}")
            return None

    async def _random_ai_turn(
        self,
        game: Game,
        player_id: int,
        valid_actions: List[Action],
    ) -> Optional[Action]:
        """加权随机策略（回退方案）。"""
        # 优先看牌（如果还没看）
        look_actions = [a for a in valid_actions if a.action_type == ActionType.LOOK]
        if look_actions and random.random() < 0.3:
            return look_actions[0]

        # 加权随机选择
        weights: list[float] = []
        for a in valid_actions:
            if a.action_type == ActionType.FOLD:
                weights.append(1.0)
            elif a.action_type == ActionType.CALL:
                weights.append(5.0)
            elif a.action_type == ActionType.LOOK:
                weights.append(0.0)
            elif ActionType.RAISE_2X <= a.action_type <= ActionType.RAISE_6X:
                mult = int(a.action_type)
                weights.append(max(0.5, 3.0 - mult * 0.5))
            elif a.action_type == ActionType.COMPARE:
                rc = game.state.round_count
                weights.append(min(4.0, rc * 1.0))
            else:
                weights.append(1.0)

        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0.0
        for a, w in zip(valid_actions, weights):
            cumulative += w
            if r <= cumulative:
                return a
        return valid_actions[-1]

    async def wait_human_action(
        self,
        game: Game,
        player_id: int,
        connections: Dict[int, Any],
        timeout: float = 30.0,
    ) -> Optional[Action]:
        """等待人类玩家操作。"""
        ws = connections.get(player_id)
        if ws is None:
            return Action(ActionType.FOLD)  # 无连接则弃牌

        try:
            data = await asyncio.wait_for(ws.receive_json(), timeout=timeout)
            return self._parse_action(data, player_id, game)
        except (asyncio.TimeoutError, Exception):
            return Action(ActionType.FOLD)  # 超时弃牌

    def _parse_action(self, data: Dict, player_id: int, game: Game) -> Action:
        """解析客户端发来的动作。"""
        action_str = data.get("action", "fold")

        if action_str == "fold":
            return Action(ActionType.FOLD)
        elif action_str == "call":
            return Action(ActionType.CALL)
        elif action_str == "look":
            return Action(ActionType.LOOK)
        elif action_str == "raise":
            mult = data.get("multiplier", 2)
            mult = max(2, min(6, mult))
            return Action(ActionType(mult))
        elif action_str == "compare":
            target = data.get("target", -1)
            return Action(ActionType.COMPARE, target=target)
        else:
            return Action(ActionType.FOLD)

    def record_action(self, room_id: str, record: Dict) -> None:
        """记录动作。"""
        if room_id not in self._action_records:
            self._action_records[room_id] = []
        self._action_records[room_id].append(record)

    def save_replay(
        self,
        room_id: str,
        game: Game,
        seats: List[SeatConfig],
        initial_chips: int,
        min_bet: int,
    ) -> str:
        """保存对局回放。自动从 game.action_history 提取动作记录。"""
        result = game.get_result()

        # 优先使用手动记录，否则从 game history 自动提取
        recorded = self._action_records.get(room_id, [])
        if not recorded and game.state.action_history:
            for rec in game.state.action_history:
                recorded.append({
                    "round": rec.round,
                    "player": rec.player,
                    "action": rec.action.action_type.name.lower(),
                    "amount": rec.amount,
                })

        replay_data = {
            "config": {
                "num_players": game.state.num_players,
                "initial_chips": initial_chips,
                "min_bet": min_bet,
            },
            "players": [
                {
                    "seat": i,
                    "type": seats[i].player_type if i < len(seats) else "ai",
                    "name": seats[i].display_name if i < len(seats) else f"P{i}",
                    "model": seats[i].ai_model if i < len(seats) else None,
                }
                for i in range(game.state.num_players)
            ],
            "actions": recorded,
            "result": {
                "winner": result.winner,
                "chip_changes": result.chip_changes,
                "rankings": result.rankings,
                "final_hands": [
                    str(h) if h else None for h in result.final_hands
                ],
            },
        }

        replay_id = self.replay_store.save(replay_data)
        self._action_records.pop(room_id, None)
        return replay_id
