"""WebSocket 对局端点。"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from engine.actions import ActionType

router = APIRouter(tags=["game"])

# 全局依赖（由 main.py 注入）
_room_manager = None
_game_runner = None
_active_games: Dict[str, set] = {}  # room_id -> set of ws connections


def set_dependencies(room_manager, game_runner) -> None:
    global _room_manager, _game_runner
    _room_manager = room_manager
    _game_runner = game_runner


@router.websocket("/ws/room/{room_id}/play")
async def websocket_game(websocket: WebSocket, room_id: str):
    """WebSocket 对局通道。"""
    await websocket.accept()

    room = _room_manager.get_room(room_id)
    if not room:
        await websocket.send_json({"event": "error", "message": "Room not found"})
        await websocket.close()
        return

    # 找到该玩家对应的座位号
    player_id = await _find_available_seat(room, websocket)

    if player_id is None:
        await websocket.send_json({"event": "error", "message": "No available seat"})
        await websocket.close()
        return

    room.connections[player_id] = websocket

    try:
        # 如果还没开始游戏且人够了，自动开始
        if room.game is None:
            await _maybe_start_game(room_id)

        # 游戏开始后通知所有连接的玩家
        if room.game and not room.game.is_finished():
            for pid, ws in room.connections.items():
                obs = room.game.get_observation(pid)
                try:
                    await ws.send_json({
                        "event": "game_start",
                        "position": pid,
                        "state": _serialize_observation(obs, pid, room.seats),
                        "current_player": room.game.state.current_player_idx,
                    })
                except Exception:
                    pass

            # 处理 AI 回合并通知人类玩家
            await _process_ai_turns(room_id)
            if room.game.is_finished():
                await _handle_game_end(room_id)
            else:
                await _notify_current_human(room_id)

        # 主循环
        while True:
            data = await websocket.receive_json()

            # 游戏结束后，支持 new_game / new_session
            if room.game is None or room.game.is_finished():
                if data.get("action") == "new_session":
                    room.reset_session()
                    await _maybe_start_game(room_id)
                    if room.game and not room.game.is_finished():
                        for pid, ws in room.connections.items():
                            obs = room.game.get_observation(pid)
                            try:
                                await ws.send_json({
                                    "event": "game_start",
                                    "position": pid,
                                    "state": _serialize_observation(obs, pid, room.seats),
                                    "current_player": room.game.state.current_player_idx,
                                })
                            except Exception:
                                pass
                        await _process_ai_turns(room_id)
                        if room.game.is_finished():
                            await _handle_game_end(room_id)
                        else:
                            await _notify_current_human(room_id)
                elif data.get("action") == "new_game":
                    await _start_new_game(room_id, websocket, player_id)
                continue

            action = _game_runner._parse_action(data, player_id, room.game)

            # 验证是否轮到该玩家
            if room.game.state.current_player_idx != player_id:
                await websocket.send_json({"event": "error", "message": "Not your turn"})
                continue

            # 验证动作合法性
            from engine.actions import ActionValidator
            if not ActionValidator.validate(action, player_id, room.game.state):
                await websocket.send_json({"event": "error", "message": f"Invalid action: {data.get('action')}"})
                continue

            # 执行动作
            _game_runner.record_action(room_id, {
                "round": room.game.state.round_count,
                "player": player_id,
                "action": data.get("action"),
                "multiplier": data.get("multiplier"),
                "target": data.get("target"),
            })

            room.game.step(action)

            # 广播状态更新
            await _broadcast_state(room_id, player_id, data)

            # 处理后续 AI 回合
            if not room.game.is_finished():
                await _process_ai_turns(room_id)

            # 检查游戏结束
            if room.game.is_finished():
                await _handle_game_end(room_id)
                continue  # 不跳出循环，等待 new_game

            # 通知下一个人类玩家
            await _notify_current_human(room_id)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Game WS error: {e}")
    finally:
        room.connections.pop(player_id, None)


async def _start_new_game(room_id: str, websocket, player_id: int) -> None:
    """开始新一局。"""
    room = _room_manager.get_room(room_id)
    if not room:
        return

    # 对局已结束，不能再开新回合
    if room.session_over:
        await websocket.send_json({
            "event": "error",
            "message": "Session is over. Start a new session.",
        })
        return

    # 递增回合数
    room.session_round += 1

    # 使用上一回合结束时的筹码和淘汰状态
    room.game = _game_runner.start_game(
        room_id=room_id,
        num_players=room.num_players,
        initial_chips=room.initial_chips,
        min_bet=room.min_bet,
        seats=room.seats,
        player_chips=room.player_chips,
        eliminated=room.eliminated,
    )

    # 通知所有玩家
    if room.game and not room.game.is_finished():
        for pid, ws in room.connections.items():
            obs = room.game.get_observation(pid)
            try:
                await ws.send_json({
                    "event": "game_start",
                    "position": pid,
                    "state": _serialize_observation(obs, pid, room.seats),
                    "current_player": room.game.state.current_player_idx,
                })
            except Exception:
                pass

        # 处理 AI 回合
        await _process_ai_turns(room_id)
        if room.game.is_finished():
            await _handle_game_end(room_id)
        else:
            await _notify_current_human(room_id)


async def _notify_current_human(room_id: str) -> None:
    """通知当前回合的人类玩家。"""
    room = _room_manager.get_room(room_id)
    if not room or not room.game or room.game.is_finished():
        return

    pid = room.game.state.current_player_idx
    if pid < len(room.seats) and room.seats[pid].player_type == "human":
        ws = room.connections.get(pid)
        if ws:
            obs = room.game.get_observation(pid)
            await ws.send_json({
                "event": "your_turn",
                "state": _serialize_observation(obs, pid, room.seats),
                "current_player": pid,
            })


async def _find_available_seat(room, websocket) -> int | None:
    """为 WebSocket 连接找到可用的人类座位。"""
    for i, seat in enumerate(room.seats):
        if seat.player_type == "human" and i not in room.connections:
            return i
    return None


async def _maybe_start_game(room_id: str) -> None:
    """检查是否可以开始游戏。"""
    room = _room_manager.get_room(room_id)
    if room is None:
        return

    # 检查人类玩家是否都已连接
    for i, seat in enumerate(room.seats):
        if seat.player_type == "human" and i not in room.connections:
            return  # 还有人在等待连接

    # 首局初始化 session
    room.session_round = 1

    # 开始游戏
    room.game = _game_runner.start_game(
        room_id=room_id,
        num_players=room.num_players,
        initial_chips=room.initial_chips,
        min_bet=room.min_bet,
        seats=room.seats,
    )


async def _process_ai_turns(room_id: str) -> None:
    """处理连续的 AI 回合。"""
    room = _room_manager.get_room(room_id)
    if not room or not room.game or room.game.is_finished():
        return

    max_ai_steps = 200  # 防止无限循环（max_rounds=50 × 4 玩家）
    for _ in range(max_ai_steps):
        if room.game.is_finished():
            break

        pid = room.game.state.current_player_idx
        seat = room.seats[pid] if pid < len(room.seats) else None

        if seat and seat.player_type == "ai":
            action = await _game_runner.handle_ai_turn(room.game, pid, room.seats)
            if action:
                _game_runner.record_action(room_id, {
                    "round": room.game.state.round_count,
                    "player": pid,
                    "action": str(action.action_type.name).lower(),
                })
                room.game.step(action)
            else:
                break
        else:
            break  # 轮到人类玩家


async def _broadcast_state(room_id: str, actor_id: int, action_data: dict) -> None:
    """广播状态更新。"""
    room = _room_manager.get_room(room_id)
    if not room or not room.game:
        return

    event = {
        "event": "player_action",
        "player": actor_id,
        "action": action_data.get("action"),
        "pot": room.game.state.pot,
        "current_player": room.game.state.current_player_idx,
    }

    for pid, ws in room.connections.items():
        try:
            await ws.send_json(event)
        except Exception:
            pass


async def _handle_game_end(room_id: str) -> None:
    """处理游戏结束。"""
    room = _room_manager.get_room(room_id)
    if not room or not room.game:
        return

    result = room.game.get_result()

    # 保存回放
    replay_id = _game_runner.save_replay(
        room_id, room.game, room.seats, room.initial_chips, room.min_bet
    )

    # 收集所有玩家的牌和牌型
    all_hands = []
    from engine.hand_evaluator import evaluate, HandType
    _HAND_TYPE_NAMES = {
        HandType.HIGH_CARD: "散牌",
        HandType.PAIR: "对子",
        HandType.STRAIGHT: "顺子",
        HandType.FLUSH: "同花",
        HandType.STRAIGHT_FLUSH: "同花顺",
        HandType.THREE_OF_A_KIND: "豹子",
    }
    for i, p in enumerate(room.game.state.players):
        hand_rank = evaluate(p.cards) if p.cards else None
        hand_info = {
            "name": room.seats[i].display_name if i < len(room.seats) else f"P{i}",
            "player_type": room.seats[i].player_type if i < len(room.seats) else "ai",
            "cards": [str(c) for c in p.cards] if p.cards else [],
            "hand_type": _HAND_TYPE_NAMES.get(hand_rank.hand_type, "") if hand_rank else "",
            "is_active": p.is_active,
            "total_bet": p.total_bet,
            "chips": p.chips,
        }
        all_hands.append(hand_info)

    end_event = {
        "event": "round_end",
        "winner": result.winner,
        "chip_changes": result.chip_changes,
        "all_hands": all_hands,
        "replay_id": replay_id,
        "session_round": room.session_round,
        "player_chips": room.player_chips,
        "eliminated": room.eliminated,
        "session_over": room.session_over,
    }

    for pid, ws in room.connections.items():
        try:
            await ws.send_json(end_event)
        except Exception:
            pass

    # 更新 room 的累计筹码和淘汰状态
    for i, p in enumerate(room.game.state.players):
        room.player_chips[i] = p.chips
        if p.chips <= 0:
            room.eliminated[i] = True

    # 检查是否只剩一个玩家有筹码
    active_count = sum(1 for e in room.eliminated if not e)
    if active_count <= 1:
        room.session_over = True

    # 不立即清除 game，等下一局开始时重置
    room.game = None


def _serialize_observation(obs: dict, player_id: int, seats=None) -> dict:
    """序列化观察数据。"""
    player_states = obs.get("player_states", [])
    if seats:
        for i, ps in enumerate(player_states):
            if i < len(seats):
                ps["name"] = seats[i].display_name
                ps["player_type"] = seats[i].player_type
    return {
        "my_cards": [str(c) for c in obs.get("my_cards", [])],
        "my_chips": obs.get("my_chips", 0),
        "has_looked": obs.get("has_looked", False),
        "pot": obs.get("pot", 0),
        "current_bet": obs.get("current_bet", 0),
        "round_count": obs.get("round_count", 0),
        "my_position": obs.get("my_position", player_id),
        "active_players": obs.get("active_players", 0),
        "player_states": player_states,
    }
