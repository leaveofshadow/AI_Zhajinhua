"""房间生命周期管理。"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from engine.game import Game, GamePhase
from server.schemas import RoomCreate, SeatConfig, RoomResponse


class Room:
    """房间实例。"""

    def __init__(self, config: RoomCreate) -> None:
        self.id: str = str(uuid.uuid4())[:8]
        self.num_players = config.num_players
        self.initial_chips = config.initial_chips
        self.min_bet = config.min_bet
        self.speed = config.speed

        # 初始化座位
        if config.seats and len(config.seats) == config.num_players:
            self.seats: List[SeatConfig] = config.seats
        else:
            self.seats = [
                SeatConfig(player_type="ai", display_name=f"AI-{i}")
                for i in range(config.num_players)
            ]

        self.game: Optional[Game] = None
        self.connections: Dict[int, object] = {}  # player_id -> WebSocket

        # 多回合对局状态
        self.player_chips: List[int] = [self.initial_chips] * self.num_players
        self.eliminated: List[bool] = [False] * self.num_players
        self.session_round: int = 0
        self.session_over: bool = False

    @property
    def phase(self) -> str:
        if self.game is None:
            return "waiting"
        return self.game.state.phase.name.lower()

    @property
    def active_players(self) -> List[int]:
        """返回未淘汰的玩家索引列表。"""
        return [i for i in range(self.num_players) if not self.eliminated[i]]

    def reset_session(self) -> None:
        """重置对局，筹码重新分配。"""
        self.player_chips = [self.initial_chips] * self.num_players
        self.eliminated = [False] * self.num_players
        self.session_round = 0
        self.session_over = False
        self.game = None

    def to_response(self) -> RoomResponse:
        return RoomResponse(
            id=self.id,
            num_players=self.num_players,
            initial_chips=self.initial_chips,
            min_bet=self.min_bet,
            speed=self.speed,
            seats=self.seats,
            phase=self.phase,
            pot=self.game.state.pot if self.game else 0,
            current_player=self.game.state.current_player_idx if self.game and not self.game.is_finished() else -1,
        )


class RoomManager:
    """房间管理器。"""

    def __init__(self) -> None:
        self.rooms: Dict[str, Room] = {}

    def create_room(self, config: RoomCreate) -> Room:
        room = Room(config)
        self.rooms[room.id] = room
        return room

    def get_room(self, room_id: str) -> Optional[Room]:
        return self.rooms.get(room_id)

    def close_room(self, room_id: str) -> bool:
        if room_id in self.rooms:
            del self.rooms[room_id]
            return True
        return False

    def update_seats(self, room_id: str, seats: List[SeatConfig]) -> tuple:
        """返回 (room, error_msg)。
        room 非 None 表示成功；error_msg 非 None 表示错误原因。
        """
        room = self.rooms.get(room_id)
        if room is None:
            return None, "Room not found"
        if room.phase != "waiting":
            return None, "Game already started"
        if len(seats) != room.num_players:
            return None, f"Expected {room.num_players} seats, got {len(seats)}"
        room.seats = seats
        return room, None

    def list_rooms(self) -> List[Room]:
        return list(self.rooms.values())
