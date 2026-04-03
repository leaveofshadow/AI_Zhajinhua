"""Pydantic 请求/响应模型。"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class SeatConfig(BaseModel):
    """座位配置。"""
    player_type: Literal["ai", "human"] = "ai"
    ai_model: Optional[str] = None
    ai_level: Optional[Literal["easy", "medium"]] = None
    display_name: str = ""


class RoomCreate(BaseModel):
    """创建房间请求。"""
    num_players: int = Field(default=4, ge=3, le=6)
    initial_chips: int = Field(default=1000, ge=100)
    min_bet: int = Field(default=10, ge=1)
    speed: Literal["normal", "fast", "turbo"] = "normal"
    seats: Optional[List[SeatConfig]] = None


class RoomResponse(BaseModel):
    """房间信息响应。"""
    id: str
    num_players: int
    initial_chips: int
    min_bet: int
    speed: str
    seats: List[SeatConfig]
    phase: str = "waiting"
    pot: int = 0
    current_player: int = -1


class ActionRequest(BaseModel):
    """玩家动作请求。"""
    action: str
    multiplier: Optional[int] = None
    target: Optional[int] = None


class ReplayResponse(BaseModel):
    """回放信息响应。"""
    id: str
    timestamp: str
    config: Dict
    players: List[Dict]
    actions: List[Dict]
    result: Dict


class ModelInfo(BaseModel):
    """AI 模型信息。"""
    name: str
    path: str
    elo: Optional[float] = None
    version: Optional[int] = None


class TrainingStartRequest(BaseModel):
    """启动训练请求。"""
    num_episodes: int = 50000
    num_players: int = 4
    learning_rate: float = 3e-4
    device: str = "auto"


class TrainingStatus(BaseModel):
    """训练状态响应。"""
    is_running: bool
    current_episode: int = 0
    total_episodes: int = 0
    elapsed_seconds: float = 0.0
    metrics: Dict = {}


class ErrorResponse(BaseModel):
    """错误响应。"""
    error: str
    detail: Optional[str] = None
