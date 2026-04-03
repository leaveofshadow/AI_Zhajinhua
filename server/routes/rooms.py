"""房间 CRUD 端点。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from server.schemas import RoomCreate, RoomResponse, SeatConfig

router = APIRouter(prefix="/rooms", tags=["rooms"])

# 全局 RoomManager 实例（由 main.py 注入）
_room_manager = None


def set_room_manager(manager) -> None:
    global _room_manager
    _room_manager = manager


@router.post("", response_model=RoomResponse)
async def create_room(config: RoomCreate):
    """创建房间。"""
    room = _room_manager.create_room(config)
    return room.to_response()


@router.get("", response_model=list[RoomResponse])
async def list_rooms():
    """列出所有房间。"""
    return [r.to_response() for r in _room_manager.list_rooms()]


@router.get("/{room_id}", response_model=RoomResponse)
async def get_room(room_id: str):
    """查询房间状态。"""
    room = _room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room.to_response()


@router.patch("/{room_id}/seats", response_model=RoomResponse)
async def update_seats(room_id: str, seats: list[SeatConfig]):
    """修改座位配置。"""
    room, error = _room_manager.update_seats(room_id, seats)
    if error:
        status = 400 if "Expected" in error else 404
        raise HTTPException(status_code=status, detail=error)
    return room.to_response()


@router.delete("/{room_id}")
async def close_room(room_id: str):
    """关闭房间。"""
    if not _room_manager.close_room(room_id):
        raise HTTPException(status_code=404, detail="Room not found")
    return {"status": "closed"}
