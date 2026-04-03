"""回放查询端点。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from server.schemas import ReplayResponse

router = APIRouter(prefix="/replays", tags=["replays"])

_replay_store = None


def set_replay_store(store) -> None:
    global _replay_store
    _replay_store = store


@router.get("")
async def list_replays(skip: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):
    """获取回放列表（分页）。"""
    return _replay_store.list_replays(skip=skip, limit=limit)


@router.get("/{replay_id}")
async def get_replay(replay_id: str):
    """获取单局回放详情。"""
    replay = _replay_store.get(replay_id)
    if not replay:
        raise HTTPException(status_code=404, detail="Replay not found")
    return replay
