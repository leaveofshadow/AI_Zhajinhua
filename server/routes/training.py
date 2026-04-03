"""训练控制端点。"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional

from fastapi import APIRouter

from server.schemas import TrainingStartRequest, TrainingStatus

router = APIRouter(prefix="/training", tags=["training"])

# 训练状态
_training_thread: Optional[threading.Thread] = None
_training_status = TrainingStatus(is_running=False)


@router.post("/start")
async def start_training(req: TrainingStartRequest):
    """启动训练任务（后台线程）。"""
    global _training_thread, _training_status

    if _training_status.is_running:
        return {"status": "already_running"}

    _training_status = TrainingStatus(
        is_running=True,
        current_episode=0,
        total_episodes=req.num_episodes,
    )

    def _train():
        global _training_status
        try:
            from ai.train import train, parse_args
            import sys

            # 构造命令行参数
            old_argv = sys.argv
            sys.argv = [
                "train",
                "--episodes", str(req.num_episodes),
                "--num-players", str(req.num_players),
                "--lr", str(req.learning_rate),
                "--device", req.device,
            ]
            train(parse_args())
            sys.argv = old_argv
        except Exception as e:
            print(f"Training error: {e}")
        finally:
            _training_status.is_running = False

    _training_thread = threading.Thread(target=_train, daemon=True)
    _training_thread.start()

    return {"status": "started"}


@router.post("/stop")
async def stop_training():
    """停止训练。"""
    global _training_status
    _training_status.is_running = False
    return {"status": "stopping"}


@router.get("/status", response_model=TrainingStatus)
async def get_training_status():
    """获取训练状态。"""
    return _training_status
