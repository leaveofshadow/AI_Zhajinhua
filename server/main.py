"""FastAPI 应用入口。"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes import game, replays, rooms, training
from server.services.game_runner import GameRunner
from server.services.replay_store import ReplayStore
from server.services.room_manager import RoomManager

# 确保数据目录存在
for d in ["data/models", "data/replays", "data/logs", "data/models/opponents"]:
    os.makedirs(d, exist_ok=True)

# 初始化服务
room_manager = RoomManager()
replay_store = ReplayStore()
game_runner = GameRunner(replay_store)

# FastAPI 应用
app = FastAPI(
    title="炸金花 AI 博弈系统",
    description="基于强化学习的炸金花卡牌博弈系统 API",
    version="1.0.0",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注入依赖
rooms.set_room_manager(room_manager)
game.set_dependencies(room_manager, game_runner)
replays.set_replay_store(replay_store)

# 注册路由
app.include_router(rooms.router)
app.include_router(game.router)
app.include_router(replays.router)
app.include_router(training.router)


@app.get("/")
async def root():
    return {"name": "炸金花 AI 博弈系统", "version": "1.0.0"}


@app.get("/models")
async def list_models():
    """列出可用的 AI 模型。"""
    models_dir = "data/models"
    results = []
    if os.path.exists(models_dir):
        for f in sorted(os.listdir(models_dir)):
            if f.endswith(".pt"):
                results.append({
                    "name": f[:-3],
                    "path": os.path.join(models_dir, f),
                })
    # 包含对手池模型
    opp_dir = os.path.join(models_dir, "opponents")
    if os.path.exists(opp_dir):
        for f in sorted(os.listdir(opp_dir)):
            if f.endswith(".pt"):
                results.append({
                    "name": f"opponent/{f[:-3]}",
                    "path": os.path.join(opp_dir, f),
                })
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
