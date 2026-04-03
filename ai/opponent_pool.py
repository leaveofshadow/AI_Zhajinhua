"""历史策略池：防止策略退化。"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .model import ZhaJinHuaNet

__all__ = ["OpponentPool", "PoolEntry"]


@dataclass(slots=True)
class PoolEntry:
    """策略池中的一个历史版本。"""
    model_state: dict
    elo: float = 1000.0
    version: int = 0
    win_rate: float = 0.0
    path: str = ""


class OpponentPool:
    """历史策略池管理器。

    维护 N 个历史版本的策略，新策略必须战胜所有历史版本才能升级。
    防止策略循环克制问题。
    """

    def __init__(self, max_size: int = 10, save_dir: str = "data/models/opponents") -> None:
        self.max_size = max_size
        self.save_dir = save_dir
        self.entries: List[PoolEntry] = []
        os.makedirs(save_dir, exist_ok=True)

    def add(self, model: ZhaJinHuaNet, elo: float = 1000.0, win_rate: float = 0.0) -> None:
        """添加新版本到策略池。"""
        version = len(self.entries)
        path = os.path.join(self.save_dir, f"opponent_v{version}.pt")
        torch.save(model.state_dict(), path)

        entry = PoolEntry(
            model_state=model.state_dict(),
            elo=elo,
            version=version,
            win_rate=win_rate,
            path=path,
        )
        self.entries.append(entry)

        # 超过最大容量时移除最旧的
        if len(self.entries) > self.max_size:
            removed = self.entries.pop(0)
            if removed.path and os.path.exists(removed.path):
                os.remove(removed.path)

    def sample(self) -> Optional[ZhaJinHuaNet]:
        """随机采样一个历史版本。"""
        if not self.entries:
            return None
        entry = random.choice(self.entries)
        model = ZhaJinHuaNet()
        model.load_state_dict(entry.model_state)
        return model

    def sample_state_dict(self) -> Optional[dict]:
        """随机采样一个历史版本的 state_dict。"""
        if not self.entries:
            return None
        entry = random.choice(self.entries)
        return entry.model_state

    def get_all(self) -> List[PoolEntry]:
        """获取所有历史版本。"""
        return list(self.entries)

    @property
    def size(self) -> int:
        return len(self.entries)

    def save_metadata(self) -> None:
        """保存池元数据。"""
        import json
        meta = {
            "entries": [
                {"version": e.version, "elo": e.elo, "win_rate": e.win_rate, "path": e.path}
                for e in self.entries
            ]
        }
        path = os.path.join(self.save_dir, "pool_metadata.json")
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)

    def load_metadata(self) -> None:
        """加载池元数据。"""
        import json
        path = os.path.join(self.save_dir, "pool_metadata.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            meta = json.load(f)
        self.entries = []
        for item in meta.get("entries", []):
            if os.path.exists(item["path"]):
                state_dict = torch.load(item["path"], weights_only=True)
                self.entries.append(PoolEntry(
                    model_state=state_dict,
                    elo=item.get("elo", 1000.0),
                    version=item.get("version", 0),
                    win_rate=item.get("win_rate", 0.0),
                    path=item["path"],
                ))
