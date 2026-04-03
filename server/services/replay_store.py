"""回放持久化。"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Dict, List, Optional


class ReplayStore:
    """回放存储，保存为 JSON 文件。"""

    def __init__(self, save_dir: str = "data/replays") -> None:
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, replay_data: Dict) -> str:
        """保存回放数据，返回 replay_id。"""
        replay_id = str(uuid.uuid4())[:8]
        replay_data["id"] = replay_id
        if "timestamp" not in replay_data:
            replay_data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        path = os.path.join(self.save_dir, f"{replay_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(replay_data, f, ensure_ascii=False, indent=2)

        return replay_id

    def get(self, replay_id: str) -> Optional[Dict]:
        """获取单条回放。"""
        path = os.path.join(self.save_dir, f"{replay_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_replays(self, skip: int = 0, limit: int = 20) -> List[Dict]:
        """列出回放（按时间倒序）。"""
        files = sorted(
            [f for f in os.listdir(self.save_dir) if f.endswith(".json")],
            reverse=True,
        )
        results = []
        for fname in files[skip : skip + limit]:
            path = os.path.join(self.save_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append({
                    "id": data.get("id", fname[:-5]),
                    "timestamp": data.get("timestamp", ""),
                    "num_players": len(data.get("players", [])),
                })
        return results
