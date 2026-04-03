# 炸金花 AI 博弈系统

基于强化学习（自博弈 + PPO）的炸金花卡牌游戏，支持 AI vs AI、AI vs Human、Human vs Human 三种模式。

纯学术研究项目，Python 全栈（FastAPI + Vue 3）。

## 架构

```
┌─────────────────────────────────────────┐
│           Web 前端 (Vue 3 + Vite)        │
│   游戏桌面 | 训练监控 | 策略分析面板       │
└──────────────────┬──────────────────────┘
                   │ WebSocket / REST
┌──────────────────▼──────────────────────┐
│          游戏服务器 (FastAPI)             │
│   对局管理 | 模式切换 | 房间系统 | 回放     │
└──────────┬────────────────┬─────────────┘
           │                │
┌──────────▼──────┐  ┌─────▼──────────────┐
│   游戏引擎       │  │   AI 训练框架        │
│ 牌局|规则|状态    │  │ Self-Play | PPO     │
│ 牌型判断|动作验证  │  │ 策略网络 | 价值网络   │
└─────────────────┘  └────────────────────┘
```

## 快速开始

### 环境要求

- Python 3.11+
- Node.js 18+
- PyTorch 2.0+（AI 训练）

### 启动

```bash
# 安装后端依赖
pip install -r requirements.txt

# 安装前端依赖
cd web && npm install && cd ..

# 启动后端 (端口 8001)
python -m uvicorn server.main:app --host 0.0.0.0 --port 8001 --reload

# 启动前端 (端口 5173)
cd web && npx vite --host 0.0.0.0 --port 5173
```

浏览器访问 `http://localhost:5173`

## 游戏规则

炸金花（三张牌），3-6 人对局：

| 牌型 | 说明 |
|------|------|
| 豹子 | 三条（如 AAA） |
| 同花顺 | 同花色顺子（如 AKQ 同花） |
| 同花 | 三张同花色 |
| 顺子 | 三张连续（含 A-2-3） |
| 对子 | 两张相同 |
| 散牌 | 以上都不是 |

### 操作

- **看牌** — 翻看自己的牌（暗牌 → 明牌）
- **跟注** — 跟当前注（暗牌付一半）
- **加注** — 加注到指定倍数（2x-6x 底注）
- **比牌** — 与某玩家比牌，输者出局（至少 2 轮后可用）
- **弃牌** — 放弃本局

## 项目结构

```
zhajinhua/
├── engine/          # 游戏引擎（纯逻辑，零 IO 依赖）
│   ├── cards.py     # 牌组、洗牌、发牌
│   ├── hand_evaluator.py  # 牌型判断（查表法 O(1)）
│   ├── game.py      # 牌局状态机
│   └── actions.py   # 动作定义与验证
│
├── ai/              # AI 训练与推理
│   ├── model.py     # Actor-Critic 网络
│   ├── features.py  # 观察空间 → 特征向量
│   ├── agent.py     # 推理封装
│   ├── ppo_trainer.py   # PPO 训练循环
│   └── self_play.py # 自博弈并行环境
│
├── server/          # 游戏服务器
│   ├── main.py      # FastAPI 入口
│   ├── routes/      # REST API + WebSocket
│   ├── services/    # 房间管理、对局调度、回放存储
│   └── schemas.py   # 请求/响应模型
│
├── web/             # 前端 (Vue 3 + TypeScript)
│   └── src/
│       ├── views/   # 大厅、游戏桌面、训练监控、策略分析
│       ├── components/  # 座位环、手牌、操作栏
│       ├── stores/  # Pinia 状态管理
│       └── utils/   # WebSocket、API 封装
│
└── data/            # 运行时数据
    ├── models/      # AI 模型 checkpoints
    ├── replays/     # 对局回放 JSON
    └── logs/        # 训练日志
```

## API

### REST

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/rooms` | 创建房间 |
| GET | `/api/rooms` | 房间列表 |
| GET | `/api/rooms/{id}` | 房间状态 |
| GET | `/api/replays` | 回放列表 |
| GET | `/api/models` | 可用 AI 模型 |

### WebSocket

```
ws://host/ws/room/{room_id}/play

# 客户端发送
{"action": "call"}
{"action": "look"}
{"action": "raise", "multiplier": 3}
{"action": "compare", "target": 2}
{"action": "new_game"}

# 服务端推送
{"event": "game_start", "position": 0, "state": {...}}
{"event": "your_turn", "state": {...}}
{"event": "player_action", "player": 1, "action": "call"}
{"event": "round_end", "winner": 0, "all_hands": [...], "chip_changes": [...]}
```

## AI 训练

```bash
# 启动训练
python -m ai.train --episodes 100000 --num-players 4

# 训练监控
# 浏览器访问前端训练监控页面
```

训练方案：自博弈 + PPO，从随机策略开始自我对弈，消费级 GPU（RTX 3060）即可训练。

## License

MIT
