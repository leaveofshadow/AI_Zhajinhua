# 炸金花 AI 博弈系统 - 项目设计文档

## 项目概述

纯学术研究项目。开发一个基于强化学习（自博弈 + PPO）的炸金花卡牌游戏，支持 AI vs AI、AI vs Human、Human vs Human 三种模式自由切换。Web 应用形式，Python 全栈。

### 核心约束

- 训练硬件：消费级 GPU（RTX 3060 级别）
- 无先验数据，纯自博弈从零学习
- 3-6 人标准炸金花规则
- 研究导向：完整数据记录、策略分析能力

---

## 1. 整体架构

系统分为 4 个独立层，通过清晰接口通信：

```
┌─────────────────────────────────────────────┐
│              Web 前端 (Vue 3)                 │
│   游戏桌面 | 训练监控 | 策略分析面板            │
└──────────────────┬──────────────────────────┘
                   │ WebSocket / REST
┌──────────────────▼──────────────────────────┐
│           游戏服务器 (FastAPI)                │
│   对局管理 | 模式切换 | 房间系统 | 回放          │
└──────────┬────────────────┬─────────────────┘
           │                │
┌──────────▼──────┐  ┌─────▼──────────────────┐
│   游戏引擎       │  │    AI 训练框架           │
│ 牌局|规则|状态    │  │ Self-Play|PPO|回放缓冲  │
│ 牌型判断|动作验证  │  │ 策略网络|价值网络|日志    │
└─────────────────┘  └────────────────────────┘
```

### 关键设计决策

- **游戏引擎纯逻辑，无 IO 依赖** — 训练时每秒可跑数千局，Web 展示时走正常速度
- **AI 推理与训练解耦** — 训练产出的模型文件直接加载到游戏服务器做推理
- **模式切换零成本** — 每个座位独立标记为 AI/Human，同一套对局流程
- **研究导向** — 所有对局数据（状态、动作、奖励）完整记录，供事后分析

---

## 2. 游戏引擎

### 牌型体系

```
牌型排名（从大到小）：
豹子(AAA) > 同花顺(AKQ 同花) > 同花 > 顺子 > 对子 > 单张

52 张牌，每局每人发 3 张，C(52,3) = 22,100 种手牌组合
```

### 状态空间（玩家观察）

```python
observation = {
    "my_cards": [3张牌],           # 自己的牌
    "my_chips": int,               # 自己筹码
    "pot": int,                    # 底池
    "current_bet": int,            # 当前跟注额
    "player_states": [             # 每个对手的公开信息
        {
            "is_active": bool,     # 是否还在局
            "is_looked": bool,     # 是否已看牌
            "total_bet": int,      # 本局累计下注
            "last_action": str,    # 上一个动作
        },
        ...
    ],
    "round_count": int,            # 当前轮数
    "my_position": int,            # 自己的座位号
}
```

### 动作空间

| 动作         | 说明               | 编码                |
| ------------ | ------------------ | ------------------- |
| 弃牌 (fold)  | 放弃本局           | 0                   |
| 跟注 (call)  | 跟当前注           | 1                   |
| 加注 (raise) | 加注到指定倍数     | 2-6（对应 2x ~ 6x） |
| 看牌 (look)  | 翻看自己的牌       | 7                   |
| 比牌 (compare)| 选择与某人比牌    | 8 + target_player_index |

### 核心规则

- 暗牌玩家跟注额 = 明牌玩家的一半
- 比牌：发起者需付出 2 倍当前注，输者出局，平局发起者输
- 最少轮数限制后才能比牌（防止开局直接比牌）

### 牌局流程

```
开始 → 发牌 → 暗牌状态 → 轮流行动 → 看牌/跟注/加注/比牌/弃牌
                                        ↓
                              仅剩1人或触发比牌 → 结算 → 记录
```

---

## 3. AI 训练框架

### 训练方案：自博弈 + PPO

从随机策略开始自我对弈，用 PPO 算法不断优化。无需先验数据，消费级 GPU 即可训练。

### 网络架构：Actor-Critic 双头网络

```python
class ZhaJinHuaNet(nn.Module):
    # 输入：观察空间 → 特征向量（~246 维）
    #   手牌编码: 52 维 one-hot × 3 = 156 维
    #   公开信息: 筹码/底池归一化、对手状态 one-hot ≈ 50 维
    #   位置编码: 座位号 sin/cos 编码 = 8 维
    #   历史动作: 最近 N 轮动作序列嵌入 ≈ 32 维

    # 共享特征提取层
    shared_layers → 128 维特征

    # 策略头（Actor）→ 动作概率分布
    policy_head → [fold, call, raise_x2~x6, look, compare+targets] 的概率

    # 价值头（Critic）→ 预期胜率
    value_head → 标量 [-1, 1]
```

### 训练流程

```
阶段1：随机自博弈（冷启动）
  └─ 随机策略对局 → 收集经验 → 初步 PPO 训练
  └─ ~5 万局，约 2-3 小时（消费级 GPU）

阶段2：自我博弈迭代提升
  └─ 当前最优策略 vs 自己 → 生成对局
  └─ 策略网络更新 + 经验回放
  └─ 每 1 万局评估一次，保留最优 checkpoint
  └─ ~30-50 万局，约 1-2 天

阶段3：多样化对手池
  └─ 维护 N 个历史版本策略
  └─ 新策略必须战胜所有历史版本才升级
  └─ 防止策略退化（循环克制问题）
  └─ ~50 万局，约 2-3 天
```

### 奖励设计

```
即时奖励:
  +赢的筹码 / 初始筹码      （归一化到 [-1, 1]）
  +比牌获胜小奖励            （鼓励学习比牌时机）

局末奖励:
  最终名次奖励              （第1名 +1，末位 -1，线性插值）

不使用中间步骤奖励（避免奖励稀疏问题，只看最终结果）
```

### 训练监控

```
TensorBoard 记录:
  - 策略损失 / 价值损失曲线
  - 平均对局长度
  - 动作分布直方图（fold/call/raise/compare 比例）
  - Elo 评分变化曲线
  - 不同牌型的胜率矩阵
```

### 消费级 GPU 资源估算

```
RTX 3060 (12GB VRAM):
  - 网络参数量：~500K（轻量级）
  - 并行自博弈环境：~100 个
  - 训练吞吐：~2000 局/分钟
  - 总训练时间：3-5 天达到可玩水平
```

---

## 4. 游戏服务器 & 模式系统

### 三种模式的统一处理

```python
class Seat:
    player_type: Literal["ai", "human"]
    ai_model: Optional[str]        # 模型 checkpoint 路径
    ws_connection: Optional[WebSocket]  # 人类玩家的连接

# 模式只是座位配置的预设
MODES = {
    "ai_vs_ai":    [AI, AI, AI, AI, AI, AI],
    "ai_vs_human": [Human, AI, AI, AI, AI],
    "human_vs_human": [Human, Human, Human, AI, AI, AI],
}
# 用户可自由编辑每个座位 → 灵活组合
```

### 对局流程（同一套逻辑，两种节奏）

```
AI 回合：引擎直接调用策略网络推理 → 立即返回动作 → 毫秒级
Human 回合：通过 WebSocket 等待人类操作 → 超时自动弃牌 → 秒级

纯 AI 对局模式：可加速运行，每秒跑完一局，用于训练演示
混合模式：正常节奏，等待人类操作
纯人对局：AI 完全不介入
```

### 房间系统

```python
Room = {
    "id": uuid,
    "seats": [Seat × 3~6],
    "game_state": GameState,
    "speed": "normal" | "fast" | "turbo",
    "initial_chips": 1000,
    "min_bet": 10,
    "max_rounds": 50,
    "replay_enabled": true,
}
```

### API 设计

```
WebSocket 通道（实时对局）:
  ws://api/room/{id}/play
  → 客户端发送: {"action": "call"} / {"action": "raise", "amount": 100}
  → 服务端推送: {"event": "your_turn", "state": {...}}
                {"event": "action_made", "player": 2, "action": "raise"}
                {"event": "round_end", "winner": 3, "hands": [...]}

REST API（管理类）:
  POST   /rooms              创建房间
  GET    /rooms/{id}         房间状态
  PATCH  /rooms/{id}/seats   修改座位配置（切换 AI/Human）
  GET    /replays            对局回放列表
  GET    /replays/{id}       单局回放数据
  GET    /models             可用 AI 模型列表
```

### 对局回放

```python
Replay = {
    "id": uuid,
    "timestamp": datetime,
    "players": [{"type": "ai", "model": "v3", "seat": 0}, ...],
    "actions": [
        {"round": 1, "player": 0, "action": "look", "cards": [...]},
        {"round": 1, "player": 1, "action": "call"},
        {"round": 2, "player": 0, "action": "compare", "target": 2, "result": "win"},
    ],
    "final_hands": [{"player": 0, "cards": [...], "hand_type": "flush"}, ...],
    "chip_changes": [+300, -100, -200],
}
```

---

## 5. 前端界面

学术研究导向，重点在清晰展示。Vue 3 + Element Plus + ECharts。

### 四个核心页面

1. **大厅 (Lobby)** — 创建房间、快速开始三种模式、选择 AI 模型
2. **游戏桌面 (Game Table)** — 环形座位布局、手牌展示、操作按钮栏、底池/筹码显示
3. **训练监控 (Training Monitor)** — 实时训练曲线、动作分布、Elo 评分、训练控制
4. **策略分析 (Strategy Analysis)** — 对局回放播放、逐步查看 AI 决策详情、统计图表

### 技术选型

```
框架: Vue 3 + TypeScript
UI 库: Element Plus
图表: ECharts
通信: WebSocket（对局实时）+ Axios（REST 管理）
状态: Pinia
```

---

## 6. 项目结构 & 数据流

### 目录结构

```
zhajinhua/
├── engine/                     # 游戏引擎（纯逻辑，零依赖）
│   ├── cards.py                # 牌组、洗牌、发牌
│   ├── hand_evaluator.py       # 牌型判断与比较
│   ├── game.py                 # 牌局状态机、规则执行
│   ├── actions.py              # 动作定义与验证
│   └── test_engine.py          # 引擎单元测试
│
├── ai/                         # AI 训练与推理
│   ├── model.py                # Actor-Critic 网络
│   ├── features.py             # 观察空间 → 特征向量
│   ├── agent.py                # 推理封装（给定状态 → 选动作）
│   ├── ppo_trainer.py          # PPO 训练循环
│   ├── self_play.py            # 自博弈并行环境
│   ├── opponent_pool.py        # 历史策略池
│   └── train.py                # 训练入口脚本
│
├── server/                     # 游戏服务器
│   ├── main.py                 # FastAPI 入口
│   ├── routes/
│   │   ├── rooms.py            # 房间 CRUD
│   │   ├── game.py             # WebSocket 对局
│   │   ├── replays.py          # 回放查询
│   │   └── training.py         # 训练控制 API
│   ├── services/
│   │   ├── room_manager.py     # 房间生命周期
│   │   ├── game_runner.py      # 对局调度（统一 AI/Human）
│   │   └── replay_store.py     # 回放存储
│   └── schemas.py              # 请求/响应模型
│
├── web/                        # 前端 (Vue 3)
│   ├── src/
│   │   ├── views/
│   │   │   ├── Lobby.vue
│   │   │   ├── GameTable.vue
│   │   │   ├── TrainingMonitor.vue
│   │   │   └── StrategyAnalysis.vue
│   │   ├── components/
│   │   │   ├── SeatRing.vue        # 环形座位布局
│   │   │   ├── CardHand.vue        # 手牌展示
│   │   │   ├── ActionBar.vue       # 操作按钮栏
│   │   │   ├── TrainingChart.vue   # 训练曲线组件
│   │   │   └── DecisionDetail.vue  # AI 决策详情
│   │   ├── stores/                  # Pinia 状态
│   │   └── utils/
│   └── ...
│
├── data/                       # 运行时数据
│   ├── models/                 # AI 模型 checkpoints
│   ├── replays/                # 对局回放 JSON
│   └── logs/                   # 训练日志
│
├── configs/
│   ├── train_config.yaml       # 训练超参数
│   └── game_config.yaml        # 游戏规则配置
│
└── requirements.txt
```

### 关键数据流

```
训练流程:
  self_play.py → engine.Game × N 局并行 → 经验元组
    → (state, action, log_prob, reward) 存入 ReplayBuffer
    → ppo_trainer.py 取 batch 更新网络
    → 保存 checkpoint 到 data/models/

对局流程:
  客户端 WebSocket → server.game_runner
    → engine.Game 执行一步
    → 若是 AI 座位: agent.act(state) → 动作（毫秒级）
    → 若是 Human 座位: 等待 WebSocket 消息（秒级）
    → 每步记录到 replay_store
    → 广播状态更新给所有客户端

策略分析流程:
  加载 replay JSON → 逐步重放
    → 加载对应模型 → 获取当时的策略分布和预期胜率
    → 前端渲染决策树和概率分布
```

### 依赖清单

```
# AI 训练
torch >= 2.0
numpy

# 游戏服务器
fastapi
uvicorn[standard]
websockets
pydantic

# 训练监控
tensorboard

# 前端
vue 3
element-plus
echarts
pinia
axios
```

---

## 7. 实施阶段 & 验收标准

### 阶段 1：游戏引擎（地基）

```
交付物: engine/ 全部模块 + 完整单元测试

验收标准:
  ✓ 52 张牌洗牌发牌正确
  ✓ 全部 6 种牌型判断无误（百万次随机测试）
  ✓ 牌型比较涵盖所有边界情况
  ✓ 完整牌局模拟：3-6 人、暗牌/明牌、比牌/弃牌/加注
  ✓ 纯 Python 无外部依赖，单秒可模拟 1000+ 局
  ✓ 测试覆盖率 > 90%
```

### 阶段 2：AI 训练框架

```
交付物: ai/ 全部模块 + 训练脚本 + 预训练模型

验收标准:
  ✓ Actor-Critic 网络能正确输出动作概率和预期价值
  ✓ PPO 训练循环收敛（策略损失下降、胜率提升）
  ✓ 自博弈并行环境正常工作
  ✓ 训练 3 小时后 AI 能做出合理决策（不会盲目弃牌）
  ✓ 24 小时后 AI 策略具备基本牌力意识
  ✓ TensorBoard 可视化正常
  ✓ 模型 checkpoint 保存/加载正常
```

### 阶段 3：游戏服务器

```
交付物: server/ 全部模块 + REST/WebSocket API

验收标准:
  ✓ 创建/管理房间，配置 3-6 人座位
  ✓ 每个座位独立设置 AI(选模型) 或 Human
  ✓ WebSocket 对局实时通信稳定
  ✓ AI 回合毫秒响应，Human 回合等待操作
  ✓ 纯 AI 房间支持加速（快进回放模式）
  ✓ 对局回放完整记录并持久化
  ✓ 所有 API 有基本错误处理
```

### 阶段 4：前端界面

```
交付物: web/ 全部页面 + 组件

验收标准:
  ✓ 大厅：创建房间、快速开始三种模式、选择 AI 模型
  ✓ 游戏桌面：环形座位、手牌展示、操作按钮、底池显示
  ✓ 训练监控：实时训练曲线、动作分布、Elo 评分
  ✓ 策略分析：回放播放、逐步查看 AI 决策详情、统计图表
  ✓ 响应式布局，主要分辨率可用
```
