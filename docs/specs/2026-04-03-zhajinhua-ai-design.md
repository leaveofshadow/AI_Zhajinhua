# 炸金花 AI 博弈系统 — 系统设计说明书

> 版本：1.0
> 日期：2026-04-03
> 性质：纯学术研究

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [游戏引擎](#3-游戏引擎)
4. [AI 训练框架](#4-ai-训练框架)
5. [游戏服务器与模式系统](#5-游戏服务器与模式系统)
6. [前端界面](#6-前端界面)
7. [项目结构与数据流](#7-项目结构与数据流)
8. [实施阶段与验收标准](#8-实施阶段与验收标准)

---

## 1. 项目概述

### 1.1 目标

开发一个基于强化学习的炸金花（三张牌）卡牌博弈系统，用于学术研究。系统支持多个 AI Agent 通过自博弈（Self-Play）不断进化，同时允许人类玩家参与对局，用于策略验证和交互式研究。

### 1.2 核心特性

- **三种对局模式自由切换**：AI vs AI、AI vs Human、Human vs Human
- **强化学习驱动**：基于自博弈 + PPO 算法，从零开始学习，无需先验数据
- **完整数据记录**：每局对局的状态、动作、奖励全程记录，支持回放与分析
- **策略可解释性**：AI 每步决策的动作概率分布和预期胜率可视化展示

### 1.3 约束条件

| 约束项         | 说明                                        |
| -------------- | ------------------------------------------- |
| 训练硬件       | 消费级 GPU（RTX 3060 12GB 或同等）          |
| 训练数据       | 无先验数据，纯自博弈冷启动                   |
| 玩家人数       | 3-6 人，标准炸金花规则                       |
| 交付形态       | Web 应用，浏览器访问                         |
| 技术栈         | Python 全栈（FastAPI + PyTorch + Vue 3）    |

### 1.4 参考资料

- AlphaGo：MCTS + 深度强化学习的里程碑
- AlphaStar：多 Agent 自博弈 + PPO 在复杂不完全信息博弈中的应用
- Libratus：CFR 在扑克 AI 中的经典实践

---

## 2. 整体架构

### 2.1 分层架构图

```
┌─────────────────────────────────────────────────────────┐
│                    表现层 — Web 前端                      │
│            Vue 3 + Element Plus + ECharts                │
│    游戏桌面  |  训练监控  |  策略分析  |  对局大厅          │
└──────────────────────┬──────────────────────────────────┘
                       │ WebSocket（对局）/ REST（管理）
┌──────────────────────▼──────────────────────────────────┐
│                  服务层 — 游戏服务器                       │
│                    FastAPI + Uvicorn                     │
│     对局管理  |  模式调度  |  房间系统  |  回放存储          │
└────────┬──────────────────────────────┬─────────────────┘
         │                              │
┌────────▼─────────┐        ┌──────────▼──────────────────┐
│   领域层 — 游戏引擎 │        │    智能层 — AI 训练框架       │
│   纯 Python 逻辑   │        │   PyTorch + Self-Play + PPO │
│  牌局 | 规则 | 状态  │        │  策略网络 | 价值网络 | 日志    │
│  牌型判断 | 动作验证 │        │  经验缓冲 | 对手池 | 监控     │
└──────────────────┘        └─────────────────────────────┘
```

### 2.2 核心设计原则

| 原则                   | 说明                                                         |
| ---------------------- | ------------------------------------------------------------ |
| 引擎无 IO              | 游戏引擎为纯逻辑模块，不依赖网络、文件等 IO，训练时可高速批量运行 |
| 训练与推理解耦         | 训练产出模型 checkpoint，游戏服务器加载后做推理，两者独立部署     |
| 座位级模式控制         | 每个座位独立标记 AI/Human，模式切换无需重启对局                 |
| 全量数据记录           | 所有对局数据完整保存，支持事后策略分析                          |
| 轻量依赖               | 仅使用成熟稳定的核心依赖，避免过度工程                          |

---

## 3. 游戏引擎

游戏引擎是整个系统的地基，实现完整的炸金花规则，同时满足训练时的高吞吐量和 Web 对战时的状态追踪需求。

### 3.1 牌型体系

标准 52 张扑克牌（无大小王），每人发 3 张。

**牌型排名**（从大到小）：

| 排名 | 牌型   | 说明                    | 示例         |
| ---- | ------ | ----------------------- | ------------ |
| 1    | 豹子   | 三张同点数              | A♠ A♥ A♦     |
| 2    | 同花顺 | 同花色的连续三张         | A♠ K♠ Q♠    |
| 3    | 同花   | 同花色的非连续三张       | A♠ K♠ 9♠    |
| 4    | 顺子   | 不同花色的连续三张       | A♠ K♥ Q♦    |
| 5    | 对子   | 两张同点数 + 一张散牌    | A♠ A♥ 9♦    |
| 6    | 单张   | 无以上任何组合           | A♠ K♥ 9♦    |

**特殊规则**：
- A-2-3 为最小顺子
- 同牌型按点数逐张比较
- 花色不参与比较（同点数同牌型视为平局）

**状态空间规模**：C(52, 3) = 22,100 种手牌组合。

### 3.2 玩家观察空间（不完全信息）

每个玩家只能看到自己的手牌和其他玩家的公开行为，这是不完全信息博弈的核心。

```python
observation = {
    # --- 私有信息 ---
    "my_cards": list[Card],       # 自己的 3 张牌（未看牌时为空）
    "my_chips": int,              # 自己当前筹码
    "has_looked": bool,           # 自己是否已看牌

    # --- 公共信息 ---
    "pot": int,                   # 底池总额
    "current_bet": int,           # 当前跟注额
    "round_count": int,           # 当前轮数（第几轮下注）
    "my_position": int,           # 自己的座位号（0-5）
    "active_players": int,        # 还在局中的玩家数

    # --- 对手信息 ---
    "player_states": list[dict],  # 每个对手的公开状态
    # 每个对手包含:
    #   is_active: bool           是否还在局中
    #   is_looked: bool           是否已看牌
    #   total_bet: int            本局累计下注额
    #   last_action: str          上一个动作
    #   chip_count: int           筹码（公开可见）
}
```

### 3.3 动作空间

| 动作           | 编码               | 说明                                       |
| -------------- | ------------------ | ------------------------------------------ |
| 弃牌 (fold)    | 0                  | 放弃本局，已下注归底池                       |
| 跟注 (call)    | 1                  | 跟当前注额（暗牌为明注的一半）               |
| 加注 (raise)   | 2, 3, 4, 5, 6     | 加注到当前注的 2x / 3x / 4x / 5x / 6x      |
| 看牌 (look)    | 7                  | 翻看自己的牌（不可逆）                       |
| 比牌 (compare) | 8 + target_index  | 与指定对手比牌，发起者需付 2 倍当前注        |

### 3.4 核心规则

1. **暗牌机制**：开局所有玩家处于暗牌状态（未看牌）。暗牌玩家跟注额为明牌玩家的一半
2. **看牌**：玩家随时可以选择看牌。看牌后跟注额翻倍，但获得牌面信息优势
3. **比牌**：
   - 最少经过 N 轮（可配置，默认 2 轮）后才可比牌
   - 发起者需付出 2 倍当前注
   - 牌型大者胜，平局时发起者判负
   - 比牌后输者出局
4. **弃牌**：随时可以弃牌，已下注归底池
5. **局结束条件**：仅剩 1 人 或 某次比牌后仅剩 1 人
6. **防死循环**：设置最大轮数限制（默认 50 轮），超时强制比牌

### 3.5 牌局流程

```
初始化
  │
  ▼
发牌（每人3张，暗牌状态）
  │
  ▼
┌─ 轮流行动 ◄──────────────────────────┐
│   │                                    │
│   ├─ 看牌 → 翻看手牌，状态切换          │
│   ├─ 跟注 → 按当前注额下注             │
│   ├─ 加注 → 提高注额                   │
│   ├─ 比牌 → 选对手比较，输者出局        │
│   ├─ 弃牌 → 退出本局                   │
│   │                                    │
│   ▼                                    │
│  检查结束条件                           │
│   ├─ 仅剩1人 → 结算                    │
│   └─ 继续 ────────────────────────────┘
│
▼
结算
  ├─ 计算底池分配
  ├─ 记录对局数据
  └─ 更新筹码
```

### 3.6 性能要求

- 纯 Python 实现，无外部依赖
- 单秒可模拟 1000+ 局（用于训练时批量生成经验）
- 接口设计支持向量化并行（多个牌局同时推进）

---

## 4. AI 训练框架

### 4.1 训练方案选型

**方案：自博弈 + PPO（Proximal Policy Optimization）**

选择理由：
- 无需先验数据，纯自博弈从零学习
- PyTorch 生态成熟，实现简洁
- 消费级 GPU（RTX 3060 12GB）即可训练
- 在不完全信息博弈中有成功先例（AlphaStar 简化版）
- 扩展性好，后续可对比其他 RL 算法

### 4.2 网络架构

采用 Actor-Critic 双头网络：

```
输入层
  │  特征向量（~246 维）
  ▼
┌─────────────────────────────┐
│     共享特征提取层             │
│  Linear(246, 256)            │
│  ReLU                        │
│  Linear(256, 128)            │
│  ReLU                        │
│  → 128 维特征表示             │
└──────┬──────────────┬────────┘
       │              │
       ▼              ▼
┌──────────────┐ ┌──────────────┐
│  策略头 Actor  │ │ 价值头 Critic │
│ Linear(128,64)│ │ Linear(128,64)│
│ ReLU         │ │ ReLU          │
│ Linear(64,N) │ │ Linear(64,1)  │
│ Softmax      │ │ Tanh          │
└──────────────┘ └──────────────┘
  N = 动作空间大小     输出: [-1, 1]
  输出: 动作概率分布    预期胜率
```

**特征工程**：

| 特征类别     | 维度   | 编码方式                                  |
| ------------ | ------ | ----------------------------------------- |
| 手牌         | 156 维 | 每张牌 52 维 one-hot，3 张拼接             |
| 公共信息     | ~50 维 | 筹码/底池归一化、对手状态 one-hot           |
| 位置编码     | 8 维   | 座位号 sin/cos 位置编码                     |
| 历史动作     | 32 维  | 最近 N 轮的动作序列嵌入                     |
| **总计**     | **~246 维** |                                         |

**网络参数量**：约 500K 参数，轻量级设计适合消费级 GPU。

### 4.3 自博弈训练流程

```
┌─────────────────────────────────────────────────────┐
│                    训练主循环                         │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐   │
│  │ 策略网络  │───▶│ 自博弈    │───▶│ 经验收集      │   │
│  │ (当前版本)│    │ (并行×100)│    │ (ReplayBuffer)│  │
│  └──────────┘    └──────────┘    └──────┬───────┘   │
│       ▲                                 │           │
│       │         ┌──────────┐            │           │
│       └─────────│ PPO 更新  │◄───────────┘           │
│                 │ (策略+价值)│                        │
│                 └──────────┘                         │
│                      │                               │
│                      ▼                               │
│                 ┌──────────┐                         │
│                 │ 评估 &    │                         │
│                 │ Checkpoint│                        │
│                 └──────────┘                         │
└─────────────────────────────────────────────────────┘
```

**三阶段训练计划**：

**阶段 1：冷启动（~5 万局，约 2-3 小时）**
- 随机策略自博弈，收集初始经验
- 初步 PPO 训练，建立基本策略
- AI 学会基本规则：不会盲目弃牌、不会一直加注

**阶段 2：迭代提升（~30-50 万局，约 1-2 天）**
- 当前最优策略自我对弈
- 每 1 万局评估一次，保留最优 checkpoint
- AI 逐渐学会牌力评估、位置意识、下注尺度

**阶段 3：对手池防退化（~50 万局，约 2-3 天）**
- 维护 N 个（默认 10 个）历史版本策略
- 新策略必须战胜所有历史版本才升级
- 防止策略循环克制问题
- 最终产出多版本模型（弱/中/强）

### 4.4 奖励设计

```python
# 局末奖励（主要信号）
def compute_reward(player, game_result):
    # 最终名次奖励
    rank_reward = linear_interpolate(
        rank=game_result.rank,
        best=1.0,           # 第1名
        worst=-1.0,         # 最后一名
        total=game_result.num_players
    )

    # 筹码变化奖励（归一化）
    chip_reward = game_result.chip_change / initial_chips

    # 综合奖励
    return 0.6 * rank_reward + 0.4 * clip(chip_reward, -1, 1)
```

**设计要点**：
- 主要依赖局末最终结果，避免中间步骤奖励带来的偏差
- 名次奖励 + 筹码变化加权，兼顾生存和盈利
- 归一化到 [-1, 1]，训练稳定

### 4.5 训练监控

所有训练指标通过 TensorBoard 实时记录：

| 指标类别     | 具体指标                                       |
| ------------ | ---------------------------------------------- |
| 训练损失     | 策略损失、价值损失、熵（策略多样性）            |
| 对局统计     | 平均对局长度、每轮动作分布                      |
| 策略质量     | Elo 评分变化曲线、不同版本间胜率矩阵            |
| 行为分析     | 动作分布直方图、不同牌型的胜率、诈唬频率         |

### 4.6 资源估算

```
硬件: RTX 3060 12GB VRAM
网络参数量: ~500K
并行自博弈环境: ~100 个
训练吞吐: ~2000 局/分钟
显存占用: ~4GB（含经验缓冲）
总训练时间: 3-5 天达到可玩水平
```

---

## 5. 游戏服务器与模式系统

### 5.1 统一模式架构

系统通过座位级配置实现三种模式的无缝切换：

```python
class Seat:
    """单个座位配置"""
    player_type: Literal["ai", "human"]
    ai_model: Optional[str]          # 模型 checkpoint 路径（AI 模式）
    ws_connection: Optional[WebSocket]  # WebSocket 连接（Human 模式）
    display_name: str                 # 显示名称

class Room:
    """房间配置"""
    id: UUID
    seats: list[Seat]                # 3-6 个座位
    speed: Literal["normal", "fast", "turbo"]
    initial_chips: int = 1000
    min_bet: int = 10
    max_rounds: int = 50
```

**预设模式**：

| 模式            | 默认配置                              | 说明                     |
| --------------- | ------------------------------------- | ------------------------ |
| AI vs AI        | 6 个 AI 座位                          | 纯 AI 对决，可加速观摩    |
| AI vs Human     | 1 Human + 5 AI                        | 人机对战                  |
| Human vs Human  | 3 Human + 3 AI                        | 多人混合对局              |

用户可在创建房间后自由修改任意座位的类型和 AI 模型版本。

### 5.2 对局调度

```python
class GameRunner:
    """统一的对局调度器，处理 AI 和 Human 混合场景"""

    async def run_turn(self, room: Room, game: Game):
        current_seat = room.seats[game.current_player]

        if current_seat.player_type == "ai":
            # AI 回合：同步推理，毫秒级完成
            state = game.get_observation(game.current_player)
            action = self.agent.act(state, model=current_seat.ai_model)
            game.apply_action(action)
            await self.broadcast_action(room, action)

        else:
            # Human 回合：通过 WebSocket 等待，秒级
            await self.notify_player(room, game.current_player)
            action = await self.wait_for_action(
                room, game.current_player, timeout=30
            )
            game.apply_action(action)
            await self.broadcast_action(room, action)
```

**节奏控制**：
- **normal**：人类玩家 30 秒操作时间，AI 等待 1 秒（模拟思考）
- **fast**：人类玩家 15 秒，AI 即时
- **turbo**：纯 AI 房间专用，每秒可跑完一局，用于训练演示

### 5.3 API 设计

**WebSocket 通道**（实时对局）：

```
端点: ws://api/room/{room_id}/play

客户端 → 服务端:
  {"action": "call"}
  {"action": "raise", "multiplier": 3}
  {"action": "compare", "target": 2}
  {"action": "look"}
  {"action": "fold"}

服务端 → 客户端:
  {"event": "game_start", "hands": {...}, "position": 0}
  {"event": "your_turn", "state": {...}, "timeout": 30}
  {"event": "player_action", "player": 2, "action": "raise", "amount": 100}
  {"event": "round_end", "winner": 3, "hands": [...], "chip_changes": [...]}
  {"event": "game_error", "message": "invalid action"}
```

**REST API**（管理类操作）：

| 方法   | 路径                  | 说明                     |
| ------ | --------------------- | ------------------------ |
| POST   | `/rooms`              | 创建房间                 |
| GET    | `/rooms/{id}`         | 查询房间状态             |
| PATCH  | `/rooms/{id}/seats`   | 修改座位配置             |
| DELETE | `/rooms/{id}`         | 关闭房间                 |
| GET    | `/replays`            | 对局回放列表（分页）     |
| GET    | `/replays/{id}`       | 单局回放详情             |
| GET    | `/models`             | 可用 AI 模型列表         |
| POST   | `/training/start`     | 启动训练任务             |
| POST   | `/training/stop`      | 停止训练                 |
| GET    | `/training/status`    | 训练状态与指标           |

### 5.4 对局回放格式

```json
{
    "id": "uuid",
    "timestamp": "2026-04-03T14:30:00Z",
    "config": {
        "num_players": 4,
        "initial_chips": 1000,
        "min_bet": 10
    },
    "players": [
        {"seat": 0, "type": "ai", "model": "v3", "name": "AI-Alpha"},
        {"seat": 1, "type": "human", "name": "Player1"},
        {"seat": 2, "type": "ai", "model": "v2", "name": "AI-Beta"},
        {"seat": 3, "type": "ai", "model": "v3", "name": "AI-Gamma"}
    ],
    "actions": [
        {"round": 0, "player": 0, "action": "look", "cards": ["A♠", "K♠", "Q♠"]},
        {"round": 0, "player": 1, "action": "call", "amount": 10},
        {"round": 0, "player": 2, "action": "call", "amount": 10},
        {"round": 0, "player": 3, "action": "raise", "multiplier": 3, "amount": 30},
        {"round": 1, "player": 0, "action": "raise", "multiplier": 4, "amount": 40},
        {"round": 1, "player": 1, "action": "fold"},
        {"round": 2, "player": 0, "action": "compare", "target": 3, "result": "win"}
    ],
    "result": {
        "winner": 0,
        "final_hands": [
            {"player": 0, "cards": ["A♠", "K♠", "Q♠"], "hand_type": "straight_flush"},
            {"player": 2, "cards": ["J♥", "10♥", "9♥"], "hand_type": "flush"},
            {"player": 3, "cards": ["K♦", "K♣", "7♠"], "hand_type": "pair"}
        ],
        "chip_changes": [+450, -30, -60, -360]
    }
}
```

---

## 6. 前端界面

### 6.1 技术选型

| 技术         | 选型          | 理由                         |
| ------------ | ------------- | ---------------------------- |
| 框架         | Vue 3 + TS    | 轻量、响应式、组件化          |
| UI 组件库    | Element Plus  | 开箱即用，表格/表单/弹窗齐全  |
| 图表库       | ECharts       | 训练曲线、分布图、热力图      |
| 状态管理     | Pinia         | Vue 3 官方推荐               |
| 实时通信     | WebSocket     | 对局实时同步                  |
| HTTP 客户端  | Axios         | REST API 调用                 |

### 6.2 页面设计

#### 页面 1：对局大厅（Lobby）

**功能**：
- 快速开始：一键进入 AI vs AI / AI vs Human / Human vs Human 三种模式
- 自定义房间：选择人数、初始筹码、速度、每个座位的类型和 AI 模型版本
- 可用模型列表：展示已训练的 AI 模型及其 Elo 评分

#### 页面 2：游戏桌面（Game Table）

**功能**：
- 环形座位布局：3-6 个座位环绕桌面中央底池
- 手牌展示：自己的牌正面展示，对手牌背面展示（局末翻开）
- 操作栏：弃牌/跟注/加注/看牌/比牌按钮，根据当前状态动态启用
- 实时状态：底池金额、各玩家筹码、当前轮数
- 操作日志：右侧面板实时滚动显示所有玩家动作

#### 页面 3：训练监控（Training Monitor）

**功能**：
- 训练控制：启动/停止训练、选择超参数配置、加载 checkpoint
- 实时图表：策略损失/价值损失曲线、动作分布直方图、Elo 评分曲线
- 训练统计：已完成对局数、训练时长、当前版本号

#### 页面 4：策略分析（Strategy Analysis）

**功能**：
- 回放播放器：选择历史对局，支持播放/暂停/逐帧/变速
- AI 决策详情：每一步展示 AI 的动作概率分布和预期胜率
- 统计面板：
  - 各牌型胜率热力图
  - 不同位置的盈亏分析
  - 诈唬（差牌加注）成功率
  - AI 版本间对比分析

### 6.3 组件清单

| 组件               | 职责                         | 所属页面          |
| ------------------ | ---------------------------- | ----------------- |
| `SeatRing`         | 环形座位布局与玩家信息展示    | GameTable         |
| `CardHand`         | 手牌渲染（正面/背面/翻开）   | GameTable         |
| `ActionBar`        | 操作按钮栏（动态状态）       | GameTable         |
| `ActionLog`        | 实时操作日志面板             | GameTable         |
| `RoomConfig`       | 房间配置表单                 | Lobby             |
| `ModelSelector`    | AI 模型选择器（含 Elo）      | Lobby, GameTable  |
| `TrainingChart`    | 训练曲线图表                 | TrainingMonitor   |
| `ActionDistChart`  | 动作分布直方图               | TrainingMonitor   |
| `ReplayPlayer`     | 回放播放控制器               | StrategyAnalysis  |
| `DecisionDetail`   | AI 单步决策详情面板          | StrategyAnalysis  |
| `StatsHeatmap`     | 胜率/策略热力图              | StrategyAnalysis  |

---

## 7. 项目结构与数据流

### 7.1 目录结构

```
zhajinhua/
│
├── engine/                         # 游戏引擎（纯逻辑，零外部依赖）
│   ├── __init__.py
│   ├── cards.py                    # Card/Deck 类，洗牌、发牌
│   ├── hand_evaluator.py           # 牌型判断、比较、排序
│   ├── game.py                     # GameState 状态机，规则执行
│   ├── actions.py                  # Action 定义、验证、执行
│   └── tests/
│       ├── test_cards.py
│       ├── test_hand_evaluator.py
│       ├── test_game.py
│       └── test_actions.py
│
├── ai/                             # AI 训练与推理
│   ├── __init__.py
│   ├── model.py                    # Actor-Critic 双头网络
│   ├── features.py                 # 观察空间 → 特征向量编码
│   ├── agent.py                    # 推理封装（状态 → 动作）
│   ├── ppo_trainer.py              # PPO 训练循环
│   ├── self_play.py                # 并行自博弈环境
│   ├── opponent_pool.py            # 历史策略池管理
│   ├── train.py                    # 训练入口脚本
│   └── configs/
│       ├── default.yaml            # 默认训练超参数
│       └── fast.yaml               # 快速实验配置
│
├── server/                         # 游戏服务器
│   ├── __init__.py
│   ├── main.py                     # FastAPI 应用入口
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── rooms.py                # 房间 CRUD 端点
│   │   ├── game.py                 # WebSocket 对局端点
│   │   ├── replays.py              # 回放查询端点
│   │   └── training.py             # 训练控制端点
│   ├── services/
│   │   ├── __init__.py
│   │   ├── room_manager.py         # 房间生命周期管理
│   │   ├── game_runner.py          # 对局调度（AI/Human 统一）
│   │   └── replay_store.py         # 回放持久化
│   └── schemas.py                  # Pydantic 请求/响应模型
│
├── web/                            # 前端 (Vue 3)
│   ├── src/
│   │   ├── App.vue
│   │   ├── main.ts
│   │   ├── router/
│   │   │   └── index.ts
│   │   ├── views/
│   │   │   ├── Lobby.vue
│   │   │   ├── GameTable.vue
│   │   │   ├── TrainingMonitor.vue
│   │   │   └── StrategyAnalysis.vue
│   │   ├── components/
│   │   │   ├── SeatRing.vue
│   │   │   ├── CardHand.vue
│   │   │   ├── ActionBar.vue
│   │   │   ├── ActionLog.vue
│   │   │   ├── RoomConfig.vue
│   │   │   ├── ModelSelector.vue
│   │   │   ├── TrainingChart.vue
│   │   │   ├── ActionDistChart.vue
│   │   │   ├── ReplayPlayer.vue
│   │   │   ├── DecisionDetail.vue
│   │   │   └── StatsHeatmap.vue
│   │   ├── stores/
│   │   │   ├── room.ts
│   │   │   ├── game.ts
│   │   │   └── training.ts
│   │   └── utils/
│   │       ├── ws.ts               # WebSocket 客户端封装
│   │       └── api.ts              # REST API 封装
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── data/                           # 运行时数据（gitignore）
│   ├── models/                     # AI 模型 checkpoints
│   ├── replays/                    # 对局回放 JSON 文件
│   └── logs/                       # 训练日志 + TensorBoard
│
├── configs/
│   ├── train_config.yaml           # 训练超参数
│   └── game_config.yaml            # 游戏规则配置
│
├── docs/
│   └── specs/
│       └── 2026-04-03-zhajinhua-ai-design.md  # 本文档
│
├── CLAUDE.md                       # Claude Code 项目指引
├── requirements.txt                # Python 依赖
└── .gitignore
```

### 7.2 关键数据流

#### 训练流程

```
self_play.py
  │
  ├─ 启动 N 个并行引擎实例 (engine.Game)
  │
  ├─ 每局:
  │   ├─ 初始化牌局 → 轮流调用策略网络选动作 → 推进状态
  │   ├─ 收集 (state, action, log_prob, reward) 经验元组
  │   └─ 存入 ReplayBuffer
  │
  ▼
ppo_trainer.py
  │
  ├─ 从 ReplayBuffer 采样 batch
  ├─ 计算策略损失 (clipped surrogate) + 价值损失
  ├─ 反向传播更新网络参数
  ├─ 每 K 步评估并保存 checkpoint → data/models/
  └─ TensorBoard 写入指标 → data/logs/
```

#### 对局流程

```
客户端 (Vue)
  │
  ├─ WebSocket 连接: ws://api/room/{id}/play
  │
  ├─ 发送动作:
  │   {"action": "raise", "multiplier": 3}
  │
  ▼
GameRunner (server)
  │
  ├─ engine.Game 推进一步
  │
  ├─ 判断下一个玩家:
  │   ├─ AI 座位 → agent.act(state) → 毫秒级返回
  │   └─ Human 座位 → 等待 WebSocket 消息 → 秒级
  │
  ├─ 记录到 ReplayStore → data/replays/{id}.json
  │
  └─ 广播状态更新给所有客户端
```

#### 策略分析流程

```
StrategyAnalysis 页面
  │
  ├─ 加载回放列表: GET /replays
  ├─ 选择对局: GET /replays/{id}
  │
  ├─ 逐步重放:
  │   ├─ 渲染每步的牌面和动作
  │   ├─ 加载对应 AI 模型
  │   ├─ 获取该步的策略分布 (policy_head 输出)
  │   ├─ 获取该步的预期胜率 (value_head 输出)
  │   └─ 展示决策详情
  │
  └─ 批量统计:
      ├─ 各牌型胜率矩阵
      ├─ 位置-筹码变化热力图
      └─ AI 版本对比雷达图
```

### 7.3 依赖清单

```txt
# requirements.txt

# AI 训练
torch>=2.0
numpy

# 游戏服务器
fastapi
uvicorn[standard]
websockets
pydantic

# 训练监控
tensorboard

# 开发工具
pytest
pytest-cov
```

```json
// web/package.json (核心依赖)
{
    "vue": "^3.4",
    "element-plus": "^2.5",
    "echarts": "^5.5",
    "pinia": "^2.1",
    "axios": "^1.6",
    "typescript": "^5.3"
}
```

---

## 8. 实施阶段与验收标准

### 阶段 1：游戏引擎

**目标**：实现完整的炸金花规则引擎，支持高速批量模拟。

**交付物**：
- `engine/` 全部模块
- 完整单元测试套件

**验收标准**：
- [ ] 52 张牌洗牌、发牌正确（随机性验证）
- [ ] 全部 6 种牌型判断无误（百万次随机测试通过）
- [ ] 牌型比较涵盖所有边界情况（豹子 vs 同花顺、同点数对子等）
- [ ] 完整牌局模拟：支持 3-6 人、暗牌/明牌切换、比牌/弃牌/加注
- [ ] 纯 Python 无外部依赖，单秒可模拟 1000+ 局
- [ ] 接口支持向量化（多局并行推进）
- [ ] 测试覆盖率 > 90%

### 阶段 2：AI 训练框架

**目标**：搭建 Actor-Critic 网络和 PPO 训练循环，训练出可用的 AI 模型。

**交付物**：
- `ai/` 全部模块
- 训练入口脚本和配置文件
- 至少一个预训练模型 checkpoint

**验收标准**：
- [ ] Actor-Critic 网络正确输出动作概率分布和预期价值
- [ ] 特征编码（手牌/公开信息/位置/历史）正确实现
- [ ] PPO 训练循环收敛（策略损失下降、价值损失下降）
- [ ] 自博弈并行环境正常工作（多局同时运行）
- [ ] 对手池机制正常（历史版本保存与加载）
- [ ] 训练 3 小时后 AI 做出合理决策（非盲目弃牌/加注）
- [ ] 训练 24 小时后 AI 具备牌力意识和位置意识
- [ ] TensorBoard 可视化正常显示所有指标
- [ ] 模型 checkpoint 保存/加载功能正常

### 阶段 3：游戏服务器

**目标**：实现 FastAPI 后端，支持三种对局模式和完整回放。

**交付物**：
- `server/` 全部模块
- REST API 和 WebSocket 端点

**验收标准**：
- [ ] 创建/关闭/查询房间功能正常
- [ ] 支持 3-6 人座位配置
- [ ] 每个座位独立设置 AI（选择模型版本）或 Human
- [ ] WebSocket 对局实时通信稳定（延迟 < 100ms）
- [ ] AI 回合毫秒响应，Human 回合等待操作（超时自动弃牌）
- [ ] 纯 AI 房间支持加速模式（turbo 速度）
- [ ] 对局回放完整记录（含全部动作和最终手牌）
- [ ] 回放查询 API 支持分页和过滤
- [ ] 所有 API 有基本错误处理和参数校验

### 阶段 4：前端界面

**目标**：实现四个核心页面，提供完整的交互体验。

**交付物**：
- `web/` 全部页面和组件

**验收标准**：
- [ ] **大厅**：创建房间（自定义人数/筹码/座位配置）、三种模式快速开始、AI 模型选择
- [ ] **游戏桌面**：环形座位布局、手牌展示（正/背面）、操作按钮栏、底池/筹码显示、操作日志
- [ ] **训练监控**：启动/停止训练、实时训练曲线、动作分布图、Elo 评分
- [ ] **策略分析**：回放选择与播放（播放/暂停/逐帧/变速）、AI 决策详情（概率分布+胜率）、统计图表
- [ ] 响应式布局，1920×1080 和 1440×900 分辨率可用
