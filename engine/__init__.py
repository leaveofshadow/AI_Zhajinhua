"""炸金花游戏引擎 — 纯逻辑模块，零外部依赖。"""

from .actions import Action, ActionType, ActionValidator
from .cards import Card, Deck, deal_multiple_hands
from .game import (
    Game,
    GamePhase,
    GameResult,
    GameState,
    PlayerState,
    VectorizedGameEnv,
)
from .hand_evaluator import HandRank, HandType, compare, evaluate

__all__ = [
    # cards
    "Card",
    "Deck",
    "deal_multiple_hands",
    # hand_evaluator
    "HandType",
    "HandRank",
    "evaluate",
    "compare",
    # actions
    "ActionType",
    "Action",
    "ActionValidator",
    # game
    "GamePhase",
    "PlayerState",
    "GameState",
    "Game",
    "GameResult",
    "VectorizedGameEnv",
]
