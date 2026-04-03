"""牌组系统：Card、Deck 类及批量发牌功能。"""

from __future__ import annotations

import random
from typing import List

__all__ = ["Card", "Deck", "deal_multiple_hands"]

# 花色符号映射
_SUIT_SYMBOLS = ["♠", "♥", "♦", "♣"]
_SUIT_NAMES = ["spade", "heart", "diamond", "club"]

# 点数符号映射 (rank 0-12 对应 2-A)
_RANK_SYMBOLS = [
    "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"
]

# 预计算全部 52 张 Card 单例
_CARDS: list[Card] | None = None


class Card:
    """一张扑克牌。

    suit: 花色 0-3 (♠♥♦♣)
    rank: 点数 0-12 (2=0, 3=1, ..., A=12)

    内部用 _id = suit * 13 + rank 做整数编码，方便 one-hot 和比较。
    """

    __slots__ = ("_id", "suit", "rank")

    def __init__(self, suit: int, rank: int) -> None:
        if not (0 <= suit <= 3 and 0 <= rank <= 12):
            raise ValueError(f"Invalid card: suit={suit}, rank={rank}")
        self._id: int = suit * 13 + rank
        self.suit: int = suit
        self.rank: int = rank

    @property
    def id(self) -> int:
        """唯一标识 0-51，可用于 one-hot 编码。"""
        return self._id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self._id == other._id

    def __lt__(self, other: Card) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        # 先比 rank（牌力），再比 suit
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.suit < other.suit

    def __le__(self, other: Card) -> bool:
        return self == other or self < other

    def __gt__(self, other: Card) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return other < self

    def __ge__(self, other: Card) -> bool:
        return self == other or self > other

    def __hash__(self) -> int:
        return self._id

    def __repr__(self) -> str:
        return f"Card({_SUIT_NAMES[self.suit]}, {_RANK_SYMBOLS[self.rank]})"

    def __str__(self) -> str:
        return f"{_RANK_SYMBOLS[self.rank]}{_SUIT_SYMBOLS[self.suit]}"

    @classmethod
    def from_id(cls, card_id: int) -> Card:
        """从整数 id (0-51) 获取 Card 单例。"""
        _ensure_card_cache()
        return _CARDS[card_id]


def _ensure_card_cache() -> None:
    """惰性初始化 52 张 Card 单例。"""
    global _CARDS
    if _CARDS is None:
        _CARDS = [Card(s, r) for s in range(4) for r in range(13)]


def all_cards() -> list[Card]:
    """返回全部 52 张牌的列表。"""
    _ensure_card_cache()
    return list(_CARDS)  # 返回副本


class Deck:
    """一副标准 52 张扑克牌，支持洗牌和发牌。"""

    __slots__ = ("_cards",)

    def __init__(self, seed: int | None = None) -> None:
        _ensure_card_cache()
        self._cards: list[Card] = list(_CARDS)  # 副本
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self._cards)
        else:
            random.shuffle(self._cards)

    @property
    def remaining(self) -> int:
        """牌堆剩余张数。"""
        return len(self._cards)

    def shuffle(self) -> None:
        """Fisher-Yates 洗牌（random.shuffle 内部即为此算法）。"""
        random.shuffle(self._cards)

    def deal(self, n: int = 1) -> list[Card]:
        """从牌堆顶部发 n 张牌。"""
        if n > len(self._cards):
            raise ValueError(
                f"Cannot deal {n} cards, only {len(self._cards)} remaining"
            )
        dealt = self._cards[:n]
        self._cards = self._cards[n:]
        return dealt

    def reset(self) -> None:
        """重置牌堆为完整 52 张并洗牌。"""
        _ensure_card_cache()
        self._cards = list(_CARDS)
        random.shuffle(self._cards)


def deal_multiple_hands(
    num_players: int,
    num_hands: int = 1,
) -> list[list[list[Card]]]:
    """批量发牌：一次生成多局的手牌结果。

    返回: [hand_idx][player_idx][card_idx]
    每局每人 3 张牌。
    """
    _ensure_card_cache()
    results: list[list[list[Card]]] = []
    for _ in range(num_hands):
        cards = list(_CARDS)
        random.shuffle(cards)
        hands: list[list[Card]] = []
        idx = 0
        for _ in range(num_players):
            hands.append(cards[idx : idx + 3])
            idx += 3
        results.append(hands)
    return results
