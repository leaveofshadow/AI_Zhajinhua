"""牌型判断与比较系统。

支持炸金花全部 6 种牌型判断，使用查表法实现 O(1) 评估。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import Dict, Tuple

from .cards import Card, all_cards as _all_cards

__all__ = ["HandType", "HandRank", "evaluate", "compare"]


class HandType(IntEnum):
    """牌型等级，值越大牌型越大。"""
    HIGH_CARD = 0       # 单张（散牌）
    PAIR = 1             # 对子
    STRAIGHT = 2         # 顺子
    FLUSH = 3            # 同花
    STRAIGHT_FLUSH = 4   # 同花顺
    THREE_OF_A_KIND = 5  # 豹子


@dataclass(frozen=True, slots=True)
class HandRank:
    """手牌评估结果，用于牌型比较。"""
    hand_type: HandType
    primary: int        # 主比较值
    kickers: tuple      # 踢脚牌 rank 值（降序），逐张比较用

    def __lt__(self, other: HandRank) -> bool:
        if not isinstance(other, HandRank):
            return NotImplemented
        return compare(self, other) < 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandRank):
            return NotImplemented
        return compare(self, other) == 0

    def __gt__(self, other: HandRank) -> bool:
        if not isinstance(other, HandRank):
            return NotImplemented
        return compare(self, other) > 0

    def __le__(self, other: HandRank) -> bool:
        return self == other or self < other

    def __ge__(self, other: HandRank) -> bool:
        return self == other or self > other


def compare(hand1: HandRank, hand2: HandRank) -> int:
    """比较两手牌大小。

    返回 >0 表示 hand1 赢, <0 表示 hand2 赢, =0 平局。
    只比较 HandType 和 rank 值，花色不参与比较。
    """
    if hand1.hand_type != hand2.hand_type:
        return int(hand1.hand_type) - int(hand2.hand_type)
    if hand1.primary != hand2.primary:
        return hand1.primary - hand2.primary
    for k1, k2 in zip(hand1.kickers, hand2.kickers):
        if k1 != k2:
            return k1 - k2
    return 0


# ---------------------------------------------------------------------------
# 查表法：预计算 C(52,3) = 22,100 种手牌 → HandRank
# ---------------------------------------------------------------------------

_LookupTable: Dict[int, HandRank] | None = None


def _build_lookup_table() -> Dict[int, HandRank]:
    """构建全部 22,100 种三张牌组合的 HandRank 查找表。

    key = 排序后的 (id0, id1, id2) 元组，用于快速查找。
    """
    table: Dict[int, HandRank] = {}
    all_cards = _all_cards()

    for c1, c2, c3 in combinations(all_cards, 3):
        rank = _evaluate_direct(c1, c2, c3)
        # 用排序后的 id 元组作为 key
        key = tuple(sorted((c1.id, c2.id, c3.id)))
        # 将 key 编码为单个 int 以节省内存
        key_int = key[0] * 52 * 52 + key[1] * 52 + key[2]
        table[key_int] = rank

    return table


def _evaluate_direct(c1: Card, c2: Card, c3: Card) -> HandRank:
    """直接计算三张牌的牌型（用于构建查找表）。"""
    ranks = sorted([c1.rank, c2.rank, c3.rank], reverse=True)
    suits = [c1.suit, c2.suit, c3.suit]

    is_flush = suits[0] == suits[1] == suits[2]
    is_three = ranks[0] == ranks[1] == ranks[2]
    is_pair = (ranks[0] == ranks[1]) or (ranks[1] == ranks[2])

    # 顺子判断
    is_straight = False
    straight_high = 0

    if ranks[0] - ranks[2] == 2 and ranks[0] != ranks[1] or (
        ranks[0] == ranks[1] and ranks[1] != ranks[2]
    ):
        # 不是顺子的条件先排除（有三张或对子）
        pass

    # 正常顺子：三张连续
    if not is_three and not is_pair:
        if ranks[0] - ranks[2] == 2 and len(set(ranks)) == 3:
            is_straight = True
            straight_high = ranks[0]
        # A-2-3 特殊顺子
        elif ranks == [12, 1, 0]:
            is_straight = True
            straight_high = 1  # A-2-3 视为最小的顺子，high=1(即3)
        # Q-K-A 顺子
        elif ranks == [12, 11, 10]:
            is_straight = True
            straight_high = 12

    # 确定牌型
    if is_three:
        return HandRank(
            hand_type=HandType.THREE_OF_A_KIND,
            primary=ranks[0],
            kickers=(),
        )

    if is_straight and is_flush:
        return HandRank(
            hand_type=HandType.STRAIGHT_FLUSH,
            primary=straight_high,
            kickers=(),
        )

    if is_flush:
        return HandRank(
            hand_type=HandType.FLUSH,
            primary=ranks[0],
            kickers=(ranks[0], ranks[1], ranks[2]),
        )

    if is_straight:
        return HandRank(
            hand_type=HandType.STRAIGHT,
            primary=straight_high,
            kickers=(),
        )

    if is_pair:
        # 找出对子和踢脚牌
        if ranks[0] == ranks[1]:
            pair_rank = ranks[0]
            kicker = ranks[2]
        else:
            pair_rank = ranks[1]
            kicker = ranks[0]
        return HandRank(
            hand_type=HandType.PAIR,
            primary=pair_rank,
            kickers=(pair_rank, kicker),
        )

    # 单张（散牌）
    return HandRank(
        hand_type=HandType.HIGH_CARD,
        primary=ranks[0],
        kickers=(ranks[0], ranks[1], ranks[2]),
    )


def evaluate(cards: list[Card] | tuple[Card, ...]) -> HandRank:
    """评估三张牌的牌型。

    使用查表法，O(1) 复杂度。
    """
    if len(cards) != 3:
        raise ValueError(f"Expected 3 cards, got {len(cards)}")

    global _LookupTable
    if _LookupTable is None:
        _LookupTable = _build_lookup_table()

    key = tuple(sorted((cards[0].id, cards[1].id, cards[2].id)))
    key_int = key[0] * 52 * 52 + key[1] * 52 + key[2]
    return _LookupTable[key_int]
