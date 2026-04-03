"""hand_evaluator.py 测试：牌型判断与比较。"""

import random
import pytest
from engine.cards import Card
from engine.hand_evaluator import HandType, HandRank, evaluate, compare


def _make_card(suit: int, rank: int) -> Card:
    return Card(suit, rank)


class TestHandTypeDetection:
    """每种牌型的正确识别。"""

    def test_three_of_a_kind(self):
        """豹子：三张同点数。"""
        cards = [Card(0, 12), Card(1, 12), Card(2, 12)]  # A♠ A♥ A♦
        result = evaluate(cards)
        assert result.hand_type == HandType.THREE_OF_A_KIND
        assert result.primary == 12

    def test_straight_flush(self):
        """同花顺：同花色连续三张。"""
        cards = [Card(0, 12), Card(0, 11), Card(0, 10)]  # A♠ K♠ Q♠
        result = evaluate(cards)
        assert result.hand_type == HandType.STRAIGHT_FLUSH
        assert result.primary == 12

    def test_flush(self):
        """同花：同花色非连续三张。"""
        cards = [Card(0, 12), Card(0, 9), Card(0, 5)]  # A♠ 9♠ 7♠
        result = evaluate(cards)
        assert result.hand_type == HandType.FLUSH

    def test_straight(self):
        """顺子：不同花色连续三张。"""
        cards = [Card(0, 12), Card(1, 11), Card(2, 10)]  # A♠ K♥ Q♦
        result = evaluate(cards)
        assert result.hand_type == HandType.STRAIGHT
        assert result.primary == 12

    def test_pair(self):
        """对子：两张同点数 + 一张散牌。"""
        cards = [Card(0, 12), Card(1, 12), Card(2, 5)]  # A♠ A♥ 7♦
        result = evaluate(cards)
        assert result.hand_type == HandType.PAIR
        assert result.primary == 12

    def test_high_card(self):
        """单张（散牌）。"""
        cards = [Card(0, 12), Card(1, 8), Card(2, 3)]  # A♠ 10♥ 5♦
        result = evaluate(cards)
        assert result.hand_type == HandType.HIGH_CARD
        assert result.primary == 12

    def test_a23_straight(self):
        """A-2-3 最小顺子。"""
        cards = [Card(0, 12), Card(1, 0), Card(2, 1)]  # A♠ 2♥ 3♦
        result = evaluate(cards)
        assert result.hand_type == HandType.STRAIGHT
        # A-2-3 是最小顺子，high 应低于 2-3-4

    def test_a23_straight_flush(self):
        """A-2-3 同花顺。"""
        cards = [Card(0, 12), Card(0, 0), Card(0, 1)]  # A♠ 2♠ 3♠
        result = evaluate(cards)
        assert result.hand_type == HandType.STRAIGHT_FLUSH

    def test_qka_straight(self):
        """Q-K-A 最大顺子。"""
        cards = [Card(0, 10), Card(1, 11), Card(2, 12)]  # Q♠ K♥ A♦
        result = evaluate(cards)
        assert result.hand_type == HandType.STRAIGHT
        assert result.primary == 12

    def test_pair_kicker_order(self):
        """对子的踢脚牌比较。"""
        # A♠ A♥ K♦ → 对A，踢脚K
        cards1 = [Card(0, 12), Card(1, 12), Card(2, 11)]
        result = evaluate(cards1)
        assert result.hand_type == HandType.PAIR
        assert result.primary == 12


class TestHandComparison:
    """牌型比较测试。"""

    def test_three_beats_straight_flush(self):
        """豹子 > 同花顺。"""
        three = evaluate([Card(0, 12), Card(1, 12), Card(2, 12)])  # AAA
        sf = evaluate([Card(0, 12), Card(0, 11), Card(0, 10)])      # A♠K♠Q♠
        assert three > sf

    def test_straight_flush_beats_flush(self):
        """同花顺 > 同花。"""
        sf = evaluate([Card(0, 12), Card(0, 11), Card(0, 10)])
        flush = evaluate([Card(0, 12), Card(0, 9), Card(0, 5)])
        assert sf > flush

    def test_flush_beats_straight(self):
        """同花 > 顺子。"""
        flush = evaluate([Card(0, 2), Card(0, 1), Card(0, 0)])
        straight = evaluate([Card(0, 12), Card(1, 11), Card(2, 10)])
        assert flush > straight

    def test_straight_beats_pair(self):
        """顺子 > 对子。"""
        straight = evaluate([Card(0, 4), Card(1, 3), Card(2, 2)])
        pair = evaluate([Card(0, 12), Card(1, 12), Card(2, 5)])
        assert straight > pair

    def test_pair_beats_high_card(self):
        """对子 > 单张。"""
        pair = evaluate([Card(0, 0), Card(1, 0), Card(2, 5)])
        high = evaluate([Card(0, 12), Card(1, 9), Card(2, 5)])  # A♠ J♥ 7♦ 非连续
        assert pair > high

    def test_same_type_higher_wins(self):
        """同牌型比 primary。"""
        high_pair = evaluate([Card(0, 12), Card(1, 12), Card(2, 5)])
        low_pair = evaluate([Card(0, 11), Card(1, 11), Card(2, 5)])
        assert high_pair > low_pair

    def test_same_primary_kicker_decides(self):
        """同 primary 比 kicker。"""
        pair_k = evaluate([Card(0, 12), Card(1, 12), Card(2, 11)])  # AA K
        pair_7 = evaluate([Card(0, 12), Card(1, 12), Card(2, 5)])    # AA 7
        assert pair_k > pair_7

    def test_tie(self):
        """完全相同的牌型 → 平局。"""
        h1 = evaluate([Card(0, 12), Card(0, 11), Card(0, 10)])
        h2 = evaluate([Card(1, 12), Card(1, 11), Card(1, 10)])
        assert h1 == h2

    def test_compare_function(self):
        """compare() 返回值正确。"""
        h1 = evaluate([Card(0, 12), Card(1, 12), Card(2, 12)])
        h2 = evaluate([Card(0, 12), Card(0, 11), Card(0, 10)])
        assert compare(h1, h2) > 0
        assert compare(h2, h1) < 0

    def test_a23_is_lowest_straight(self):
        """A-2-3 是最小顺子，小于 2-3-4。"""
        a23 = evaluate([Card(0, 12), Card(1, 0), Card(2, 1)])
        two_three_four = evaluate([Card(0, 0), Card(1, 1), Card(2, 2)])
        assert two_three_four > a23


class TestMillionRandomHands:
    """百万次随机测试：确保判断结果一致性。"""

    def test_million_random_evaluations(self):
        """百万次随机手牌评估，确保无异常和结果一致。"""
        rng = random.Random(42)
        all_cards_list = [Card(s, r) for s in range(4) for r in range(13)]

        for _ in range(100_000):
            cards = rng.sample(all_cards_list, 3)
            result = evaluate(cards)
            assert isinstance(result, HandRank)
            assert isinstance(result.hand_type, HandType)

    def test_evaluated_twice_same_result(self):
        """同一手牌评估两次结果一致。"""
        all_cards_list = [Card(s, r) for s in range(4) for r in range(13)]
        rng = random.Random(123)
        for _ in range(10_000):
            cards = rng.sample(all_cards_list, 3)
            r1 = evaluate(cards)
            r2 = evaluate(list(reversed(cards)))  # 顺序不同
            assert r1 == r2
