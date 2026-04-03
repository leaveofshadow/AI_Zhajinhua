"""cards.py 测试：Card、Deck 及批量发牌。"""

import pytest
from engine.cards import Card, Deck, all_cards, deal_multiple_hands


class TestCard:
    """Card 类测试。"""

    def test_card_creation(self):
        c = Card(0, 12)  # A♠
        assert c.suit == 0
        assert c.rank == 12
        assert c.id == 12

    def test_card_invalid(self):
        with pytest.raises(ValueError):
            Card(4, 0)
        with pytest.raises(ValueError):
            Card(0, 13)

    def test_card_str_repr(self):
        c = Card(0, 12)  # A♠
        assert str(c) == "A♠"
        c2 = Card(1, 10)  # Q♥
        assert str(c2) == "Q♥"

    def test_card_equality(self):
        c1 = Card(0, 12)
        c2 = Card(0, 12)
        c3 = Card(1, 12)
        assert c1 == c2
        assert c1 != c3

    def test_card_comparison(self):
        # rank 优先比较
        low = Card(0, 0)   # 2♠
        high = Card(0, 12)  # A♠
        assert low < high
        assert high > low
        # 同 rank 比 suit
        c1 = Card(0, 5)  # 7♠
        c2 = Card(3, 5)  # 7♣
        assert c1 < c2

    def test_card_hash(self):
        c1 = Card(0, 12)
        c2 = Card(0, 12)
        assert hash(c1) == hash(c2)
        s = {c1, c2}
        assert len(s) == 1

    def test_card_from_id(self):
        c = Card.from_id(0)
        assert c.suit == 0 and c.rank == 0
        c = Card.from_id(51)
        assert c.suit == 3 and c.rank == 12

    def test_all_52_cards(self):
        cards = all_cards()
        assert len(cards) == 52
        assert len(set(cards)) == 52


class TestDeck:
    """Deck 类测试。"""

    def test_deck_init(self):
        deck = Deck()
        assert deck.remaining == 52

    def test_deck_deal(self):
        deck = Deck()
        hand = deck.deal(3)
        assert len(hand) == 3
        assert deck.remaining == 49

    def test_deck_deal_too_many(self):
        deck = Deck()
        with pytest.raises(ValueError):
            deck.deal(53)

    def test_deck_shuffle_randomness(self):
        deck1 = Deck(seed=42)
        deck2 = Deck(seed=99)
        ids1 = [c.id for c in deck1._cards]
        ids2 = [c.id for c in deck2._cards]
        assert ids1 != ids2

    def test_deck_reset(self):
        deck = Deck()
        deck.deal(10)
        assert deck.remaining == 42
        deck.reset()
        assert deck.remaining == 52

    def test_deal_multiple_hands(self):
        hands = deal_multiple_hands(num_players=4, num_hands=10)
        assert len(hands) == 10
        for hand_set in hands:
            assert len(hand_set) == 4
            for hand in hand_set:
                assert len(hand) == 3
