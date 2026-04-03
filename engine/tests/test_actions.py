"""actions.py 测试：动作验证规则。"""

import pytest
from engine.actions import Action, ActionType, ActionValidator
from engine.game import Game, GamePhase


class TestActionType:
    """ActionType 枚举测试。"""

    def test_action_values(self):
        assert ActionType.FOLD == 0
        assert ActionType.CALL == 1
        assert ActionType.RAISE_2X == 2
        assert ActionType.RAISE_6X == 6
        assert ActionType.LOOK == 7
        assert ActionType.COMPARE == 8

    def test_raise_multiplier(self):
        for mult in range(2, 7):
            action = Action(ActionType(mult))
            assert action.raise_multiplier == mult

    def test_non_raise_multiplier(self):
        action = Action(ActionType.FOLD)
        assert action.raise_multiplier == 0


class TestActionValidation:
    """动作合法性验证。"""

    def _make_game(self, num_players=4, **kwargs):
        game = Game(num_players=num_players, initial_chips=1000, min_bet=10, **kwargs)
        game.start()
        return game

    def test_fold_always_valid(self):
        """弃牌始终合法。"""
        game = self._make_game()
        pid = game.state.current_player_idx
        action = Action(ActionType.FOLD)
        assert ActionValidator.validate(action, pid, game.state)

    def test_call_with_chips(self):
        """有筹码时跟注合法。"""
        game = self._make_game()
        pid = game.state.current_player_idx
        action = Action(ActionType.CALL)
        assert ActionValidator.validate(action, pid, game.state)

    def test_look_only_when_dark(self):
        """只有未看牌时才能看牌。"""
        game = self._make_game()
        pid = game.state.current_player_idx

        # 未看牌时合法
        assert ActionValidator.validate(Action(ActionType.LOOK), pid, game.state)

        # 看牌后不合法
        game.step(Action(ActionType.LOOK))
        assert not ActionValidator.validate(Action(ActionType.LOOK), pid, game.state)

    def test_raise_with_enough_chips(self):
        """筹码足够时加注合法。"""
        game = self._make_game()
        pid = game.state.current_player_idx
        action = Action(ActionType.RAISE_2X)
        assert ActionValidator.validate(action, pid, game.state)

    def test_compare_too_early(self):
        """轮数不足时比牌不合法。"""
        game = self._make_game(min_compare_round=2)
        pid = game.state.current_player_idx
        action = Action(ActionType.COMPARE, target=1)
        assert not ActionValidator.validate(action, pid, game.state)

    def test_compare_self(self):
        """不能和自己比牌。"""
        game = self._make_game()
        pid = game.state.current_player_idx
        action = Action(ActionType.COMPARE, target=pid)
        assert not ActionValidator.validate(action, pid, game.state)

    def test_compare_inactive_player(self):
        """不能和已弃牌玩家比牌。"""
        game = self._make_game()
        # 先让玩家1弃牌
        pid0 = game.state.current_player_idx
        game.step(Action(ActionType.FOLD))  # 玩家0弃牌
        if not game.is_finished():
            pid = game.state.current_player_idx
            action = Action(ActionType.COMPARE, target=pid0)
            assert not ActionValidator.validate(action, pid, game.state)

    def test_get_valid_actions_structure(self):
        """合法动作列表结构正确。"""
        game = self._make_game()
        pid = game.state.current_player_idx
        actions = ActionValidator.get_valid_actions(pid, game.state)
        assert len(actions) > 0
        # 应包含弃牌
        assert any(a.action_type == ActionType.FOLD for a in actions)

    def test_inactive_player_no_actions(self):
        """已弃牌玩家没有合法动作。"""
        game = self._make_game()
        pid = game.state.current_player_idx
        game.step(Action(ActionType.FOLD))
        actions = ActionValidator.get_valid_actions(pid, game.state)
        assert len(actions) == 0

    def test_compare_after_min_rounds(self):
        """达到最低轮数后比牌合法。"""
        game = Game(num_players=3, initial_chips=10000, min_bet=10, min_compare_round=1)
        game.start()

        # 轮一圈让 round_count 增加
        for _ in range(3):
            if game.is_finished():
                break
            cp = game.state.current_player_idx
            game.step(Action(ActionType.CALL))

        if not game.is_finished():
            cp = game.state.current_player_idx
            valid = game.get_valid_actions(cp)
            compare_actions = [a for a in valid if a.action_type == ActionType.COMPARE]
            assert len(compare_actions) > 0
