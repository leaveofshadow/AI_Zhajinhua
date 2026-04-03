"""game.py 测试：牌局状态机完整模拟。"""

import random
import pytest
from engine.actions import Action, ActionType
from engine.cards import Card
from engine.game import Game, GamePhase, VectorizedGameEnv


class TestGameBasics:
    """基本牌局流程测试。"""

    def test_game_init(self):
        game = Game(num_players=4)
        assert game.state.num_players == 4

    def test_game_start(self):
        game = Game(num_players=4)
        game.start()
        assert game.state.phase == GamePhase.PLAYING
        assert game.state.pot > 0
        for p in game.state.players:
            assert len(p.cards) == 3
            assert p.chips < 1000  # 已扣底注

    def test_invalid_player_count(self):
        with pytest.raises(ValueError):
            Game(num_players=2)
        with pytest.raises(ValueError):
            Game(num_players=7)

    def test_step_not_started(self):
        game = Game(num_players=4)
        with pytest.raises(RuntimeError):
            game.step(Action(ActionType.FOLD))


class TestGameActions:
    """动作执行测试。"""

    def _make_started_game(self, num_players=4):
        game = Game(num_players=num_players, initial_chips=1000, min_bet=10)
        game.start()
        return game

    def test_fold(self):
        game = self._make_started_game()
        pid = game.state.current_player_idx
        game.step(Action(ActionType.FOLD))
        assert not game.state.players[pid].is_active

    def test_look(self):
        game = self._make_started_game()
        pid = game.state.current_player_idx
        assert not game.state.players[pid].has_looked
        game.step(Action(ActionType.LOOK))
        assert game.state.players[pid].has_looked

    def test_call_dark(self):
        """暗牌跟注 = 明注的一半。"""
        game = self._make_started_game()
        pid = game.state.current_player_idx
        chips_before = game.state.players[pid].chips
        game.step(Action(ActionType.CALL))
        spent = chips_before - game.state.players[pid].chips
        assert spent == game.state.min_bet // 2  # 暗牌半注

    def test_call_light(self):
        """明牌跟注 = 全额。"""
        game = self._make_started_game()
        pid = game.state.current_player_idx
        game.step(Action(ActionType.LOOK))
        # LOOK 后游戏推进到下一个玩家，需要等到 pid 再次轮到
        # 直接验证 _call_amount 逻辑
        from engine.actions import ActionValidator
        amount = ActionValidator._call_amount(game.state.players[pid], game.state)
        assert amount == game.state.min_bet  # 明牌全额

    def test_raise(self):
        """加注。"""
        game = self._make_started_game()
        pid = game.state.current_player_idx
        chips_before = game.state.players[pid].chips
        game.step(Action(ActionType.RAISE_2X))
        spent = chips_before - game.state.players[pid].chips
        assert spent == game.state.min_bet * 2

    def test_compare(self):
        """比牌流程。"""
        game = self._make_started_game()
        pid = game.state.current_player_idx
        # 前2轮只跟注，第3轮比牌
        for _ in range(game.state.num_players * 2):
            if game.is_finished():
                break
            cp = game.state.current_player_idx
            game.step(Action(ActionType.CALL))

        if not game.is_finished():
            cp = game.state.current_player_idx
            # 找一个活跃对手
            for i in range(game.state.num_players):
                if i != cp and game.state.players[i].is_active:
                    game.step(Action(ActionType.COMPARE, target=i))
                    break

    def test_compare_too_early(self):
        """轮数不够时不能比牌。"""
        game = self._make_started_game()
        pid = game.state.current_player_idx
        # 找另一个活跃玩家
        for i in range(game.state.num_players):
            if i != pid and game.state.players[i].is_active:
                action = Action(ActionType.COMPARE, target=i)
                valid = game.get_valid_actions(pid)
                compare_actions = [a for a in valid if a.action_type == ActionType.COMPARE]
                # 前2轮不应有比牌选项
                assert len(compare_actions) == 0
                break

    def test_look_not_twice(self):
        """不能看牌两次。"""
        game = self._make_started_game()
        pid = game.state.current_player_idx
        game.step(Action(ActionType.LOOK))
        # 下次轮到该玩家时不能再看牌
        valid = game.get_valid_actions(pid)
        look_actions = [a for a in valid if a.action_type == ActionType.LOOK]
        assert len(look_actions) == 0


class TestGameScenarios:
    """完整牌局场景测试。"""

    def test_fold_to_winner(self):
        """所有人弃牌只剩一人。"""
        game = Game(num_players=3, initial_chips=1000, min_bet=10)
        game.start()

        # 第一个人弃牌
        game.step(Action(ActionType.FOLD))
        # 第二个人弃牌
        if not game.is_finished():
            game.step(Action(ActionType.FOLD))

        assert game.is_finished()
        result = game.get_result()
        active = [i for i, p in enumerate(game.state.players) if p.is_active]
        assert len(active) == 1
        assert result.winner == active[0]

    def test_full_random_game_3p(self):
        """3人随机策略完整对局。"""
        self._run_random_game(3)

    def test_full_random_game_4p(self):
        """4人随机策略完整对局。"""
        self._run_random_game(4)

    def test_full_random_game_6p(self):
        """6人随机策略完整对局。"""
        self._run_random_game(6)

    def _run_random_game(self, num_players):
        game = Game(num_players=num_players, initial_chips=1000, min_bet=10)
        game.start()
        rng = random.Random(42)

        for _ in range(500):
            if game.is_finished():
                break
            pid = game.state.current_player_idx
            valid = game.get_valid_actions(pid)
            if not valid:
                break
            action = rng.choice(valid)
            game.step(action)

        assert game.is_finished()
        result = game.get_result()
        assert result.winner >= 0

    def test_max_rounds_forced_compare(self):
        """达到最大轮数后强制比牌。"""
        game = Game(num_players=3, initial_chips=10000, min_bet=1, max_rounds=5)
        game.start()
        rng = random.Random(99)

        for _ in range(500):
            if game.is_finished():
                break
            pid = game.state.current_player_idx
            valid = game.get_valid_actions(pid)
            # 只选跟注，不弃牌不比牌，强制打满轮数
            call_actions = [a for a in valid if a.action_type == ActionType.CALL]
            look_actions = [a for a in valid if a.action_type == ActionType.LOOK]
            if call_actions:
                action = call_actions[0]
            elif look_actions:
                action = look_actions[0]
            else:
                action = rng.choice(valid)
            game.step(action)

        assert game.is_finished()


class TestObservation:
    """观察空间测试。"""

    def test_dark_player_no_cards(self):
        """未看牌玩家看不到手牌。"""
        game = Game(num_players=4)
        game.start()
        pid = game.state.current_player_idx
        obs = game.get_observation(pid)
        assert not game.state.players[pid].has_looked
        assert obs["my_cards"] == []
        assert obs["has_looked"] is False

    def test_looked_player_sees_cards(self):
        """看牌后能看到手牌。"""
        game = Game(num_players=4)
        game.start()
        pid = game.state.current_player_idx
        game.step(Action(ActionType.LOOK))
        obs = game.get_observation(pid)
        assert obs["has_looked"] is True
        assert len(obs["my_cards"]) == 3

    def test_observation_structure(self):
        """观察空间结构完整。"""
        game = Game(num_players=4)
        game.start()
        obs = game.get_observation(0)
        required_keys = [
            "my_cards", "my_chips", "has_looked", "pot",
            "current_bet", "round_count", "my_position",
            "active_players", "player_states",
        ]
        for key in required_keys:
            assert key in obs, f"Missing key: {key}"

    def test_player_states_count(self):
        """player_states 包含所有玩家。"""
        game = Game(num_players=4)
        game.start()
        obs = game.get_observation(0)
        assert len(obs["player_states"]) == 4


class TestVectorizedEnv:
    """向量化环境测试。"""

    def test_parallel_games(self):
        env = VectorizedGameEnv(num_envs=10, num_players=4)
        obs_list = env.reset_all()
        assert len(obs_list) == 10

    def test_parallel_random_play(self):
        env = VectorizedGameEnv(num_envs=5, num_players=3)
        env.reset_all()
        rng = random.Random(42)

        for _ in range(200):
            actions = env.get_all_valid_actions()
            if all(len(a) == 0 for a in actions):
                break
            chosen = []
            for valid in actions:
                if valid:
                    chosen.append(rng.choice(valid))
                else:
                    chosen.append(None)
            env.step_all(chosen)

        finished = env.get_finished_env_ids()
        assert len(finished) > 0

    def test_reset_finished_envs(self):
        env = VectorizedGameEnv(num_envs=3, num_players=3)
        env.reset_all()
        rng = random.Random(42)

        # 运行到部分结束
        for _ in range(300):
            if env.get_finished_env_ids():
                break
            actions = env.get_all_valid_actions()
            chosen = [rng.choice(v) if v else None for v in actions]
            env.step_all(chosen)

        finished = env.get_finished_env_ids()
        if finished:
            env.reset_envs(finished)
            # 被重置的环境应该有观察值
            for eid in finished:
                obs = env._games[eid].get_observation(0)
                assert obs is not None

    def test_benchmark(self):
        """性能基准测试：应 >100 局/秒（保守测试）。"""
        env = VectorizedGameEnv(num_envs=1, num_players=4)
        games_per_sec = env.benchmark(num_games=100)
        assert games_per_sec > 100, f"Too slow: {games_per_sec:.1f} games/sec"
