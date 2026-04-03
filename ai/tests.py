"""AI 训练框架单元测试。"""

import pytest
import torch

from engine.game import Game
from ai.model import ZhaJinHuaNet, POLICY_DIM
from ai.features import encode_observation, FEATURE_DIM
from ai.agent import Agent
from ai.ppo_trainer import PPOTrainer, RolloutBuffer, Experience
from ai.self_play import SelfPlayEnv
from ai.opponent_pool import OpponentPool


class TestModel:
    """网络模型测试。"""

    def test_model_creation(self):
        model = ZhaJinHuaNet()
        assert model.input_dim == FEATURE_DIM
        assert model.policy_dim == POLICY_DIM

    def test_forward_pass(self):
        model = ZhaJinHuaNet()
        x = torch.randn(4, FEATURE_DIM)
        logits, value = model(x)
        assert logits.shape == (4, POLICY_DIM)
        assert value.shape == (4, 1)

    def test_single_input(self):
        model = ZhaJinHuaNet()
        x = torch.randn(FEATURE_DIM)
        logits, value = model(x)
        assert logits.shape == (POLICY_DIM,)
        assert value.shape == (1,)

    def test_value_range(self):
        model = ZhaJinHuaNet()
        x = torch.randn(100, FEATURE_DIM)
        _, value = model(x)
        assert value.min() >= -1.0
        assert value.max() <= 1.0

    def test_parameter_count(self):
        model = ZhaJinHuaNet()
        params = sum(p.numel() for p in model.parameters())
        assert params < 500000  # 轻量级


class TestFeatures:
    """特征编码测试。"""

    def test_encode_observation(self):
        game = Game(num_players=4)
        game.start()
        obs = game.get_observation(0)
        feat = encode_observation(obs)
        assert feat.shape == (FEATURE_DIM,)
        assert feat.dtype == torch.float32

    def test_dark_cards_zero(self):
        """未看牌时手牌编码全零。"""
        game = Game(num_players=4)
        game.start()
        obs = game.get_observation(0)
        feat = encode_observation(obs)
        # 前 156 维是手牌，未看牌时全零
        assert (feat[:156] == 0).all()

    def test_looked_cards_nonzero(self):
        """看牌后手牌编码非零。"""
        from engine.actions import Action, ActionType
        game = Game(num_players=4)
        game.start()
        game.step(Action(ActionType.LOOK))  # P0 看牌
        obs = game.get_observation(0)
        feat = encode_observation(obs)
        # 看牌后前 156 维应该有非零值
        assert (feat[:156] != 0).any()

    def test_encode_batch(self):
        from ai.features import encode_batch
        game = Game(num_players=4)
        game.start()
        obs1 = game.get_observation(0)
        obs2 = game.get_observation(1)
        batch = encode_batch([obs1, obs2])
        assert batch.shape == (2, FEATURE_DIM)


class TestAgent:
    """Agent 推理测试。"""

    def test_agent_act(self):
        model = ZhaJinHuaNet()
        agent = Agent(model=model)
        game = Game(num_players=4)
        game.start()
        obs = game.get_observation(0)
        actions = game.get_valid_actions(0)
        action = agent.act(obs, actions)
        assert action is not None

    def test_agent_greedy(self):
        model = ZhaJinHuaNet()
        agent = Agent(model=model, epsilon=0.0)  # 纯 greedy
        game = Game(num_players=4)
        game.start()
        obs = game.get_observation(0)
        actions = game.get_valid_actions(0)
        action = agent.act(obs, actions)
        assert action is not None

    def test_agent_epsilon_explore(self):
        model = ZhaJinHuaNet()
        agent = Agent(model=model, epsilon=1.0)  # 纯随机
        game = Game(num_players=4)
        game.start()
        obs = game.get_observation(0)
        actions = game.get_valid_actions(0)
        # 运行多次，应该有不同的动作
        results = set()
        for _ in range(20):
            action = agent.act(obs, actions)
            results.add(action.action_type)
        assert len(results) > 1  # 应该有探索性

    def test_agent_batch(self):
        model = ZhaJinHuaNet()
        agent = Agent(model=model)
        game = Game(num_players=4)
        game.start()
        obs_list = [game.get_observation(i) for i in range(4)]
        actions_list = [game.get_valid_actions(i) for i in range(4)]
        results = agent.act_batch(obs_list, actions_list)
        assert len(results) == 4

    def test_compare_action_index(self):
        """COMPARE 应该映射到 index 8，不与 FOLD(0) 冲突。"""
        from engine.actions import Action, ActionType
        idx = Agent._action_to_index(Action(ActionType.COMPARE, target=1))
        assert idx == 8
        assert idx != Agent._action_to_index(Action(ActionType.FOLD))


class TestPPOTrainer:
    """PPO 训练器测试。"""

    def test_buffer_operations(self):
        buffer = RolloutBuffer()
        exp = Experience(
            state=torch.randn(FEATURE_DIM),
            action_index=0,
            log_prob=-1.0,
            reward=0.5,
            value=0.0,
            done=True,
        )
        buffer.push(exp)
        assert len(buffer) == 1

    def test_buffer_clear(self):
        buffer = RolloutBuffer()
        for i in range(10):
            buffer.push(Experience(
                state=torch.randn(FEATURE_DIM),
                action_index=0, log_prob=0, reward=float(i),
                value=0.0, done=(i == 9),
            ))
        assert len(buffer) == 10
        buffer.clear()
        assert len(buffer) == 0

    def test_gae_computation(self):
        """GAE 计算应该在 done=True 时正确 reset。"""
        buffer = RolloutBuffer()
        # 模拟一个 5 步 episode
        for i in range(5):
            buffer.push(Experience(
                state=torch.randn(FEATURE_DIM),
                action_index=1,
                log_prob=-0.5,
                reward=0.0,
                value=0.1 * i,
                done=(i == 4),
            ))
        advantages, returns = buffer.compute_gae()
        assert advantages.shape == (5,)
        assert returns.shape == (5,)

    def test_train_on_buffer(self):
        model = ZhaJinHuaNet()
        trainer = PPOTrainer(model, mini_batch_size=16, ppo_epochs=2)

        # 收集一批经验
        for _ in range(64):
            trainer.buffer.push(Experience(
                state=torch.randn(FEATURE_DIM),
                action_index=1,
                log_prob=-1.0,
                reward=0.5,
                value=0.0,
                done=True,
            ))

        metrics = trainer.train_on_buffer()
        assert metrics is not None
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        # buffer 应该被清空
        assert len(trainer.buffer) == 0


class TestSelfPlay:
    """自博弈测试。"""

    def test_run_episode(self):
        model = ZhaJinHuaNet()
        agent = Agent(model=model, epsilon=0.5)
        env = SelfPlayEnv(agent=agent, num_players=3)
        experiences = env.run_episode()
        assert len(experiences) > 0
        for exp in experiences:
            assert exp.state.shape == (FEATURE_DIM,)
            assert -2.0 <= exp.reward <= 2.0

    def test_done_marking(self):
        """只有最后一个 step 应该 done=True。"""
        model = ZhaJinHuaNet()
        agent = Agent(model=model, epsilon=0.5)
        env = SelfPlayEnv(agent=agent, num_players=3)
        experiences = env.run_episode()
        assert len(experiences) > 0
        # 中间步骤 done=False，最后一步 done=True
        for i, exp in enumerate(experiences):
            if i < len(experiences) - 1:
                assert exp.done is False, f"Step {i} should not be done"
                assert exp.reward == 0.0, f"Step {i} should have 0 reward"
            else:
                assert exp.done is True, "Last step should be done"
                assert exp.reward != 0.0, "Last step should have non-zero reward"

    def test_run_batch(self):
        model = ZhaJinHuaNet()
        agent = Agent(model=model, epsilon=0.5)
        env = SelfPlayEnv(agent=agent, num_players=3)
        experiences = env.run_batch(num_episodes=3)
        assert len(experiences) > 0

    def test_evaluate_vs_random(self):
        model = ZhaJinHuaNet()
        agent = Agent(model=model, epsilon=0.0)
        env = SelfPlayEnv(agent=agent, num_players=3)
        result = env.evaluate_vs_random(num_episodes=20)
        assert "win_rate" in result
        assert 0.0 <= result["win_rate"] <= 1.0


class TestOpponentPool:
    """对手池测试。"""

    def test_add_and_sample(self, tmp_path):
        pool = OpponentPool(max_size=5, save_dir=str(tmp_path / "opponents"))
        model = ZhaJinHuaNet()
        pool.add(model, elo=1000.0)
        assert pool.size == 1

        sampled = pool.sample()
        assert sampled is not None

    def test_max_size(self, tmp_path):
        pool = OpponentPool(max_size=3, save_dir=str(tmp_path / "opponents2"))
        model = ZhaJinHuaNet()
        for i in range(5):
            pool.add(model, elo=1000.0 + i)
        assert pool.size == 3

    def test_empty_sample(self, tmp_path):
        pool = OpponentPool(save_dir=str(tmp_path / "empty"))
        assert pool.sample() is None
