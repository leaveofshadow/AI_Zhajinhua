"""AI 训练框架单元测试。"""

import pytest
import torch

from engine.game import Game
from ai.model import ZhaJinHuaNet, POLICY_DIM
from ai.features import encode_observation, FEATURE_DIM
from ai.agent import Agent
from ai.ppo_trainer import PPOTrainer, ReplayBuffer, Experience
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


class TestPPOTrainer:
    """PPO 训练器测试。"""

    def test_buffer_operations(self):
        buffer = ReplayBuffer(capacity=100)
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

    def test_buffer_capacity(self):
        buffer = ReplayBuffer(capacity=5)
        for i in range(10):
            buffer.push(Experience(
                state=torch.randn(FEATURE_DIM),
                action_index=0, log_prob=0, reward=0, value=0, done=True
            ))
        assert len(buffer) == 5

    def test_train_step(self):
        model = ZhaJinHuaNet()
        trainer = PPOTrainer(model)

        # 手动添加一些经验
        for _ in range(32):
            trainer.buffer.push(Experience(
                state=torch.randn(FEATURE_DIM),
                action_index=0,
                log_prob=-1.0,
                reward=0.5,
                value=0.0,
                done=True,
            ))

        metrics = trainer.train_step(batch_size=32)
        assert metrics is not None
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics


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
            assert -1.0 <= exp.reward <= 1.0

    def test_run_batch(self):
        model = ZhaJinHuaNet()
        agent = Agent(model=model, epsilon=0.5)
        env = SelfPlayEnv(agent=agent, num_players=3)
        experiences = env.run_batch(num_episodes=3)
        assert len(experiences) > 0


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
