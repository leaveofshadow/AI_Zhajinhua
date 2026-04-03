"""并行自博弈环境。"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import torch

from engine.actions import Action, ActionType
from engine.game import Game
from .agent import Agent
from .ppo_trainer import Experience

__all__ = ["SelfPlayEnv"]


class SelfPlayEnv:
    """自博弈环境：用当前策略进行对局并收集经验。"""

    def __init__(
        self,
        agent: Agent,
        num_players: int = 4,
        initial_chips: int = 1000,
        min_bet: int = 10,
    ) -> None:
        self.agent = agent
        self.num_players = num_players
        self.initial_chips = initial_chips
        self.min_bet = min_bet

    def run_episode(self) -> List[Experience]:
        """运行一局自博弈，收集经验。

        使用 Monte Carlo 方法：所有中间步骤 reward=0, done=False，
        最后一步得到最终奖励, done=True。
        """
        game = Game(
            num_players=self.num_players,
            initial_chips=self.initial_chips,
            min_bet=self.min_bet,
        )
        game.start()

        step_records: List[Dict] = []

        for _ in range(500):
            if game.is_finished():
                break

            pid = game.state.current_player_idx
            obs = game.get_observation(pid)
            valid_actions = game.get_valid_actions(pid)

            if not valid_actions:
                break

            from .features import encode_observation
            features = encode_observation(obs)

            with torch.no_grad():
                logits, value = self.agent.model(features.unsqueeze(0))
                logits = logits.squeeze(0)
                value = value.squeeze(-1).item()

            # 屏蔽非法动作
            valid_indices = [Agent._action_to_index(a) for a in valid_actions]
            mask = torch.full_like(logits, float("-inf"))
            for idx in valid_indices:
                mask[idx] = 0.0
            masked_logits = logits + mask

            # 从策略分布中采样
            dist = torch.distributions.Categorical(logits=masked_logits)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx)).item()

            action = Agent._index_to_action(action_idx, valid_actions)

            step_records.append({
                "pid": pid,
                "features": features,
                "action_idx": action_idx,
                "log_prob": log_prob,
                "value": value,
            })

            game.step(action)

        # 生成 Experience 列表
        experiences: List[Experience] = []
        n_steps = len(step_records)

        if n_steps > 0 and game.is_finished():
            result = game.get_result()

            for i, record in enumerate(step_records):
                pid = record["pid"]
                is_last = (i == n_steps - 1)

                if is_last:
                    # 最后一步：给予最终奖励
                    chip_change = result.chip_changes[pid]
                    reward = max(-1.0, min(1.0, chip_change / self.initial_chips))

                    rank = result.rankings[pid]
                    rank_reward = 1.0 - 2.0 * rank / max(1, self.num_players - 1)
                    final_reward = 0.6 * rank_reward + 0.4 * reward
                else:
                    final_reward = 0.0

                experiences.append(Experience(
                    state=record["features"],
                    action_index=record["action_idx"],
                    log_prob=record["log_prob"],
                    reward=final_reward,
                    value=record["value"],
                    done=is_last,
                ))

        return experiences

    def run_episode_with_opponents(
        self,
        opponent_positions: Set[int],
        opponent_agent: Optional[Agent] = None,
    ) -> tuple[List[Experience], Dict]:
        """与历史版本对手对局，只收集当前策略的经验。

        Args:
            opponent_positions: 使用对手策略的座位号集合
            opponent_agent: 对手 Agent（None 则用随机）

        Returns:
            (experiences, stats) 元组
        """
        game = Game(
            num_players=self.num_players,
            initial_chips=self.initial_chips,
            min_bet=self.min_bet,
        )
        game.start()

        step_records: List[Dict] = []

        for _ in range(500):
            if game.is_finished():
                break

            pid = game.state.current_player_idx
            obs = game.get_observation(pid)
            valid_actions = game.get_valid_actions(pid)

            if not valid_actions:
                break

            from .features import encode_observation
            features = encode_observation(obs)

            # 选择用哪个 agent
            current_agent = opponent_agent if pid in opponent_positions else self.agent

            with torch.no_grad():
                logits, value = current_agent.model(features.unsqueeze(0))
                logits = logits.squeeze(0)
                value = value.squeeze(-1).item()

            valid_indices = [Agent._action_to_index(a) for a in valid_actions]
            mask = torch.full_like(logits, float("-inf"))
            for idx in valid_indices:
                mask[idx] = 0.0
            masked_logits = logits + mask

            dist = torch.distributions.Categorical(logits=masked_logits)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx)).item()

            action = Agent._index_to_action(action_idx, valid_actions)

            # 只记录当前策略（非对手）的经验
            if pid not in opponent_positions:
                step_records.append({
                    "pid": pid,
                    "features": features,
                    "action_idx": action_idx,
                    "log_prob": log_prob,
                    "value": value,
                })

            game.step(action)

        stats: Dict = {"result": "unfinished"}
        if game.is_finished():
            result = game.get_result()
            stats = {
                "winner": result.winner,
                "our_wins": sum(
                    1 for pid in range(self.num_players)
                    if pid not in opponent_positions and result.rankings[pid] == 0
                ),
                "opponent_wins": sum(
                    1 for pid in opponent_positions
                    if result.rankings[pid] == 0
                ),
            }

        # 生成经验
        experiences: List[Experience] = []
        n_steps = len(step_records)

        if n_steps > 0 and game.is_finished():
            result = game.get_result()
            for i, record in enumerate(step_records):
                is_last = (i == n_steps - 1)
                pid = record["pid"]

                if is_last:
                    chip_change = result.chip_changes[pid]
                    reward = max(-1.0, min(1.0, chip_change / self.initial_chips))
                    rank = result.rankings[pid]
                    rank_reward = 1.0 - 2.0 * rank / max(1, self.num_players - 1)
                    final_reward = 0.6 * rank_reward + 0.4 * reward
                else:
                    final_reward = 0.0

                experiences.append(Experience(
                    state=record["features"],
                    action_index=record["action_idx"],
                    log_prob=record["log_prob"],
                    reward=final_reward,
                    value=record["value"],
                    done=is_last,
                ))

        return experiences, stats

    def run_batch(self, num_episodes: int) -> List[Experience]:
        """运行多局收集经验。"""
        all_experiences: List[Experience] = []
        for _ in range(num_episodes):
            all_experiences.extend(self.run_episode())
        return all_experiences

    def evaluate_vs_random(self, num_episodes: int = 100) -> Dict[str, float]:
        """与随机策略对比评估。"""
        wins = 0
        total = 0
        for _ in range(num_episodes):
            game = Game(
                num_players=self.num_players,
                initial_chips=self.initial_chips,
                min_bet=self.min_bet,
            )
            game.start()

            for _ in range(500):
                if game.is_finished():
                    break
                pid = game.state.current_player_idx
                valid = game.get_valid_actions(pid)
                if not valid:
                    break

                if pid == 0:
                    # AI agent (greedy)
                    obs = game.get_observation(pid)
                    action = self.agent.act(obs, valid)
                else:
                    # Random
                    import random
                    action = random.choice(valid)

                game.step(action)

            if game.is_finished():
                result = game.get_result()
                if result.rankings[0] == 0:
                    wins += 1
                total += 1

        win_rate = wins / total if total > 0 else 0.0
        return {"win_rate": win_rate, "wins": wins, "total": total}
