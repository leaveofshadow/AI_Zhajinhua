"""并行自博弈环境。"""

from __future__ import annotations

import random
from typing import List

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
        """运行一局自博弈，收集经验。"""
        game = Game(
            num_players=self.num_players,
            initial_chips=self.initial_chips,
            min_bet=self.min_bet,
        )
        game.start()

        experiences: List[Experience] = []
        step_records: List[dict] = []  # 暂存每步信息

        for _ in range(500):
            if game.is_finished():
                break

            pid = game.state.current_player_idx
            obs = game.get_observation(pid)
            valid_actions = game.get_valid_actions(pid)

            if not valid_actions:
                break

            # 策略推理
            features = torch.tensor([], dtype=torch.float32)  # placeholder
            from .features import encode_observation
            features = encode_observation(obs)

            with torch.no_grad():
                logits, value = self.agent.model(features.unsqueeze(0))
                logits = logits.squeeze(0)

            # 屏蔽非法动作
            valid_indices = [Agent._action_to_index(a) for a in valid_actions]
            mask = torch.full_like(logits, float("-inf"))
            for idx in valid_indices:
                mask[idx] = 0.0
            masked_logits = logits + mask

            # 采样动作（训练时需要随机性）
            dist = torch.distributions.Categorical(logits=masked_logits)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx)).item()

            # 映射回 Action
            action = Agent._index_to_action(action_idx, valid_actions)

            step_records.append({
                "pid": pid,
                "features": features,
                "action_idx": action_idx,
                "log_prob": log_prob,
                "value": value.item(),
            })

            game.step(action)

        # 计算奖励并生成 Experience
        if game.is_finished():
            result = game.get_result()
            for record in step_records:
                pid = record["pid"]
                # 奖励 = 筹码变化归一化
                chip_change = result.chip_changes[pid] if pid < len(result.chip_changes) else 0
                reward = chip_change / self.initial_chips  # 归一化到 [-1, 1] 量级
                reward = max(-1.0, min(1.0, reward))

                # 名次奖励
                rank = result.rankings[pid] if pid < len(result.rankings) else self.num_players - 1
                rank_reward = 1.0 - 2.0 * rank / max(1, self.num_players - 1)
                final_reward = 0.6 * rank_reward + 0.4 * reward

                experiences.append(Experience(
                    state=record["features"],
                    action_index=record["action_idx"],
                    log_prob=record["log_prob"],
                    reward=final_reward,
                    value=record["value"],
                    done=True,  # 所有经验都是局末的
                ))

        return experiences

    def run_batch(self, num_episodes: int) -> List[Experience]:
        """运行多局收集经验。"""
        all_experiences: List[Experience] = []
        for _ in range(num_episodes):
            all_experiences.extend(self.run_episode())
        return all_experiences
