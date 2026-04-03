"""训练入口脚本。

用法:
    python -m ai.train --episodes 10000 --eval-interval 1000
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import torch

from .agent import Agent
from .model import ZhaJinHuaNet
from .opponent_pool import OpponentPool
from .ppo_trainer import PPOTrainer
from .self_play import SelfPlayEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="炸金花 AI 训练")
    parser.add_argument("--episodes", type=int, default=50000, help="总训练局数")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="熵系数")
    parser.add_argument("--eval-interval", type=int, default=1000, help="评估间隔(局)")
    parser.add_argument("--save-dir", type=str, default="data/models", help="模型保存目录")
    parser.add_argument("--num-players", type=int, default=4, help="玩家人数")
    parser.add_argument("--epsilon-start", type=float, default=0.3, help="初始探索率")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="最终探索率")
    parser.add_argument("--device", type=str, default="auto", help="训练设备")
    parser.add_argument("--log-dir", type=str, default="data/logs", help="TensorBoard 日志目录")
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def train(args: argparse.Namespace) -> None:
    """主训练循环。"""
    device = get_device(args.device)
    print(f"Training on: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 初始化模型
    model = ZhaJinHuaNet().to(device)
    agent = Agent(model=model, device=str(device), epsilon=args.epsilon_start)
    trainer = PPOTrainer(
        model=model,
        lr=args.lr,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        device=str(device),
    )
    env = SelfPlayEnv(agent=agent, num_players=args.num_players)
    pool = OpponentPool(save_dir=os.path.join(args.save_dir, "opponents"))

    # TensorBoard (可选)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.log_dir)
        print(f"TensorBoard logging to: {args.log_dir}")
    except ImportError:
        print("TensorBoard not available, skipping logging")

    print(f"Starting training: {args.episodes} episodes, {args.num_players} players")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    start_time = time.time()
    total_steps = 0

    for episode in range(1, args.episodes + 1):
        # 线性衰减探索率
        progress = episode / args.episodes
        epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * progress
        agent.set_epsilon(epsilon)

        # 收集经验
        experiences = env.run_episode()
        trainer.buffer.extend(experiences)
        total_steps += len(experiences)

        # 训练
        if len(trainer.buffer) >= args.batch_size:
            metrics = trainer.train_step(batch_size=args.batch_size)
            if metrics and writer:
                writer.add_scalar("loss/policy", metrics["policy_loss"], episode)
                writer.add_scalar("loss/value", metrics["value_loss"], episode)
                writer.add_scalar("loss/entropy", metrics["entropy"], episode)
                writer.add_scalar("train/epsilon", epsilon, episode)

        # 定期评估和保存
        if episode % args.eval_interval == 0:
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed

            # 保存 checkpoint
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_ep{episode}.pt")
            trainer.save(ckpt_path)

            # 更新对手池
            pool.add(model, win_rate=0.0)

            print(
                f"[Episode {episode}/{args.episodes}] "
                f"eps/s={eps_per_sec:.1f} "
                f"buffer={len(trainer.buffer)} "
                f"epsilon={epsilon:.3f} "
                f"saved={ckpt_path}"
            )

            if writer:
                writer.add_scalar("train/episodes_per_sec", eps_per_sec, episode)
                writer.add_scalar("train/buffer_size", len(trainer.buffer), episode)

    # 保存最终模型
    final_path = os.path.join(args.save_dir, "final_model.pt")
    trainer.save(final_path)
    print(f"Training complete. Final model saved to: {final_path}")
    print(f"Total time: {time.time() - start_time:.1f}s")

    if writer:
        writer.close()


if __name__ == "__main__":
    train(parse_args())
