"""Evaluate trained RLIF or expert checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rlif.algorithms.rlpd import RLPDTrainer
from rlif.envs.make_env import make_env
from rlif.models.actor import GaussianActor
from rlif.utils.checkpoint import load_checkpoint
from rlif.utils.config import load_config
from rlif.utils.normalization import normalized_score
from rlif.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate expert or RLIF checkpoints")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["expert", "learner"], default="learner")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.rlif.seed)

    env = make_env(config.env.env_id, config.rlif.seed, config.env.max_episode_steps, deterministic=True)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = args.device

    if args.mode == "expert":
        model = GaussianActor(observation_dim, action_dim, tuple(config.expert.hidden_dims)).to(device)
        checkpoint = load_checkpoint(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["actor_state_dict"])
        model.eval()
        returns = []
        normalized_returns = []
        for _ in range(args.episodes):
            observation, _ = env.reset()
            done = False
            episode_return = 0.0
            while not done:
                observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _ = model.sample(observation_tensor, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action.squeeze(0).detach().cpu().numpy())
                done = bool(terminated or truncated)
                episode_return += float(reward)
            returns.append(episode_return)
            normalized_returns.append(normalized_score(config.env.name, episode_return))
        print({"eval_return": float(sum(returns) / len(returns)), "eval_normalized_return": float(sum(normalized_returns) / len(normalized_returns))})
        return

    learner = RLPDTrainer(
        observation_dim=observation_dim,
        action_dim=action_dim,
        actor_hidden_dims=tuple(config.rlpd.actor_hidden_dims),
        critic_hidden_dims=tuple(config.rlpd.critic_hidden_dims),
        critic_ensemble_size=config.rlpd.critic_ensemble_size,
        critic_subset_size=config.rlpd.critic_subset_size,
        learning_rate=config.rlpd.learning_rate,
        weight_decay=config.rlpd.weight_decay,
        discount=config.rlpd.discount,
        tau=config.rlpd.tau,
        entropy_backup=config.rlpd.entropy_backup,
        automatic_entropy_tuning=config.rlpd.automatic_entropy_tuning,
        target_entropy_scale=config.rlpd.target_entropy_scale,
        grad_clip_norm=config.rlpd.grad_clip_norm,
        device=device,
    )
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    learner.load_state_dict(checkpoint["learner"])
    learner.actor.eval()
    returns = []
    normalized_returns = []
    for _ in range(args.episodes):
        observation, _ = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            action = learner.act(observation_tensor, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action.squeeze(0).detach().cpu().numpy())
            done = bool(terminated or truncated)
            episode_return += float(reward)
        returns.append(episode_return)
        normalized_returns.append(normalized_score(config.env.name, episode_return))
    print({"eval_return": float(sum(returns) / len(returns)), "eval_normalized_return": float(sum(normalized_returns) / len(normalized_returns))})


if __name__ == "__main__":
    main()
