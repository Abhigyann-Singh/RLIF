"""Train the simulation expert and reference value model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from rlif.algorithms.expert import train_expert_pipeline
from rlif.envs.make_env import make_env
from rlif.data.loaders import resolve_offline_dataset
from rlif.utils.config import load_config
from rlif.utils.logging import MetricsLogger
from rlif.utils.normalization import normalized_score
from rlif.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BC expert and reference Q for RLIF")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", type=str, default=None, help="Optional Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional Weights & Biases entity")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional Weights & Biases run name")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        choices=("online", "offline", "disabled"),
        help="W&B tracking mode when --wandb-project is set",
    )
    parser.add_argument("--eval-episodes", type=int, default=5, help="Evaluation episodes for progress logging")
    parser.add_argument("--eval-interval-epochs", type=int, default=5, help="Run expert eval every N BC epochs")
    return parser.parse_args()


def evaluate_actor(actor: torch.nn.Module, env: object, episodes: int, device: str, env_name: str) -> dict[str, float]:
    returns: list[float] = []
    normalized_returns: list[float] = []
    for _ in range(max(1, episodes)):
        observation, _ = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _ = actor.sample(observation_tensor, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action.squeeze(0).detach().cpu().numpy())
            done = bool(terminated or truncated)
            episode_return += float(reward)
        returns.append(episode_return)
        normalized_returns.append(normalized_score(env_name, episode_return))
    return {
        "expert/eval_return": float(np.mean(returns)) if returns else 0.0,
        "expert/eval_normalized_return": float(np.mean(normalized_returns)) if normalized_returns else 0.0,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.rlif.seed)
    print(f"[Expert] using device: {args.device}", flush=True)

    env = make_env(config.env.env_id, config.rlif.seed, config.env.max_episode_steps, deterministic=False)
    eval_env = make_env(config.env.env_id, config.rlif.seed + 1, config.env.max_episode_steps, deterministic=True)
    dataset = resolve_offline_dataset(
        env,
        config.data.expert_demo_path or config.data.offline_path,
        None,
    )
    output_dir = Path(config.data.output_dir) / config.env.name / "expert"

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=None if args.wandb_run_name is None else str(args.wandb_run_name),
                mode=args.wandb_mode,
                config=asdict(config),
                dir=str(output_dir),
            )
        except ImportError:
            print("wandb is not installed; continuing without W&B logging.")
        except Exception as error:  # noqa: BLE001
            if args.wandb_mode == "online":
                print(f"wandb online init failed ({error}); retrying in offline mode.")
                try:
                    import wandb

                    wandb_run = wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        name=None if args.wandb_run_name is None else str(args.wandb_run_name),
                        mode="offline",
                        config=asdict(config),
                        dir=str(output_dir),
                    )
                except Exception as offline_error:  # noqa: BLE001
                    print(f"wandb offline fallback failed ({offline_error}); continuing without W&B logging.")
            else:
                print(f"wandb init failed ({error}); continuing without W&B logging.")

    logger = MetricsLogger(output_dir, wandb_run=wandb_run)

    def progress_callback(step: int, metrics: dict[str, float]) -> None:
        logger.log(step, metrics)

    def eval_callback(actor: torch.nn.Module) -> dict[str, float]:
        return evaluate_actor(actor, eval_env, args.eval_episodes, args.device, config.env.name)

    result = train_expert_pipeline(
        dataset=dataset,
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        expert_kind=config.expert.kind,
        expert_hidden_dims=tuple(config.expert.hidden_dims),
        ref_q_hidden_dims=tuple(config.expert.ref_q_hidden_dims),
        expert_lr=config.expert.learning_rate,
        expert_weight_decay=config.expert.weight_decay,
        ref_q_lr=config.expert.learning_rate,
        ref_q_weight_decay=config.expert.weight_decay,
        batch_size=config.expert.batch_size,
        expert_epochs=config.expert.epochs,
        ref_q_epochs=config.expert.ref_q_epochs,
        discount=config.rlpd.discount,
        device=args.device,
        output_dir=output_dir,
        grad_clip_norm=config.expert.grad_clip_norm,
        progress_callback=progress_callback,
        evaluation_fn=eval_callback,
        eval_interval_epochs=args.eval_interval_epochs,
    )
    final_step = config.expert.epochs if config.expert.kind == "offline_rl" else config.expert.epochs + config.expert.ref_q_epochs
    logger.log(final_step, result.metrics)
    logger.finish()
    print(f"Expert saved to {result.actor_path}")
    if result.reference_q_path is not None:
        print(f"Reference Q saved to {result.reference_q_path}")


if __name__ == "__main__":
    main()
