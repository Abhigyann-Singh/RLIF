"""Train RLIF with RLPD as the online learner."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch

from rlif.agents.oracle import InterventionOracle
from rlif.algorithms.rlif import RLIFRunner
from rlif.algorithms.rlpd import RLPDTrainer
from rlif.data.loaders import resolve_offline_dataset
from rlif.data.replay_buffer import ReplayBuffer
from rlif.envs.make_env import make_env
from rlif.models.actor import GaussianActor
from rlif.models.critic import CriticEnsemble, QNetwork
from rlif.utils.checkpoint import load_checkpoint, save_checkpoint
from rlif.utils.config import load_config
from rlif.utils.logging import MetricsLogger
from rlif.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RLIF on top of RLPD")
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
    parser.add_argument("--resume-from", type=str, default=None, help="Optional checkpoint to resume RLIF training from")
    return parser.parse_args()


def _load_expert_actor(path: Path, observation_dim: int, action_dim: int, hidden_dims: tuple[int, ...], device: str) -> GaussianActor:
    actor = GaussianActor(observation_dim, action_dim, hidden_dims).to(device)
    checkpoint = load_checkpoint(path, map_location=device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor


def _load_reference_q(path: Path, observation_dim: int, action_dim: int, hidden_dims: tuple[int, ...], device: str) -> QNetwork | CriticEnsemble:
    checkpoint = load_checkpoint(path, map_location=device)
    if "critic_state_dict" in checkpoint:
        ensemble_size = int(checkpoint.get("critic_ensemble_size", 2))
        critic_hidden_dims = tuple(checkpoint.get("critic_hidden_dims", hidden_dims))
        critic = CriticEnsemble(observation_dim, action_dim, critic_hidden_dims, ensemble_size).to(device)
        critic.load_state_dict(checkpoint["critic_state_dict"])
        critic.eval()
        return critic

    q = QNetwork(observation_dim, action_dim, hidden_dims).to(device)
    q.load_state_dict(checkpoint["q_state_dict"])
    q.eval()
    return q


def _restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])  # type: ignore[arg-type]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.rlif.seed)

    print("[RLIF] building environments", flush=True)
    env = make_env(config.env.env_id, config.rlif.seed, config.env.max_episode_steps, deterministic=False)
    eval_env = make_env(config.env.env_id, config.rlif.seed + 1, config.env.max_episode_steps, deterministic=config.env.deterministic_eval)

    print("[RLIF] loading offline dataset", flush=True)
    dataset = resolve_offline_dataset(
        env,
        config.data.offline_path or config.data.expert_demo_path,
        None,
        trajectory_limit=config.data.offline_trajectory_limit or None,
    )

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = args.device
    print(f"[RLIF] using device: {device}", flush=True)
    output_dir = Path(config.data.output_dir) / config.env.name / "rlif"

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
        except Exception as error:  # noqa: BLE001 - W&B can fail for auth/permission/network reasons
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
                except Exception as offline_error:  # noqa: BLE001 - keep RLIF running if logging fails
                    print(f"wandb offline fallback failed ({offline_error}); continuing without W&B logging.")
            else:
                print(f"wandb init failed ({error}); continuing without W&B logging.")

    logger = MetricsLogger(output_dir, wandb_run=wandb_run)

    print("[RLIF] filling replay buffers", flush=True)
    offline_buffer = ReplayBuffer(observation_dim, action_dim, config.rlif.offline_buffer_size, device=device)
    if config.rlif.zero_offline_rewards:
        dataset = {**dataset, "rewards": np.zeros_like(dataset["rewards"], dtype=np.float32)}
    offline_buffer.add_batch(dataset, source=1.0)
    online_buffer = ReplayBuffer(observation_dim, action_dim, config.rlif.online_buffer_size, device=device)

    print("[RLIF] loading expert checkpoints", flush=True)
    expert_dir = Path(config.data.output_dir) / config.env.name / "expert"
    expert_actor = _load_expert_actor(expert_dir / "expert_actor.pt", observation_dim, action_dim, tuple(config.expert.hidden_dims), device)
    reference_q = None
    if config.rlif.intervention_mode == "value":
        reference_q = _load_reference_q(expert_dir / "expert_reference_q.pt", observation_dim, action_dim, tuple(config.expert.ref_q_hidden_dims), device)

    print("[RLIF] constructing learner and oracle", flush=True)
    oracle = InterventionOracle(
        mode=config.rlif.intervention_mode,
        beta=config.rlif.beta,
        alpha=config.rlif.alpha,
        delta=config.rlif.delta,
        relative_threshold=config.rlif.relative_threshold,
        expert_actor=expert_actor,
        reference_q=reference_q,
        device=device,
    )

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

    runner = RLIFRunner(
        env=env,
        eval_env=eval_env,
        learner=learner,
        oracle=oracle,
        online_buffer=online_buffer,
        offline_buffer=offline_buffer,
        takeover_steps=config.rlif.takeover_steps,
        warmup_steps=config.rlif.warmup_steps,
        utd_ratio=config.rlpd.utd_ratio,
        batch_size=config.rlpd.batch_size,
        offline_ratio=config.rlpd.offline_ratio,
        device=device,
        env_name=config.env.name,
        debug_interval_updates=config.rlif.debug_interval_updates,
        debug_preview_size=config.rlif.debug_preview_size,
        debug_snapshot_path=output_dir / "replay_debug.jsonl",
    )

    resume_path = args.resume_from or config.rlif.resume_from or None
    resume_state = None
    if resume_path:
        checkpoint = load_checkpoint(resume_path, map_location=device)
        learner.load_state_dict(checkpoint["learner"])
        if "online_buffer" in checkpoint:
            online_buffer.load_state_dict(checkpoint["online_buffer"])
        resume_state = checkpoint.get("train_state")
        if checkpoint.get("rng_state") is not None:
            _restore_rng_state(checkpoint["rng_state"])
        print(f"[RLIF] resumed from {resume_path}", flush=True)

    print("[RLIF] starting training loop", flush=True)
    history = runner.train(
        pretrain_epochs=config.rlif.pretrain_epochs,
        pretrain_train_steps_per_epoch=config.rlif.pretrain_train_steps_per_epoch,
        rounds=config.rlif.rounds,
        trajectories_per_round=config.rlif.trajectories_per_round,
        round_train_epochs=config.rlif.round_train_epochs,
        round_train_steps_per_epoch=config.rlif.round_train_steps_per_epoch,
        eval_interval_rounds=config.rlif.eval_interval_rounds,
        checkpoint_interval_rounds=config.rlif.checkpoint_interval_rounds,
        eval_episodes=config.rlif.eval_episodes,
        output_dir=output_dir,
        logger=logger,
        resume_state=resume_state,
    )
    if history:
        logger.log(len(history), history[-1])
    logger.finish()
    final_checkpoint = {"learner": learner.state_dict(), "config": config}
    latest_checkpoint_path = output_dir / "latest.pt"
    if latest_checkpoint_path.exists():
        latest_checkpoint = load_checkpoint(latest_checkpoint_path, map_location="cpu")
        final_checkpoint.update({key: value for key, value in latest_checkpoint.items() if key != "config"})
    save_checkpoint(output_dir / "final.pt", final_checkpoint)
    print(f"RLIF checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
