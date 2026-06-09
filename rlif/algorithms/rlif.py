"""RLIF rollout collection and training orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import random
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from rlif.agents.oracle import InterventionOracle
from rlif.algorithms.rlpd import RLPDTrainer
from rlif.data.replay_buffer import Batch, ReplayBuffer
from rlif.utils.logging import JsonlLogger
from rlif.utils.checkpoint import save_checkpoint
from rlif.utils.normalization import normalized_score


@dataclass
class EpisodeMetrics:
    return_: float
    normalized_return: float
    length: int
    interventions: int
    takeover_steps: int


class InterventionLogger:
    def __init__(self) -> None:
        self.previous_transition_index: int | None = None

    def record_transition(self, buffer: ReplayBuffer, current_transition_index: int, intervened: bool) -> bool:
        penalty_assigned = False
        if intervened and self.previous_transition_index is not None:
            buffer.rewards[self.previous_transition_index] = -1.0
            penalty_assigned = True
        self.previous_transition_index = current_transition_index
        return penalty_assigned

    def reset(self) -> None:
        self.previous_transition_index = None


class RLIFRunner:
    def __init__(
        self,
        env: Any,
        eval_env: Any,
        learner: RLPDTrainer,
        oracle: InterventionOracle,
        online_buffer: ReplayBuffer,
        offline_buffer: ReplayBuffer,
        takeover_steps: int,
        warmup_steps: int,
        utd_ratio: int,
        batch_size: int,
        offline_ratio: float,
        device: str,
        env_name: str,
        debug_interval_updates: int = 0,
        debug_preview_size: int = 8,
        debug_snapshot_path: str | Path | None = None,
    ) -> None:
        self.env = env
        self.eval_env = eval_env
        self.learner = learner
        self.oracle = oracle
        self.online_buffer = online_buffer
        self.offline_buffer = offline_buffer
        self.takeover_steps = takeover_steps
        self.warmup_steps = warmup_steps
        self.utd_ratio = utd_ratio
        self.batch_size = batch_size
        # RLPD uses fixed 50/50 online/offline minibatches during online training.
        self.offline_ratio = offline_ratio
        self.device = device
        self.env_name = env_name
        self.intervention_logger = InterventionLogger()
        self._takeover_remaining = 0
        self.debug_interval_updates = max(0, int(debug_interval_updates))
        self.debug_preview_size = max(0, int(debug_preview_size))
        self.debug_writer = None
        if debug_snapshot_path is not None and self.debug_interval_updates > 0:
            self.debug_writer = JsonlLogger(debug_snapshot_path)

    def _replay_composition(self) -> dict[str, float]:
        total = len(self.offline_buffer) + len(self.online_buffer)
        if total == 0:
            return {"offline_fraction": 0.0, "online_fraction": 0.0}
        offline_fraction = float(len(self.offline_buffer)) / float(total)
        return {"offline_fraction": offline_fraction, "online_fraction": 1.0 - offline_fraction}

    def _merge_batches(self, batches: list[Batch]) -> Batch:
        if len(batches) == 1:
            return batches[0]
        return Batch(
            observations=torch.cat([batch.observations for batch in batches], dim=0),
            actions=torch.cat([batch.actions for batch in batches], dim=0),
            rewards=torch.cat([batch.rewards for batch in batches], dim=0),
            next_observations=torch.cat([batch.next_observations for batch in batches], dim=0),
            dones=torch.cat([batch.dones for batch in batches], dim=0),
            source=torch.cat([batch.source for batch in batches], dim=0),
        )

    def _buffer_snapshot(self, buffer: ReplayBuffer) -> dict[str, Any]:
        size = len(buffer)
        if size == 0:
            return {"size": 0}

        rewards = buffer.rewards[:size].reshape(-1)
        dones = buffer.dones[:size].reshape(-1)
        sources = buffer.sources[:size].reshape(-1)
        action_norms = np.linalg.norm(buffer.actions[:size], axis=1)
        preview_size = min(self.debug_preview_size, size)
        start = size - preview_size

        return {
            "size": size,
            "reward_mean": float(np.mean(rewards)),
            "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
            "reward_negative_one_count": int(np.sum(np.isclose(rewards, -1.0))),
            "reward_zero_count": int(np.sum(np.isclose(rewards, 0.0))),
            "done_count": int(np.sum(dones > 0.5)),
            "online_fraction": float(np.mean(sources <= 0.5)),
            "reward_preview": rewards[start:].tolist(),
            "done_preview": dones[start:].tolist(),
            "source_preview": sources[start:].tolist(),
            "action_norm_preview": action_norms[start:].tolist(),
        }

    def _batch_snapshot(self, batch: Batch) -> dict[str, Any]:
        rewards = batch.rewards.detach().cpu().numpy().reshape(-1)
        dones = batch.dones.detach().cpu().numpy().reshape(-1)
        sources = batch.source.detach().cpu().numpy().reshape(-1)
        actions = batch.actions.detach().cpu().numpy()
        preview_size = min(self.debug_preview_size, len(rewards))

        return {
            "size": int(len(rewards)),
            "reward_mean": float(np.mean(rewards)) if len(rewards) > 0 else 0.0,
            "reward_min": float(np.min(rewards)) if len(rewards) > 0 else 0.0,
            "reward_max": float(np.max(rewards)) if len(rewards) > 0 else 0.0,
            "reward_negative_one_count": int(np.sum(np.isclose(rewards, -1.0))),
            "reward_zero_count": int(np.sum(np.isclose(rewards, 0.0))),
            "online_fraction": float(np.mean(sources <= 0.5)) if len(sources) > 0 else 0.0,
            "preview": [
                {
                    "reward": float(rewards[index]),
                    "done": float(dones[index]),
                    "source": float(sources[index]),
                    "action": actions[index].tolist(),
                }
                for index in range(preview_size)
            ],
        }

    def _sample_training_batch(self, offline_only: bool = False) -> Batch:
        if offline_only or len(self.online_buffer) == 0:
            if len(self.offline_buffer) == 0:
                raise ValueError("Cannot sample training data before loading the offline buffer")
            return self.offline_buffer.sample(self.batch_size)

        if len(self.offline_buffer) == 0:
            raise ValueError("RLPD online updates require a non-empty offline dataset")
        if self.batch_size % 2 != 0:
            raise ValueError("RLPD symmetric sampling requires an even batch size")

        half_batch = self.batch_size // 2
        return self._merge_batches(
            [
                self.online_buffer.sample(half_batch),
                self.offline_buffer.sample(half_batch),
            ]
        )

    def _apply_updates(
        self,
        num_updates: int,
        update_step: int,
        logger: Any | None,
        phase: str,
        offline_only: bool = False,
        base_step: int = 0,
    ) -> tuple[int, list[dict[str, float]]]:
        update_metrics: list[dict[str, float]] = []
        for i in range(num_updates):
            batch = self._sample_training_batch(offline_only=offline_only)
            metrics = self.learner.update(batch)
            batch_offline_fraction = float(batch.source.float().mean().item()) if batch.source.numel() > 0 else 0.0
            metrics["batch_offline_fraction"] = batch_offline_fraction
            metrics["batch_online_fraction"] = 1.0 - batch_offline_fraction
            metrics["utd_step"] = float(i)
            metrics["phase"] = phase
            update_step += 1
            # Compute a monotonic logging step aligned to env steps when available
            step_to_log = int(base_step) if phase == "online_rollout" else int(base_step + i)
            if logger is not None:
                logger.log(step_to_log, metrics)
            if (
                self.debug_writer is not None
                and self.debug_interval_updates > 0
                and phase != "pretrain"
                and update_step % self.debug_interval_updates == 0
            ):
                self.debug_writer.log(
                    {
                        "step": step_to_log,
                        "update_step": update_step,
                        "global_env_step": int(base_step),
                        "phase": phase,
                        "replay_buffer": {
                            "offline": self._buffer_snapshot(self.offline_buffer),
                            "online": self._buffer_snapshot(self.online_buffer),
                        },
                        "minibatch": self._batch_snapshot(batch),
                    }
                )
            update_metrics.append(metrics)
        return update_step, update_metrics

    def _apply_online_utd_updates(
        self,
        update_step: int,
        logger: Any | None,
        base_step: int,
    ) -> tuple[int, list[dict[str, float]]]:
        update_metrics: list[dict[str, float]] = []
        last_batch: Batch | None = None
        utd_updates = max(1, int(self.utd_ratio))

        for utd_step in range(utd_updates):
            batch = self._sample_training_batch(offline_only=False)
            last_batch = batch
            metrics = self.learner.update_critic(batch)
            self.learner.soft_update_targets()
            batch_offline_fraction = float(batch.source.float().mean().item()) if batch.source.numel() > 0 else 0.0
            metrics["batch_offline_fraction"] = batch_offline_fraction
            metrics["batch_online_fraction"] = 1.0 - batch_offline_fraction
            metrics["utd_step"] = float(utd_step)
            metrics["actor_updated"] = 0.0
            metrics["target_updated"] = 1.0
            metrics["phase"] = "online_rollout"
            update_step += 1

            if utd_step == utd_updates - 1:
                actor_metrics = self.learner.update_actor(batch)
                metrics.update(actor_metrics)
                metrics["actor_updated"] = 1.0

            if logger is not None:
                logger.log(int(base_step), metrics)
            if (
                self.debug_writer is not None
                and self.debug_interval_updates > 0
                and update_step % self.debug_interval_updates == 0
            ):
                self.debug_writer.log(
                    {
                        "step": int(base_step),
                        "update_step": update_step,
                        "global_env_step": int(base_step),
                        "phase": "online_rollout",
                        "replay_buffer": {
                            "offline": self._buffer_snapshot(self.offline_buffer),
                            "online": self._buffer_snapshot(self.online_buffer),
                        },
                        "minibatch": self._batch_snapshot(batch),
                    }
                )
            update_metrics.append(metrics)

        if last_batch is None:
            raise RuntimeError("Expected at least one online RLPD update")
        return update_step, update_metrics

    def collect_step(self, observation: np.ndarray, global_step: int) -> tuple[np.ndarray, bool, dict[str, Any]]:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        if global_step < self.warmup_steps:
            agent_action = torch.as_tensor(self.env.action_space.sample(), dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            agent_action = self.learner.act(observation_tensor, deterministic=False)
        expert_action = self.oracle.expert_action(observation_tensor, deterministic=True)

        intervention_started = False
        decision = None
        if self._takeover_remaining > 0:
            action_tensor = expert_action
            self._takeover_remaining -= 1
        else:
            decision = self.oracle.decide(observation_tensor, agent_action)
            if decision.intervene:
                intervention_started = True
                action_tensor = expert_action
                self._takeover_remaining = max(self.takeover_steps - 1, 0)
            else:
                action_tensor = agent_action

        action = action_tensor.squeeze(0).detach().cpu().numpy()
        next_observation, env_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        self.online_buffer.add(observation, action, 0.0, next_observation, float(done), source=0.0)
        current_index = len(self.online_buffer) - 1
        penalty_assigned = self.intervention_logger.record_transition(self.online_buffer, current_index, intervention_started)

        if done:
            self._takeover_remaining = 0
            self.intervention_logger.reset()

        return next_observation, done, {
            "env_reward": float(env_reward),
            "intervention": float(intervention_started),
            "takeover": float(action_tensor is expert_action),
            "penalty_assigned": float(penalty_assigned),
            "q_expert": None if decision is None else decision.q_expert,
            "q_agent": None if decision is None else decision.q_agent,
            "info": info,
        }

    def train(
        self,
        pretrain_epochs: int,
        pretrain_train_steps_per_epoch: int,
        rounds: int,
        trajectories_per_round: int,
        round_train_epochs: int,
        round_train_steps_per_epoch: int,
        eval_interval_rounds: int,
        checkpoint_interval_rounds: int,
        eval_episodes: int,
        output_dir: str | Path,
        logger: Any | None = None,
        resume_state: dict[str, Any] | None = None,
    ) -> list[dict[str, float]]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Online RLPD updates are performed immediately after each environment step.
        _ = (round_train_epochs, round_train_steps_per_epoch)

        global_env_step = 0
        update_step = 0
        start_round = 0
        if resume_state is not None:
            global_env_step = int(resume_state.get("global_env_step", 0))
            update_step = int(resume_state.get("global_update_step", 0))
            start_round = int(resume_state.get("round_index", 0))

        metrics_history: list[dict[str, float]] = []
        progress_bar = tqdm(total=rounds, initial=start_round, desc="RLIF rounds", dynamic_ncols=True, unit="round")

        try:
            if start_round == 0 and pretrain_epochs > 0 and pretrain_train_steps_per_epoch > 0:
                for epoch in range(pretrain_epochs):
                    update_step, epoch_metrics = self._apply_updates(
                        pretrain_train_steps_per_epoch,
                        update_step,
                        logger,
                        phase="pretrain",
                        offline_only=True,
                        base_step=global_env_step,
                    )
                    if epoch_metrics:
                        metrics_history.append({**epoch_metrics[-1], "epoch": float(epoch), "stage": 0.0})

            observation, _ = self.env.reset()
            for round_index in range(start_round, rounds):
                episode_env_return = 0.0
                episode_length = 0
                interventions = 0
                takeover_steps = 0
                episodes_collected = 0

                while episodes_collected < trajectories_per_round:
                    observation, done, info = self.collect_step(observation, global_env_step)
                    global_env_step += 1
                    episode_length += 1
                    episode_env_return += float(info["env_reward"])
                    interventions += int(info["intervention"])
                    takeover_steps += int(info["takeover"])

                    update_step, _ = self._apply_online_utd_updates(
                        update_step,
                        logger,
                        base_step=global_env_step,
                    )

                    if done:
                        episode_metrics = EpisodeMetrics(
                            return_=episode_env_return,
                            normalized_return=normalized_score(self.env_name, episode_env_return),
                            length=episode_length,
                            interventions=interventions,
                            takeover_steps=takeover_steps,
                        )
                        episode_log = {
                            "episode_env_return": episode_metrics.return_,
                            "episode_normalized_return": episode_metrics.normalized_return,
                            "episode_length": float(episode_metrics.length),
                            "episode_interventions": float(episode_metrics.interventions),
                            "episode_takeover_steps": float(episode_metrics.takeover_steps),
                            "intervention_rate": float(episode_metrics.interventions) / max(episode_metrics.length, 1),
                        }
                        if logger is not None:
                            logger.log(global_env_step, episode_log)
                        observation, _ = self.env.reset()
                        episode_env_return = 0.0
                        episode_length = 0
                        interventions = 0
                        takeover_steps = 0
                        episodes_collected += 1

                if eval_interval_rounds > 0 and (round_index + 1) % eval_interval_rounds == 0:
                    eval_metrics = self.evaluate(eval_episodes)
                    eval_metrics.update(self._replay_composition())
                    eval_metrics["round_index"] = float(round_index + 1)
                    if logger is not None:
                        logger.log(global_env_step, eval_metrics)
                    metrics_history.append(eval_metrics)

                if checkpoint_interval_rounds > 0 and (round_index + 1) % checkpoint_interval_rounds == 0:
                    save_checkpoint(
                        output_dir / "latest.pt",
                        {
                            "learner": self.learner.state_dict(),
                            "online_buffer": self.online_buffer.state_dict(),
                            "train_state": {
                                "round_index": round_index + 1,
                                "global_env_step": global_env_step,
                                "global_update_step": update_step,
                                "pretrain_done": True,
                            },
                            "rng_state": self._capture_rng_state(),
                        },
                    )
                    save_checkpoint(
                        output_dir / f"checkpoint_round_{round_index + 1}.pt",
                        {
                            "learner": self.learner.state_dict(),
                            "online_buffer": self.online_buffer.state_dict(),
                            "train_state": {
                                "round_index": round_index + 1,
                                "global_env_step": global_env_step,
                                "global_update_step": update_step,
                                "pretrain_done": True,
                            },
                            "rng_state": self._capture_rng_state(),
                        },
                    )

                progress_bar.update(1)
        finally:
            progress_bar.close()

        return metrics_history

    def evaluate(self, episodes: int) -> dict[str, float]:
        returns: list[float] = []
        normalized_returns: list[float] = []
        for _ in range(episodes):
            observation, _ = self.eval_env.reset()
            done = False
            episode_return = 0.0
            while not done:
                observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.learner.act(observation_tensor, deterministic=True).squeeze(0).detach().cpu().numpy()
                observation, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = bool(terminated or truncated)
                episode_return += float(reward)
            returns.append(episode_return)
            normalized_returns.append(normalized_score(self.env_name, episode_return))
        return {
            "eval_return": float(np.mean(returns)),
            "eval_normalized_return": float(np.mean(normalized_returns)),
        }

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
