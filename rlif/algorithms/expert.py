"""Behavior cloning expert and reference value training."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rlif.data.dataset import OfflineTransitions
from rlif.models.actor import GaussianActor
from rlif.models.critic import CriticEnsemble, QNetwork
from rlif.utils.checkpoint import save_checkpoint


def _returns_to_go(rewards: np.ndarray, terminals: np.ndarray, discount: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0
    for index in reversed(range(len(rewards))):
        if terminals[index] > 0.5:
            running_return = 0.0
        running_return = rewards[index] + discount * running_return
        returns[index] = running_return
    return returns


@dataclass
class ExpertTrainingResult:
    actor_path: Path
    reference_q_path: Path | None
    metrics: dict[str, float]


class BehaviorCloningTrainer:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        learning_rate: float,
        weight_decay: float,
        device: str,
    ) -> None:
        self.device = device
        self.actor = GaussianActor(observation_dim, action_dim, hidden_dims).to(device)
        self.optimizer = AdamW(self.actor.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()

    def train(
        self,
        dataset: OfflineTransitions,
        batch_size: int,
        epochs: int,
        grad_clip_norm: float,
        progress_callback: Callable[[int, Mapping[str, float]], None] | None = None,
        evaluation_fn: Callable[[GaussianActor], Mapping[str, float]] | None = None,
        eval_interval_epochs: int = 1,
    ) -> dict[str, float]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        losses: list[float] = []
        self.actor.train()
        latest_epoch_metrics: dict[str, float] = {}
        eval_interval_epochs = max(1, int(eval_interval_epochs))
        progress = tqdm(range(1, epochs + 1), desc="BC epochs", leave=True, dynamic_ncols=True)
        for epoch in progress:
            epoch_losses: list[float] = []
            for batch in loader:
                observations = batch["observations"].to(self.device)
                actions = batch["actions"].to(self.device)
                mean, _ = self.actor.forward(observations)
                target_pre_tanh = torch.atanh(torch.clamp(actions, -0.999999, 0.999999))
                loss = self.loss_fn(mean, target_pre_tanh)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip_norm)
                self.optimizer.step()
                scalar_loss = float(loss.item())
                losses.append(scalar_loss)
                epoch_losses.append(scalar_loss)
            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            epoch_metrics: dict[str, float] = {"expert/bc_loss": epoch_loss}
            if evaluation_fn is not None and (epoch % eval_interval_epochs == 0 or epoch == epochs):
                self.actor.eval()
                eval_metrics = dict(evaluation_fn(self.actor))
                self.actor.train()
                epoch_metrics.update({str(key): float(value) for key, value in eval_metrics.items()})
            if progress_callback is not None:
                progress_callback(epoch, epoch_metrics)
            progress.set_postfix({key.split("/")[-1]: f"{value:.4f}" for key, value in epoch_metrics.items()})
            latest_epoch_metrics = epoch_metrics
        summary = {"bc_loss": float(np.mean(losses)) if losses else 0.0, "bc_last_loss": latest_epoch_metrics.get("expert/bc_loss", 0.0)}
        if "expert/eval_return" in latest_epoch_metrics:
            summary["eval_return"] = latest_epoch_metrics["expert/eval_return"]
        if "expert/eval_normalized_return" in latest_epoch_metrics:
            summary["eval_normalized_return"] = latest_epoch_metrics["expert/eval_normalized_return"]
        return summary


class ReferenceQTrainer:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        learning_rate: float,
        weight_decay: float,
        device: str,
    ) -> None:
        self.device = device
        self.q = QNetwork(observation_dim, action_dim, hidden_dims).to(device)
        self.optimizer = AdamW(self.q.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()

    def train(
        self,
        dataset: dict[str, np.ndarray],
        batch_size: int,
        epochs: int,
        discount: float,
        grad_clip_norm: float,
        progress_callback: Callable[[int, Mapping[str, float]], None] | None = None,
    ) -> dict[str, float]:
        returns = _returns_to_go(dataset["rewards"], dataset["terminals"], discount)
        observations = torch.as_tensor(dataset["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(dataset["actions"], dtype=torch.float32, device=self.device)
        targets = torch.as_tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(-1)
        indices = torch.arange(len(observations), device=self.device)
        losses: list[float] = []
        self.q.train()
        latest_epoch_loss = 0.0
        progress = tqdm(range(1, epochs + 1), desc="Ref-Q epochs", leave=True, dynamic_ncols=True)
        for epoch in progress:
            epoch_losses: list[float] = []
            shuffled = indices[torch.randperm(len(indices), device=self.device)]
            for start in range(0, len(shuffled) - batch_size + 1, batch_size):
                batch_indices = shuffled[start : start + batch_size]
                prediction = self.q(observations[batch_indices], actions[batch_indices])
                loss = self.loss_fn(prediction, targets[batch_indices])
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q.parameters(), grad_clip_norm)
                self.optimizer.step()
                scalar_loss = float(loss.item())
                losses.append(scalar_loss)
                epoch_losses.append(scalar_loss)
            latest_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            epoch_metrics = {"expert/ref_q_loss": latest_epoch_loss}
            if progress_callback is not None:
                progress_callback(epoch, epoch_metrics)
            progress.set_postfix({"ref_q_loss": f"{latest_epoch_loss:.4f}"})
        return {"ref_q_loss": float(np.mean(losses)) if losses else 0.0, "ref_q_last_loss": latest_epoch_loss}


class OfflineCQLTrainer:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        actor_hidden_dims: tuple[int, ...],
        critic_hidden_dims: tuple[int, ...],
        learning_rate: float,
        weight_decay: float,
        discount: float,
        tau: float,
        cql_alpha: float,
        entropy_temperature: float,
        bc_weight: float,
        policy_weight: float,
        bc_warmup_fraction: float,
        device: str,
    ) -> None:
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.entropy_temperature = entropy_temperature
        self.bc_weight = bc_weight
        self.policy_weight = policy_weight
        self.bc_warmup_fraction = bc_warmup_fraction
        self.device = device

        self.actor = GaussianActor(observation_dim, action_dim, actor_hidden_dims).to(device)
        self.critic = CriticEnsemble(observation_dim, action_dim, critic_hidden_dims, ensemble_size=2).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        self.actor_optimizer = AdamW(self.actor.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train(
        self,
        dataset: OfflineTransitions,
        batch_size: int,
        epochs: int,
        grad_clip_norm: float,
        progress_callback: Callable[[int, Mapping[str, float]], None] | None = None,
        evaluation_fn: Callable[[GaussianActor], Mapping[str, float]] | None = None,
        eval_interval_epochs: int = 1,
    ) -> dict[str, float]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        latest_epoch_metrics: dict[str, float] = {}
        eval_interval_epochs = max(1, int(eval_interval_epochs))
        progress = tqdm(range(1, epochs + 1), desc="Offline CQL epochs", leave=True, dynamic_ncols=True)
        bc_warmup_epochs = max(1, int(round(epochs * self.bc_warmup_fraction)))
        for epoch in progress:
            critic_losses: list[float] = []
            bellman_losses: list[float] = []
            cql_losses: list[float] = []
            actor_losses: list[float] = []
            bc_losses: list[float] = []
            policy_losses: list[float] = []
            for batch in loader:
                observations = batch["observations"].to(self.device)
                actions = batch["actions"].to(self.device)
                rewards = batch["rewards"].to(self.device)
                next_observations = batch["next_observations"].to(self.device)
                dones = batch["dones"].to(self.device)

                with torch.no_grad():
                    next_actions, next_log_prob, _ = self.actor.sample(next_observations)
                    target_q = self.target_critic.min_q(next_observations, next_actions)
                    target_value = target_q - self.entropy_temperature * next_log_prob
                    q_target = rewards + (1.0 - dones) * self.discount * target_value

                q_values = self.critic(observations, actions)
                bellman_loss = F.smooth_l1_loss(q_values, q_target.unsqueeze(0).expand_as(q_values))
                cql_loss = self._cql_penalty(observations, next_observations, q_values)
                critic_loss = bellman_loss + self.cql_alpha * cql_loss

                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip_norm)
                self.critic_optimizer.step()

                new_actions, log_prob, _ = self.actor.sample(observations)
                q_new = self.critic.min_q(observations, new_actions)
                mean_actions, _ = self.actor.forward(observations)
                target_pre_tanh = torch.atanh(torch.clamp(actions, -0.999999, 0.999999))
                bc_loss = F.mse_loss(mean_actions, target_pre_tanh)
                q_scale = q_new.abs().mean().detach().clamp_min(1e-6)
                normalized_policy_loss = (self.entropy_temperature * log_prob - q_new).mean() / q_scale
                if epoch <= bc_warmup_epochs:
                    actor_loss = bc_loss
                else:
                    actor_loss = self.bc_weight * bc_loss + self.policy_weight * normalized_policy_loss
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip_norm)
                self.actor_optimizer.step()

                self.soft_update_targets()

                critic_losses.append(float(critic_loss.item()))
                bellman_losses.append(float(bellman_loss.item()))
                cql_losses.append(float(cql_loss.item()))
                actor_losses.append(float(actor_loss.item()))
                bc_losses.append(float(bc_loss.item()))
                policy_losses.append(float(normalized_policy_loss.item()))

            epoch_metrics: dict[str, float] = {
                "expert/offline_rl_actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
                "expert/offline_rl_bc_loss": float(np.mean(bc_losses)) if bc_losses else 0.0,
                "expert/offline_rl_policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
                "expert/offline_rl_critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
                "expert/offline_rl_bellman_loss": float(np.mean(bellman_losses)) if bellman_losses else 0.0,
                "expert/offline_rl_cql_loss": float(np.mean(cql_losses)) if cql_losses else 0.0,
            }
            if evaluation_fn is not None and (epoch % eval_interval_epochs == 0 or epoch == epochs):
                self.actor.eval()
                eval_metrics = dict(evaluation_fn(self.actor))
                self.actor.train()
                epoch_metrics.update({str(key): float(value) for key, value in eval_metrics.items()})
            if progress_callback is not None:
                progress_callback(epoch, epoch_metrics)
            progress.set_postfix({key.split("/")[-1]: f"{value:.4f}" for key, value in epoch_metrics.items()})
            latest_epoch_metrics = epoch_metrics

        return {key.replace("expert/", ""): value for key, value in latest_epoch_metrics.items()}

    def _cql_penalty(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        data_q_values: torch.Tensor,
    ) -> torch.Tensor:
        random_actions = torch.empty(
            observations.shape[0],
            self.action_dim,
            device=self.device,
        ).uniform_(-1.0, 1.0)
        current_actions, _, _ = self.actor.sample(observations)
        next_actions, _, _ = self.actor.sample(next_observations)

        random_q = self.critic(observations, random_actions)
        current_q = self.critic(observations, current_actions)
        next_q = self.critic(observations, next_actions)
        ood_q_values = torch.cat([random_q, current_q, next_q], dim=1)
        return torch.logsumexp(ood_q_values, dim=1).mean() - data_q_values.mean()

    def soft_update_targets(self) -> None:
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters(), strict=True):
                target_parameter.data.mul_(1.0 - self.tau)
                target_parameter.data.add_(self.tau * parameter.data)


def train_expert_pipeline(
    dataset: dict[str, np.ndarray],
    observation_dim: int,
    action_dim: int,
    expert_kind: str,
    expert_hidden_dims: tuple[int, ...],
    ref_q_hidden_dims: tuple[int, ...],
    expert_lr: float,
    expert_weight_decay: float,
    ref_q_lr: float,
    ref_q_weight_decay: float,
    batch_size: int,
    expert_epochs: int,
    ref_q_epochs: int,
    discount: float,
    device: str,
    output_dir: str | Path,
    grad_clip_norm: float,
    progress_callback: Callable[[int, Mapping[str, float]], None] | None = None,
    evaluation_fn: Callable[[GaussianActor], Mapping[str, float]] | None = None,
    eval_interval_epochs: int = 1,
) -> ExpertTrainingResult:
    offline_dataset = OfflineTransitions.from_arrays(dataset, device=device)
    output_dir = Path(output_dir)
    actor_path = output_dir / "expert_actor.pt"
    reference_q_path = output_dir / "expert_reference_q.pt"

    if expert_kind == "offline_rl":
        offline_rl = OfflineCQLTrainer(
            observation_dim=observation_dim,
            action_dim=action_dim,
            actor_hidden_dims=expert_hidden_dims,
            critic_hidden_dims=ref_q_hidden_dims,
            learning_rate=expert_lr,
            weight_decay=expert_weight_decay,
            discount=discount,
            tau=0.005,
            cql_alpha=1.0,
            entropy_temperature=0.2,
            bc_weight=1.0,
            policy_weight=0.1,
            bc_warmup_fraction=0.25,
            device=device,
        )
        metrics = offline_rl.train(
            offline_dataset,
            batch_size=batch_size,
            epochs=expert_epochs,
            grad_clip_norm=grad_clip_norm,
            progress_callback=progress_callback,
            evaluation_fn=evaluation_fn,
            eval_interval_epochs=eval_interval_epochs,
        )
        save_checkpoint(actor_path, {"actor_state_dict": offline_rl.actor.state_dict(), "expert_kind": "offline_rl"})
        save_checkpoint(
            reference_q_path,
            {
                "critic_state_dict": offline_rl.critic.state_dict(),
                "critic_ensemble_size": 2,
                "critic_hidden_dims": tuple(ref_q_hidden_dims),
                "expert_kind": "offline_rl",
            },
        )
        return ExpertTrainingResult(actor_path=actor_path, reference_q_path=reference_q_path, metrics=metrics)

    if expert_kind != "bc":
        raise ValueError(f"Unknown expert.kind '{expert_kind}'. Expected 'bc' or 'offline_rl'.")

    bc = BehaviorCloningTrainer(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=expert_hidden_dims,
        learning_rate=expert_lr,
        weight_decay=expert_weight_decay,
        device=device,
    )
    bc_metrics = bc.train(
        offline_dataset,
        batch_size=batch_size,
        epochs=expert_epochs,
        grad_clip_norm=grad_clip_norm,
        progress_callback=progress_callback,
        evaluation_fn=evaluation_fn,
        eval_interval_epochs=eval_interval_epochs,
    )
    save_checkpoint(actor_path, {"actor_state_dict": bc.actor.state_dict()})

    ref_q = ReferenceQTrainer(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=ref_q_hidden_dims,
        learning_rate=ref_q_lr,
        weight_decay=ref_q_weight_decay,
        device=device,
    )
    ref_metrics = ref_q.train(
        dataset=dataset,
        batch_size=batch_size,
        epochs=ref_q_epochs,
        discount=discount,
        grad_clip_norm=grad_clip_norm,
        progress_callback=(
            None
            if progress_callback is None
            else lambda epoch, metrics: progress_callback(expert_epochs + int(epoch), metrics)
        ),
    )
    save_checkpoint(reference_q_path, {"q_state_dict": ref_q.q.state_dict()})
    metrics = {**bc_metrics, **ref_metrics}
    return ExpertTrainingResult(actor_path=actor_path, reference_q_path=reference_q_path, metrics=metrics)
