"""Behavior cloning expert and reference value training."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rlif.data.dataset import OfflineTransitions
from rlif.models.actor import GaussianActor
from rlif.models.critic import QNetwork
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


def train_expert_pipeline(
    dataset: dict[str, np.ndarray],
    observation_dim: int,
    action_dim: int,
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
    output_dir = Path(output_dir)
    actor_path = output_dir / "expert_actor.pt"
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
    reference_q_path = output_dir / "expert_reference_q.pt"
    save_checkpoint(reference_q_path, {"q_state_dict": ref_q.q.state_dict()})
    metrics = {**bc_metrics, **ref_metrics}
    return ExpertTrainingResult(actor_path=actor_path, reference_q_path=reference_q_path, metrics=metrics)
