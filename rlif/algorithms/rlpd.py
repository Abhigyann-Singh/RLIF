"""RLPD-style SAC learner used as the online backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from rlif.data.replay_buffer import Batch
from rlif.models.actor import GaussianActor
from rlif.models.critic import CriticEnsemble


@dataclass
class UpdateMetrics:
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    q_mean: float
    q_target_mean: float


class RLPDTrainer:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        actor_hidden_dims: tuple[int, ...],
        critic_hidden_dims: tuple[int, ...],
        critic_ensemble_size: int,
        critic_subset_size: int | None,
        learning_rate: float,
        weight_decay: float,
        discount: float,
        tau: float,
        entropy_backup: bool,
        automatic_entropy_tuning: bool,
        target_entropy_scale: float,
        grad_clip_norm: float,
        device: str,
    ) -> None:
        self.device = device
        self.discount = discount
        self.tau = tau
        self.entropy_backup = entropy_backup
        self.automatic_entropy_tuning = automatic_entropy_tuning and entropy_backup
        self.grad_clip_norm = grad_clip_norm
        self.critic_subset_size = critic_subset_size

        self.actor = GaussianActor(observation_dim, action_dim, actor_hidden_dims).to(device)
        self.critic = CriticEnsemble(observation_dim, action_dim, critic_hidden_dims, critic_ensemble_size).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        self.actor_optimizer = AdamW(self.actor.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if self.automatic_entropy_tuning:
            self.log_alpha = torch.tensor(0.0, device=device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
            self.target_entropy = target_entropy_scale * float(action_dim)
        else:
            self.log_alpha = torch.tensor(0.0, device=device)
            self.alpha_optimizer = None
            self.target_entropy = 0.0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp() if self.entropy_backup else torch.tensor(0.0, device=self.device)

    @torch.no_grad()
    def act(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action, _, _ = self.actor.sample(observation, deterministic=deterministic)
        return action.clamp(-1.0, 1.0)

    def update_critic(self, batch: Batch) -> dict[str, float]:
        observations = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        next_observations = batch.next_observations
        dones = batch.dones

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_observations)
            target_q = self.target_critic.subset_min_q(next_observations, next_actions, self.critic_subset_size)
            if self.entropy_backup:
                target_value = target_q - self.alpha * next_log_prob
            else:
                target_value = target_q
            q_target = rewards + (1.0 - dones) * self.discount * target_value
            q_target = torch.clamp(q_target, -1e6, 1e6)

        q_values = self.critic(observations, actions)
        critic_target = q_target.unsqueeze(0).expand_as(q_values)
        critic_loss = F.smooth_l1_loss(q_values, critic_target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        return {
            "critic_loss": float(critic_loss.item()),
            "critic_updates": 1.0,
            "q_mean": float(q_values.mean().item()),
            "q_std": float(q_values.std(unbiased=False).item()),
            "q_target_mean": float(q_target.mean().item()),
            "q_target_std": float(q_target.std(unbiased=False).item()),
        }

    def update_actor(self, batch: Batch) -> dict[str, float]:
        observations = batch.observations

        new_actions, log_prob, _ = self.actor.sample(observations)
        q_new = self.critic.mean_q(observations, new_actions)
        if self.entropy_backup:
            actor_loss = (self.alpha.detach() * log_prob - q_new).mean()
        else:
            actor_loss = (-q_new).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.automatic_entropy_tuning and self.alpha_optimizer is not None:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def update(self, batch: Batch) -> dict[str, float]:
        metrics = self.update_critic(batch)
        metrics.update(self.update_actor(batch))
        self.soft_update_targets()
        return metrics

    def soft_update_targets(self) -> None:
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters(), strict=True):
                target_parameter.data.mul_(1.0 - self.tau)
                target_parameter.data.add_(self.tau * parameter.data)

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": None if self.alpha_optimizer is None else self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict.get("target_critic", state_dict["critic"]))
        if "log_alpha" in state_dict and self.automatic_entropy_tuning:
            self.log_alpha.data.copy_(state_dict["log_alpha"].to(self.device))
        if "actor_optimizer" in state_dict:
            self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        if "critic_optimizer" in state_dict:
            self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        if self.alpha_optimizer is not None and state_dict.get("alpha_optimizer") is not None:
            self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
