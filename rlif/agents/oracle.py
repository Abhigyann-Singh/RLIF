"""Intervention oracle used by RLIF rollouts."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from rlif.models.actor import GaussianActor
from rlif.models.critic import CriticEnsemble, QNetwork


@dataclass
class InterventionDecision:
    intervene: bool
    q_expert: float | None = None
    q_agent: float | None = None


class InterventionOracle:
    def __init__(
        self,
        mode: str,
        beta: float,
        alpha: float,
        delta: float,
        relative_threshold: bool,
        expert_actor: GaussianActor,
        reference_q: QNetwork | CriticEnsemble | None,
        device: str,
    ) -> None:
        self.mode = mode
        self.beta = beta
        self.alpha = alpha
        self.delta = delta
        self.relative_threshold = relative_threshold
        self.expert_actor = expert_actor
        self.reference_q = reference_q
        self.device = device

    @torch.no_grad()
    def expert_action(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        action, _, _ = self.expert_actor.sample(observation, deterministic=deterministic)
        return action

    @torch.no_grad()
    def decide(self, observation: torch.Tensor, agent_action: torch.Tensor) -> InterventionDecision:
        if self.mode == "random":
            return InterventionDecision(intervene=random.random() < self.beta)
        if self.reference_q is None:
            raise ValueError("Value-based intervention requires a reference Q model")
        expert_action = self.expert_action(observation, deterministic=True)
        q_expert = self._q_value(observation, expert_action)
        q_agent = self._q_value(observation, agent_action)
        if self.relative_threshold:
            condition = self.alpha * q_expert > q_agent
        else:
            condition = q_expert > q_agent + self.delta
        intervention_probability = self.beta if bool(condition.item()) else 1.0 - self.beta
        intervene = random.random() < intervention_probability
        return InterventionDecision(intervene=intervene, q_expert=float(q_expert.item()), q_agent=float(q_agent.item()))

    def _q_value(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if isinstance(self.reference_q, CriticEnsemble):
            return self.reference_q.mean_q(observation, action)
        return self.reference_q(observation, action)
