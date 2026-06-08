"""LayerNorm critic ensemble for RLPD."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from rlif.models.mlp import build_mlp


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.model = build_mlp(observation_dim + action_dim, hidden_dims, 1, layer_norm=True)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([observations, actions], dim=-1))


class CriticEnsemble(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: Sequence[int], ensemble_size: int) -> None:
        super().__init__()
        self.networks = nn.ModuleList(
            [QNetwork(observation_dim, action_dim, hidden_dims) for _ in range(ensemble_size)]
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        qs = [network(observations, actions) for network in self.networks]
        return torch.stack(qs, dim=0)

    def q_values(self, observations: torch.Tensor, actions: torch.Tensor, indices: list[int] | None = None) -> torch.Tensor:
        networks = self.networks if indices is None else [self.networks[index] for index in indices]
        qs = [network(observations, actions) for network in networks]
        return torch.stack(qs, dim=0)

    def min_q(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(observations, actions).min(dim=0).values

    def subset_min_q(self, observations: torch.Tensor, actions: torch.Tensor, subset_size: int | None = None) -> torch.Tensor:
        if subset_size is None or subset_size <= 0 or subset_size >= len(self.networks):
            return self.min_q(observations, actions)
        indices = torch.randperm(len(self.networks), device=observations.device)[:subset_size].tolist()
        return self.q_values(observations, actions, indices=indices).min(dim=0).values

    def mean_q(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(observations, actions).mean(dim=0)
