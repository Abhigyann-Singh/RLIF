"""Gaussian squashed actor used by SAC/RLPD and BC."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from rlif.models.mlp import build_mlp


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class GaussianActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must not be empty")
        self.backbone = build_mlp(observation_dim, hidden_dims[:-1], hidden_dims[-1], layer_norm=False)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observations)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, observations: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(observations)
        std = log_std.exp()
        if deterministic:
            pre_tanh = mean
        else:
            noise = torch.randn_like(mean)
            pre_tanh = mean + std * noise
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob(mean, log_std, pre_tanh, action)
        return action, log_prob, mean

    @staticmethod
    def _log_prob(mean: torch.Tensor, log_std: torch.Tensor, pre_tanh: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        variance = torch.exp(2.0 * log_std)
        gaussian_log_prob = -0.5 * (
            ((pre_tanh - mean) ** 2) / variance + 2.0 * log_std + torch.log(torch.tensor(2.0 * torch.pi, device=mean.device))
        )
        gaussian_log_prob = gaussian_log_prob.sum(dim=-1, keepdim=True)
        correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return gaussian_log_prob - correction


@dataclass
class ActorOutput:
    action: torch.Tensor
    log_prob: torch.Tensor
    mean_action: torch.Tensor
