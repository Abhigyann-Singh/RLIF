"""Reusable MLP blocks."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn


def build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int, layer_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)
