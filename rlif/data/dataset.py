"""Offline dataset utilities for BC and reference Q training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _to_numpy(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array)


def load_npz_trajectories(path: str | Path) -> dict[str, np.ndarray]:
    payload = np.load(Path(path), allow_pickle=True)
    data = {key: _to_numpy(payload[key]) for key in payload.files}
    required = {"observations", "actions"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Missing keys in offline dataset {path}: {sorted(missing)}")
    if "rewards" not in data:
        data["rewards"] = np.zeros(len(data["observations"]), dtype=np.float32)
    if "next_observations" not in data:
        raise ValueError(f"Dataset {path} must provide next_observations")
    if "terminals" not in data:
        data["terminals"] = np.zeros(len(data["observations"]), dtype=np.float32)
    return data


def select_first_trajectories(data: dict[str, np.ndarray], trajectory_limit: int | None) -> dict[str, np.ndarray]:
    if trajectory_limit is None:
        return data
    if trajectory_limit <= 0:
        raise ValueError("trajectory_limit must be positive when provided")

    terminals = np.asarray(data.get("terminals", np.zeros(len(data["observations"]), dtype=np.float32)))
    timeouts = np.asarray(data.get("timeouts", np.zeros(len(data["observations"]), dtype=np.float32)))
    episode_ends = np.flatnonzero((terminals > 0.5) | (timeouts > 0.5))
    if len(episode_ends) < trajectory_limit:
        raise ValueError(
            f"Requested {trajectory_limit} trajectories, but dataset only contains {len(episode_ends)} complete episodes"
        )
    stop_index = int(episode_ends[trajectory_limit - 1]) + 1
    return {key: value[:stop_index] for key, value in data.items()}


def load_d4rl_dataset(env: Any) -> dict[str, np.ndarray]:
    if hasattr(env, "get_dataset"):
        dataset = env.get_dataset()
    elif hasattr(env.unwrapped, "get_dataset"):
        dataset = env.unwrapped.get_dataset()
    else:
        raise ValueError("Environment does not expose a D4RL-style dataset API")
    next_observations = dataset.get("next_observations")
    if next_observations is None:
        raise ValueError("D4RL dataset did not provide next_observations")
    return {
        "observations": _to_numpy(dataset["observations"]),
        "actions": _to_numpy(dataset["actions"]),
        "rewards": _to_numpy(dataset.get("rewards", np.zeros(len(dataset["observations"]), dtype=np.float32))),
        "next_observations": _to_numpy(next_observations),
        "terminals": _to_numpy(dataset.get("terminals", np.zeros(len(dataset["observations"]), dtype=np.float32))),
    }


@dataclass
class OfflineTransitions(Dataset):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor

    @classmethod
    def from_arrays(cls, data: dict[str, np.ndarray], device: str = "cpu") -> "OfflineTransitions":
        observations = torch.as_tensor(data["observations"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(data["actions"], dtype=torch.float32, device=device)
        rewards = torch.as_tensor(data["rewards"], dtype=torch.float32, device=device).unsqueeze(-1)
        next_observations = torch.as_tensor(data["next_observations"], dtype=torch.float32, device=device)
        dones = torch.as_tensor(data["terminals"], dtype=torch.float32, device=device).unsqueeze(-1)
        return cls(observations, actions, rewards, next_observations, dones)

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observations": self.observations[index],
            "actions": self.actions[index],
            "rewards": self.rewards[index],
            "next_observations": self.next_observations[index],
            "dones": self.dones[index],
        }
