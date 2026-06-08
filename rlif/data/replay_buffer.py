"""Replay buffer with symmetric online/offline sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    source: torch.Tensor


class ReplayBuffer:
    def __init__(self, observation_dim: int, action_dim: int, capacity: int, device: str = "cpu") -> None:
        self.capacity = capacity
        self.device = device
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.sources = np.zeros((capacity, 1), dtype=np.float32)
        self._size = 0
        self._pointer = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: float,
        source: float,
    ) -> None:
        self.observations[self._pointer] = observation
        self.actions[self._pointer] = action
        self.rewards[self._pointer] = reward
        self.next_observations[self._pointer] = next_observation
        self.dones[self._pointer] = done
        self.sources[self._pointer] = source
        self._pointer = (self._pointer + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(self, dataset: dict[str, np.ndarray], source: float) -> None:
        total = len(dataset["observations"])
        for index in range(total):
            self.add(
                dataset["observations"][index],
                dataset["actions"][index],
                float(dataset["rewards"][index]),
                dataset["next_observations"][index],
                float(dataset["terminals"][index]),
                source,
            )

    def state_dict(self) -> dict[str, np.ndarray | int]:
        return {
            "capacity": self.capacity,
            "size": self._size,
            "pointer": self._pointer,
            "observations": self.observations[: self._size].copy(),
            "actions": self.actions[: self._size].copy(),
            "rewards": self.rewards[: self._size].copy(),
            "next_observations": self.next_observations[: self._size].copy(),
            "dones": self.dones[: self._size].copy(),
            "sources": self.sources[: self._size].copy(),
        }

    def load_state_dict(self, state_dict: dict[str, np.ndarray | int]) -> None:
        size = int(state_dict["size"])
        if size > self.capacity:
            raise ValueError(f"Checkpoint buffer size {size} exceeds capacity {self.capacity}")
        self._size = size
        self._pointer = int(state_dict["pointer"])
        self.observations[:size] = np.asarray(state_dict["observations"], dtype=np.float32)
        self.actions[:size] = np.asarray(state_dict["actions"], dtype=np.float32)
        self.rewards[:size] = np.asarray(state_dict["rewards"], dtype=np.float32)
        self.next_observations[:size] = np.asarray(state_dict["next_observations"], dtype=np.float32)
        self.dones[:size] = np.asarray(state_dict["dones"], dtype=np.float32)
        self.sources[:size] = np.asarray(state_dict["sources"], dtype=np.float32)

    def sample(self, batch_size: int) -> Batch:
        if self._size == 0:
            raise ValueError("Replay buffer is empty")
        indices = np.random.choice(self._size, batch_size, replace=self._size < batch_size)
        np.random.shuffle(indices)
        return Batch(
            observations=torch.as_tensor(self.observations[indices], device=self.device),
            actions=torch.as_tensor(self.actions[indices], device=self.device),
            rewards=torch.as_tensor(self.rewards[indices], device=self.device),
            next_observations=torch.as_tensor(self.next_observations[indices], device=self.device),
            dones=torch.as_tensor(self.dones[indices], device=self.device),
            source=torch.as_tensor(self.sources[indices], device=self.device),
        )

    def composition(self) -> dict[str, float]:
        if self._size == 0:
            return {"offline_fraction": 0.0, "online_fraction": 0.0}
        offline = float(np.sum(self.sources[: self._size] > 0.5))
        online = float(np.sum(self.sources[: self._size] <= 0.5))
        total = offline + online
        return {"offline_fraction": offline / total, "online_fraction": online / total}
