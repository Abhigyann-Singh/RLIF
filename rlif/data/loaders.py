"""Dataset resolution helpers for offline initialization and expert training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlif.data.dataset import load_npz_trajectories, select_first_trajectories


def resolve_offline_dataset(
    env: Any,
    offline_path: str | Path | None,
    d4rl_dataset_id: str | None = None,
    trajectory_limit: int | None = None,
) -> dict[str, Any]:
    if offline_path:
        path = Path(offline_path)
        if path.exists():
            dataset = load_npz_trajectories(path)
            return select_first_trajectories(dataset, trajectory_limit)
        raise FileNotFoundError(
            f"Offline dataset not found at {path}. Download or export the offline dataset for this environment, then save it as an .npz file with observations, actions, next_observations, rewards, and terminals."
        )
    if d4rl_dataset_id:
        raise ValueError(
            f"data.d4rl_dataset_id={d4rl_dataset_id} is not supported in this Gymnasium-only runtime setup. "
            "Download or export the offline dataset once and set data.offline_path or data.expert_demo_path to the local .npz file."
        )
    raise FileNotFoundError(
        "No offline dataset path was provided. Set data.offline_path or data.expert_demo_path to a valid .npz file."
    )

