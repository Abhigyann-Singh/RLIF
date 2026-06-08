"""Return normalization helpers for D4RL-style locomotion tasks."""

from __future__ import annotations

_D4RL_SCORE_RANGES: dict[str, tuple[float, float]] = {
    "hopper": (-20.272305, 3234.3),
    "walker2d": (1.629008, 4592.3),
}


def normalized_score(task_name: str, episode_return: float) -> float:
    task_key = task_name.lower()
    if task_key in _D4RL_SCORE_RANGES:
        random_score, expert_score = _D4RL_SCORE_RANGES[task_key]
        return 100.0 * (episode_return - random_score) / max(expert_score - random_score, 1e-6)
    return episode_return
