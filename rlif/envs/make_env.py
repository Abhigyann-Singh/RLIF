"""Gymnasium environment creation for MuJoCo simulation tasks."""

from __future__ import annotations

import gymnasium as gym


def make_env(env_id: str, seed: int, max_episode_steps: int, deterministic: bool = False) -> gym.Env:
    env = gym.make(env_id)
    if max_episode_steps > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if deterministic:
        try:
            env = gym.wrappers.RecordEpisodeStatistics(env)
        except Exception:
            pass
    return env
