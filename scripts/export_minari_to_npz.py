"""Export a Minari expert dataset to the .npz format used by RLIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Minari dataset to RLIF .npz format")
    parser.add_argument("--dataset-id", type=str, required=True, help="Minari dataset id, e.g. D4RL/hopper/expert-v0")
    parser.add_argument("--output", type=str, required=True, help="Output .npz path")
    return parser.parse_args()


def _episode_to_transitions(episode_data: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observations = np.asarray(getattr(episode_data, "observations"), dtype=np.float32)
    actions = np.asarray(getattr(episode_data, "actions"), dtype=np.float32)
    rewards = np.asarray(getattr(episode_data, "rewards"), dtype=np.float32)
    terminations = np.asarray(getattr(episode_data, "terminations"), dtype=bool)
    truncations = np.asarray(getattr(episode_data, "truncations"), dtype=bool)

    if observations.shape[0] == actions.shape[0] + 1:
        next_observations = observations[1:]
        observations = observations[:-1]
    elif observations.shape[0] == actions.shape[0]:
        next_observations = np.concatenate([observations[1:], observations[-1:]], axis=0)
    else:
        raise ValueError(
            f"Unexpected episode lengths: observations={observations.shape[0]} actions={actions.shape[0]}"
        )

    terminals = np.asarray(terminations[: actions.shape[0]] | truncations[: actions.shape[0]], dtype=np.float32)
    rewards = rewards[: actions.shape[0]]
    return observations, actions, rewards, next_observations, terminals


def main() -> None:
    args = parse_args()
    try:
        import minari
    except ImportError as error:
        raise SystemExit(
            "minari is not installed. Install it with `python -m pip install minari` or `python -m pip install -e .[offline]`."
        ) from error
    # Try loading the requested dataset; if not found, try common fallbacks
    base_id = args.dataset_id
    candidates: list[str] = [base_id]
    # If user supplied a D4RL style id, try the Farama 'mujoco' namespace and common variants
    if base_id.startswith("D4RL/"):
        parts = base_id.split("/")
        if len(parts) >= 2:
            env = parts[1]
        else:
            env = None
        # direct replacement
        candidates.append(base_id.replace("D4RL/", "mujoco/"))
        # try common variant suffixes
        if env:
            tail = '/'.join(parts[2:]) if len(parts) > 2 else ''
            if tail:
                candidates.append(f"mujoco/{env}/{tail}")
            for variant in ("expert-v0", "expert", "medium-v0", "simple-v0"):
                candidates.append(f"mujoco/{env}/{variant}")

    # ensure unique order-preserving
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    dataset = None
    used_dataset_id = None
    last_err: Exception | None = None
    for cand in candidates:
        try:
            print(f"Trying Minari dataset id: {cand}")
            dataset = minari.load_dataset(cand, download=True)
            used_dataset_id = cand
            break
        except Exception as e:  # noqa: BLE001 - delegate messaging to user
            print(f"Failed to load {cand}: {e}")
            last_err = e

    if dataset is None:
        raise SystemExit(
            f"Couldn't find any matching dataset for '{args.dataset_id}'. Tried: {candidates}.\nLast error: {last_err}"
        )
    observations_list: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    rewards_list: list[np.ndarray] = []
    next_observations_list: list[np.ndarray] = []
    terminals_list: list[np.ndarray] = []

    for episode_data in dataset.iterate_episodes():
        observations, actions, rewards, next_observations, terminals = _episode_to_transitions(episode_data)
        observations_list.append(observations)
        actions_list.append(actions)
        rewards_list.append(rewards)
        next_observations_list.append(next_observations)
        terminals_list.append(terminals)

    payload = {
        "observations": np.concatenate(observations_list, axis=0),
        "actions": np.concatenate(actions_list, axis=0),
        "rewards": np.concatenate(rewards_list, axis=0),
        "next_observations": np.concatenate(next_observations_list, axis=0),
        "terminals": np.concatenate(terminals_list, axis=0),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    display_id = used_dataset_id or args.dataset_id
    print(f"Saved {display_id} to {output_path}")


if __name__ == "__main__":
    main()
