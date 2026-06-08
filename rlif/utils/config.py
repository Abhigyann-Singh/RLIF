"""Configuration loading and typed experiment dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    name: str
    env_id: str
    max_episode_steps: int
    action_repeat: int = 1
    deterministic_eval: bool = True
    success_keys: tuple[str, ...] = ("success", "is_success")


@dataclass
class ExpertConfig:
    kind: str = "bc"
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    hidden_dims: tuple[int, ...] = (256, 256)
    dropout: float = 0.0
    epochs: int = 100
    grad_clip_norm: float = 10.0
    checkpoint_interval: int = 10
    ref_q_epochs: int = 100
    ref_q_hidden_dims: tuple[int, ...] = (256, 256)


@dataclass
class RLPDConfig:
    batch_size: int = 256
    learning_rate: float = 3e-4
    discount: float = 0.99
    weight_decay: float = 1e-3
    actor_hidden_dims: tuple[int, ...] = (256, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 256)
    critic_ensemble_size: int = 10
    critic_subset_size: int = 2
    tau: float = 0.005
    utd_ratio: int = 20
    offline_ratio: float = 0.5
    entropy_backup: bool = True
    automatic_entropy_tuning: bool = True
    target_entropy_scale: float = -1.0
    grad_clip_norm: float = 10.0


@dataclass
class RLIFConfig:
    intervention_mode: str = "value"
    relative_threshold: bool = False
    beta: float = 0.95
    alpha: float = 0.97
    delta: float = 0.0
    zero_offline_rewards: bool = True
    takeover_steps: int = 0
    rounds: int = 100
    trajectories_per_round: int = 5
    pretrain_epochs: int = 200
    pretrain_train_steps_per_epoch: int = 300
    round_train_epochs: int = 25
    round_train_steps_per_epoch: int = 100
    eval_interval_rounds: int = 1
    checkpoint_interval_rounds: int = 1
    eval_episodes: int = 10
    seed: int = 0
    warmup_steps: int = 0
    debug_interval_updates: int = 0
    debug_preview_size: int = 8
    online_buffer_size: int = 1_000_000
    offline_buffer_size: int = 1_000_000
    resume_from: str = ""


@dataclass
class DataConfig:
    offline_path: str = ""
    expert_demo_path: str = ""
    output_dir: str = "outputs"
    offline_trajectory_limit: int = 0


@dataclass
class ExperimentConfig:
    env: EnvConfig
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    rlpd: RLPDConfig = field(default_factory=RLPDConfig)
    rlif: RLIFConfig = field(default_factory=RLIFConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            env=EnvConfig(**payload["env"]),
            expert=ExpertConfig(**payload.get("expert", {})),
            rlpd=RLPDConfig(**payload.get("rlpd", {})),
            rlif=RLIFConfig(**payload.get("rlif", {})),
            data=DataConfig(**payload.get("data", {})),
        )


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid config file: {path}")
    return ExperimentConfig.from_dict(payload)
