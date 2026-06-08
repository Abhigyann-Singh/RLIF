"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location=map_location)
    except Exception as error:  # Retry in case torch changed default weights_only behavior
        try:
            return torch.load(Path(path), map_location=map_location, weights_only=False)  # type: ignore[call-arg]
        except Exception:
            raise error
