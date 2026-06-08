"""Minimal experiment logging utilities."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _is_wandb_run(obj: Any) -> bool:
    return hasattr(obj, "log") and hasattr(obj, "finish")


class MetricsLogger:
    def __init__(self, output_dir: str | Path, wandb_run: Any | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self.csv_path = self.output_dir / "metrics.csv"
        self._csv_header_written = self.csv_path.exists()
        self.wandb_run = wandb_run if _is_wandb_run(wandb_run) else None

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        record = {"step": step, **{key: self._json_safe(value) for key, value in metrics.items()}}
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(record)
        if self.wandb_run is not None:
            try:
                last = getattr(self, "_last_wandb_step", -1)
                step_to_log = int(step)
            except Exception:
                step_to_log = last + 1
            if step_to_log <= last:
                step_to_log = last + 1
            self._last_wandb_step = step_to_log
            try:
                self.wandb_run.log(record, step=step_to_log)
            except Exception:
                # If W&B fails for any reason, don't crash the training loop.
                pass

    def finish(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        if isinstance(value, Mapping):
            return {key: MetricsLogger._json_safe(item) for key, item in value.items()}
        return str(value)


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, Any]) -> None:
        payload = {key: MetricsLogger._json_safe(value) for key, value in record.items()}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
