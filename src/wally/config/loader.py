from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import yaml

from wally.config.model import ModelConfig
from wally.config.training import TrainConfig


def _apply_section(data: dict[str, object], cls: type) -> object:
    valid = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid}
    return cls(**filtered)


def load_config(config_path: str | Path) -> tuple[TrainConfig, ModelConfig]:
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    model_data = raw.get("model", {}) or {}
    training_data = raw.get("training", {}) or {}

    model_cfg = _apply_section(model_data, ModelConfig)
    train_cfg = _apply_section(training_data, TrainConfig)

    return train_cfg, model_cfg
