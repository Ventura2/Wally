from __future__ import annotations

from pathlib import Path
from typing import Tuple

import yaml
from pydantic import BaseModel, field_validator


class AgentConfig(BaseModel):
    replan_interval: int = 4
    episode_timeout: int = 1000
    resize: Tuple[int, int] = (64, 64)
    action_vocab_path: Path | None = None
    record_trajectory: bool = False

    @field_validator("replan_interval")
    @classmethod
    def _check_replan_interval(cls, v: int) -> int:
        if v < 1:
            raise ValueError("replan_interval must be at least 1")
        return v

    @field_validator("episode_timeout")
    @classmethod
    def _check_episode_timeout(cls, v: int) -> int:
        if v < 1:
            raise ValueError("episode_timeout must be at least 1")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})
