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

    relay_enabled: bool = False
    relay_port: int = 8081
    relay_host: str = "0.0.0.0"
    relay_max_width: int = 640
    relay_max_height: int = 360
    relay_jpeg_quality: int = 80
    relay_min_frame_interval_ms: int = 33

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

    @field_validator("relay_port")
    @classmethod
    def _check_relay_port(cls, v: int) -> int:
        if not 0 < v < 65536:
            raise ValueError("relay_port must be in (0, 65536)")
        return v

    @field_validator("relay_max_width", "relay_max_height")
    @classmethod
    def _check_relay_dims(cls, v: int) -> int:
        if v < 1:
            raise ValueError("relay_max_width/relay_max_height must be >= 1")
        return v

    @field_validator("relay_jpeg_quality")
    @classmethod
    def _check_relay_jpeg_quality(cls, v: int) -> int:
        if not 0 < v <= 100:
            raise ValueError("relay_jpeg_quality must be in (0, 100]")
        return v

    @field_validator("relay_min_frame_interval_ms")
    @classmethod
    def _check_relay_min_frame_interval_ms(cls, v: int) -> int:
        if v < 0:
            raise ValueError("relay_min_frame_interval_ms must be >= 0")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})
