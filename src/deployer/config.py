from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class SafetyConfig(BaseModel):
    prevent_bedrock_breaking: bool = True
    prevent_lava_interaction: bool = True
    prevent_void_fall: bool = True
    void_threshold: float = -64.0
    action_cooldown_ms: int = 100


class ReconnectConfig(BaseModel):
    max_attempts: int = 10
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 60.0
    backoff_multiplier: float = 2.0

    @field_validator("max_attempts")
    @classmethod
    def _check_max_attempts(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_attempts must be >= 1")
        return v


class DeployConfig(BaseModel):
    server_host: str = "localhost"
    server_port: int = 25565
    auth_mode: Literal["online", "offline"] = "offline"
    username: str = "WallyAgent"
    checkpoint_path: str = "checkpoints/latest.pt"
    goal_frame_path: str = ""
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    reconnect: ReconnectConfig = Field(default_factory=ReconnectConfig)
    log_dir: str = "logs/deploy"
    log_to_stdout: bool = False
    record_trajectory: bool = False
    output_dir: str = "data/recordings"
    render_distance: int = 4

    @field_validator("server_port")
    @classmethod
    def _check_server_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("server_port must be between 1 and 65535")
        return v

    @field_validator("render_distance")
    @classmethod
    def _check_render_distance(cls, v: int) -> int:
        if not (1 <= v <= 32):
            raise ValueError("render_distance must be between 1 and 32")
        return v

    @field_validator("username")
    @classmethod
    def _check_username(cls, v: str) -> str:
        if not v:
            raise ValueError("username must be non-empty")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> DeployConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    @classmethod
    def default(cls) -> DeployConfig:
        return cls()
