from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator


class CEMConfig(BaseModel):
    population_size: int = 64
    elite_frac: float = 0.1
    n_iterations: int = 5
    horizon: int = 8
    action_low: float = -1.0
    action_high: float = 1.0
    gradient_policy: Literal["detach", "straight_through"] = "detach"
    inventory_stall_penalty: float = 5e-2

    @field_validator("elite_frac")
    @classmethod
    def _check_elite_frac(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("elite_frac must be in the open interval (0, 1)")
        return v

    @field_validator("population_size")
    @classmethod
    def _check_population_size(cls, v: int) -> int:
        if v <= 1:
            raise ValueError("population_size must be greater than 1")
        return v

    @field_validator("n_iterations")
    @classmethod
    def _check_n_iterations(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_iterations must be at least 1")
        return v

    @field_validator("horizon")
    @classmethod
    def _check_horizon(cls, v: int) -> int:
        if v < 1:
            raise ValueError("horizon must be at least 1")
        return v

    @field_validator("inventory_stall_penalty")
    @classmethod
    def _check_inventory_stall_penalty(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("inventory_stall_penalty must be >= 0")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> CEMConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    @classmethod
    def default(cls) -> CEMConfig:
        return cls(
            population_size=64,
            elite_frac=0.1,
            n_iterations=5,
            horizon=8,
            action_low=-1.0,
            action_high=1.0,
            gradient_policy="detach",
            inventory_stall_penalty=5e-2,
        )
