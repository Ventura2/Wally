from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import BaseModel, field_validator


class CurriculumConfig(BaseModel):
    stages: list[int] = [8, 16, 32, 64]
    loss_threshold: float = 0.01
    patience: int = 5
    mix_shorter_sequences: bool = True
    mix_ratio: float = 0.2
    shaping_weight: float = 0.1

    @field_validator("stages")
    @classmethod
    def _check_stages(cls, v: list[int]) -> list[int]:
        if len(v) == 0:
            raise ValueError("stages must be non-empty")
        if v != sorted(v):
            raise ValueError("stages must be sorted ascending")
        return v

    @field_validator("loss_threshold")
    @classmethod
    def _check_loss_threshold(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("loss_threshold must be greater than 0")
        return v

    @field_validator("patience")
    @classmethod
    def _check_patience(cls, v: int) -> int:
        if v < 1:
            raise ValueError("patience must be at least 1")
        return v

    @field_validator("mix_ratio")
    @classmethod
    def _check_mix_ratio(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("mix_ratio must be between 0 and 1")
        return v

    @field_validator("shaping_weight")
    @classmethod
    def _check_shaping_weight(cls, v: float) -> float:
        if v < 0:
            raise ValueError("shaping_weight must be non-negative")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> CurriculumConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    @classmethod
    def default(cls) -> CurriculumConfig:
        return cls()


class CurriculumTrainer:
    def __init__(
        self,
        config: CurriculumConfig,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        if device is not None:
            self.device = torch.device(device)
        else:
            default = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(default)
        self.current_stage: int = 0
        self.epoch_count: int = 0
        self.best_val_loss: float = float("inf")
        self.epochs_below_threshold: int = 0

    @property
    def current_horizon(self) -> int:
        return self.config.stages[self.current_stage]

    @property
    def is_complete(self) -> bool:
        return self.current_stage >= len(self.config.stages)

    def step(self, val_loss: float) -> bool:
        self.epoch_count += 1

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        if val_loss < self.config.loss_threshold:
            self.epochs_below_threshold += 1
        else:
            self.epochs_below_threshold = 0

        if self.epochs_below_threshold >= self.config.patience:
            self.current_stage += 1
            self.epochs_below_threshold = 0
            self.best_val_loss = float("inf")
            return True

        return False

    def slice_data(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizon = self.current_horizon
        frames = frames[:, :horizon]
        actions = actions[:, :horizon]

        if not self.config.mix_shorter_sequences or self.config.mix_ratio <= 0:
            return frames, actions

        B = frames.shape[0]
        n_mix = max(1, int(B * self.config.mix_ratio))
        mix_indices = torch.randperm(B, device=frames.device)[:n_mix]

        for idx in mix_indices:
            seg_len = torch.randint(2, horizon + 1, (1,)).item()
            max_offset = max(0, horizon - seg_len)
            if max_offset > 0:
                start = torch.randint(0, max_offset + 1, (1,)).item()
            else:
                start = 0
            end = min(start + seg_len, horizon)

            actual_frames = frames[idx, :end]
            actual_actions = actions[idx, :end]
            pad_len = horizon - actual_frames.shape[0]
            if pad_len > 0:
                actual_frames = torch.cat([
                    actual_frames,
                    actual_frames[-1:].expand(pad_len, *actual_frames.shape[1:]),
                ], dim=0)
                actual_actions = torch.cat([
                    actual_actions,
                    actual_actions[-1:].expand(pad_len, *actual_actions.shape[1:]),
                ], dim=0)
            frames[idx] = actual_frames[:horizon]
            actions[idx] = actual_actions[:horizon]

        return frames, actions

    def shaped_cost(
        self,
        trajectory: torch.Tensor,
        goal_latent: torch.Tensor,
        subgoal_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        final_latent = trajectory[:, -1]
        base_cost = torch.norm(final_latent - goal_latent, p=2, dim=-1)

        if subgoal_latents is not None:
            step_latents = trajectory[:, :-1]
            dists = torch.cdist(step_latents, subgoal_latents)
            min_dists = dists.min(dim=-1).values
            proximity = min_dists.sum(dim=-1)
            shaped_term = proximity
        else:
            shaped_term = base_cost.clone()

        w = self.config.shaping_weight
        cost = (1 - w) * base_cost + w * shaped_term
        return cost

    def save_state(self, path: str | Path) -> None:
        state = {
            "current_stage": self.current_stage,
            "epoch_count": self.epoch_count,
            "best_val_loss": self.best_val_loss,
            "epochs_below_threshold": self.epochs_below_threshold,
        }
        torch.save(state, path)

    def load_state(self, path: str | Path) -> None:
        state: dict[str, Any] = torch.load(path, weights_only=False)
        self.current_stage = state["current_stage"]
        self.epoch_count = state["epoch_count"]
        self.best_val_loss = state["best_val_loss"]
        self.epochs_below_threshold = state["epochs_below_threshold"]
