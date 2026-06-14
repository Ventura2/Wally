from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml
from pydantic import BaseModel, field_validator


class CuriosityConfig(BaseModel):
    latent_dim: int = 192
    action_dim: int = 25
    hidden_dim: int = 128
    reward_scale: float = 1.0
    update_frequency: int = 1
    learning_rate: float = 1e-3

    @field_validator("reward_scale")
    @classmethod
    def _check_reward_scale(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("reward_scale must be greater than 0")
        return v

    @field_validator("update_frequency")
    @classmethod
    def _check_update_frequency(cls, v: int) -> int:
        if v < 1:
            raise ValueError("update_frequency must be at least 1")
        return v

    @field_validator("learning_rate")
    @classmethod
    def _check_learning_rate(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("learning_rate must be greater than 0")
        return v

    @field_validator("hidden_dim")
    @classmethod
    def _check_hidden_dim(cls, v: int) -> int:
        if v < 1:
            raise ValueError("hidden_dim must be at least 1")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> CuriosityConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    @classmethod
    def default(cls) -> CuriosityConfig:
        return cls()


class CuriosityModule(nn.Module):
    # ICM forward model: trained on (current, action, next) triples, not on
    # the LeWorldModel predictor's output. Unaffected by the residual-loss
    # contract change.

    def __init__(self, config: CuriosityConfig | None = None) -> None:
        super().__init__()
        self.config = config or CuriosityConfig.default()

        in_dim = self.config.latent_dim + self.config.action_dim
        self.forward_model = nn.Sequential(
            nn.Linear(in_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim),
        )

        self.optimizer = torch.optim.Adam(
            self.forward_model.parameters(),
            lr=self.config.learning_rate,
        )

    def compute_intrinsic_reward(
        self,
        current_latent: torch.Tensor,
        action: torch.Tensor,
        next_latent: torch.Tensor,
    ) -> torch.Tensor:
        predicted = self.forward_model(torch.cat([current_latent, action], dim=-1))
        error = torch.norm(predicted - next_latent, p=2, dim=-1)
        return self.config.reward_scale * error

    def train_step(
        self,
        current_latents: torch.Tensor,
        actions: torch.Tensor,
        next_latents: torch.Tensor,
    ) -> float:
        self.optimizer.zero_grad()
        predicted = self.forward_model(torch.cat([current_latents, actions], dim=-1))
        loss = nn.functional.mse_loss(predicted, next_latents)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_priority(
        self,
        current_latents: torch.Tensor,
        actions: torch.Tensor,
        next_latents: torch.Tensor,
    ) -> torch.Tensor:
        predicted = self.forward_model(torch.cat([current_latents, actions], dim=-1))
        return torch.norm(predicted - next_latents, p=2, dim=-1)
