from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import yaml
from pydantic import BaseModel, field_validator

from wally.models.embedder import Embedder
from wally.models.encoder import ViTEncoder


class EnsembleConfig(BaseModel):
    ensemble_size: int = 3
    embed_dim: int = 192
    action_dim: int = 25
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    uncertainty_threshold: float = 1.0

    @field_validator("ensemble_size")
    @classmethod
    def _check_ensemble_size(cls, v: int) -> int:
        if v < 3 or v > 5:
            raise ValueError("ensemble_size must be between 3 and 5")
        return v

    @field_validator("uncertainty_threshold")
    @classmethod
    def _check_uncertainty_threshold(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("uncertainty_threshold must be positive")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> EnsembleConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    @classmethod
    def default(cls) -> EnsembleConfig:
        return cls()


class _MLPPredictor(nn.Module):
    def __init__(self, embed_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, action], dim=-1)
        return self.net(x)


class EnsembleWorldModel(nn.Module):
    def __init__(self, config: EnsembleConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ViTEncoder(pretrained=False)
        self.action_embedder = Embedder(
            input_dim=config.action_dim,
            smoothed_dim=config.embed_dim,
            emb_dim=config.embed_dim,
            mlp_scale=4,
        )
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.members = nn.ModuleList([
            _MLPPredictor(config.embed_dim, config.embed_dim, hidden_dim)
            for _ in range(config.ensemble_size)
        ])
        self._constraints: dict[str, Callable[[torch.Tensor], bool]] = {}

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(frames)
        return tokens.mean(dim=1)

    def _embed_actions(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize action shape to (B, 1, action_dim) and run through the Embedder."""
        if action.dim() == 2:
            action = action.unsqueeze(1)
        return self.action_embedder(action).squeeze(1)

    def predict_with_member(
        self, z: torch.Tensor, action: torch.Tensor, member_idx: int
    ) -> torch.Tensor:
        a_emb = self._embed_actions(action)
        return self.members[member_idx](z, a_emb)

    def predict_with_uncertainty(
        self, z: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = torch.stack([
            self.predict_with_member(z, action, i)
            for i in range(self.config.ensemble_size)
        ])
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0).mean(dim=-1)
        return mean, variance

    def train_step(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> dict[str, float]:
        a_emb = self._embed_actions(actions)
        losses: dict[str, float] = {}
        total = 0.0
        for i, member in enumerate(self.members):
            pred = member(latents, a_emb)
            loss = nn.functional.mse_loss(pred, target_latents)
            losses[f"member_{i}"] = loss.item()
            total += loss.item()
        losses["average"] = total / len(self.members)
        return losses

    def rollout_with_uncertainty(
        self, z_0: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, _ = actions.shape
        latents = [z_0]
        cum_uncertainty = torch.zeros(B, device=z_0.device)
        z = z_0
        for h in range(H):
            a_h = actions[:, h, :]
            z_next, var = self.predict_with_uncertainty(z, a_h)
            z_next = z_next.detach()
            cum_uncertainty = cum_uncertainty + var
            latents.append(z_next)
            z = z_next
        trajectory = torch.stack(latents, dim=1)
        return trajectory, cum_uncertainty

    def select_safe_plan(
        self,
        candidates: torch.Tensor,
        costs: torch.Tensor,
        uncertainties: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        threshold = self.config.uncertainty_threshold
        safe_mask = uncertainties <= threshold
        if safe_mask.any():
            safe_costs = costs.clone()
            safe_costs[~safe_mask] = float("inf")
            best_idx = safe_costs.argmin()
            return candidates[best_idx], False
        best_idx = uncertainties.argmin()
        return candidates[best_idx], True

    def register_constraint(
        self, name: str, constraint_fn: Callable[[torch.Tensor], bool]
    ) -> None:
        self._constraints[name] = constraint_fn

    def check_constraints(self, trajectory: torch.Tensor) -> bool:
        return all(fn(trajectory) for fn in self._constraints.values())

    def filter_by_constraints(self, trajectories: torch.Tensor) -> torch.Tensor:
        if not self._constraints:
            return trajectories
        mask = torch.ones(
            trajectories.shape[0], dtype=torch.bool, device=trajectories.device
        )
        for i in range(trajectories.shape[0]):
            if not self.check_constraints(trajectories[i]):
                mask[i] = False
        return trajectories[mask]
