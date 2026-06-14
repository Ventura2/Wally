from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import yaml
from pydantic import BaseModel, field_validator

from wally.planner.cem import CEMOptimizer
from wally.planner.config import CEMConfig
from wally.planner.rollout import LatentRollout


class GradientMPCConfig(BaseModel):
    learning_rate: float = 0.01
    n_refinement_steps: int = 10
    grad_clip_norm: float = 1.0
    warm_start: bool = True
    action_low: float = -1.0
    action_high: float = 1.0

    @field_validator("learning_rate")
    @classmethod
    def _check_learning_rate(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("learning_rate must be positive")
        return v

    @field_validator("n_refinement_steps")
    @classmethod
    def _check_n_refinement_steps(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_refinement_steps must be at least 1")
        return v

    @field_validator("grad_clip_norm")
    @classmethod
    def _check_grad_clip_norm(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("grad_clip_norm must be positive")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> GradientMPCConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    @classmethod
    def default(cls) -> GradientMPCConfig:
        return cls(
            learning_rate=0.01,
            n_refinement_steps=10,
            grad_clip_norm=1.0,
            warm_start=True,
            action_low=-1.0,
            action_high=1.0,
        )


class GradientMPC:
    def __init__(
        self,
        world_model: LatentRollout,
        encoder: Callable[[torch.Tensor], torch.Tensor],
        config: GradientMPCConfig,
        *,
        cem_config: CEMConfig | None = None,
        device: torch.device | str | None = None,
        cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        action_dim: int = 25,
    ) -> None:
        self._world_model = world_model
        self._encoder = encoder
        self._config = config
        self._cem_config = cem_config if cem_config is not None else CEMConfig.default()
        self._device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._cost_fn = cost_fn if cost_fn is not None else self._default_cost
        self._action_dim = action_dim
        self._cem = CEMOptimizer()
        self._warm_start_mean: torch.Tensor | None = None

    @staticmethod
    def _default_cost(z_H: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        return ((z_H - z_g) ** 2).sum(dim=-1)

    def set_warm_start_mean(self, mean: torch.Tensor) -> None:
        self._warm_start_mean = mean

    def clear_warm_start_mean(self) -> None:
        self._warm_start_mean = None

    def refine_actions(
        self,
        z_0: torch.Tensor,
        z_g: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        actions_refined = actions.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([actions_refined], lr=self._config.learning_rate)

        final_cost = float("inf")
        for _ in range(self._config.n_refinement_steps):
            optimizer.zero_grad()
            actions_batch = (
                actions_refined.unsqueeze(0)
                if actions_refined.dim() == 2
                else actions_refined
            )
            batch_size = actions_batch.shape[0]
            z_0_exp = z_0.expand(batch_size, -1)
            trajectory = self._world_model.rollout(z_0_exp, actions_batch)
            z_H = trajectory[:, -1, :]
            z_g_exp = z_g.expand(batch_size, -1)
            cost = self._cost_fn(z_H, z_g_exp).mean()
            cost.backward()
            if actions_refined.grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    [actions_refined], self._config.grad_clip_norm,
                )
            optimizer.step()
            with torch.no_grad():
                actions_refined.copy_(
                    actions_refined.clamp(
                        self._config.action_low, self._config.action_high,
                    )
                )
            final_cost = cost.item()

        return actions_refined.detach(), final_cost

    def plan(
        self,
        current_frame: torch.Tensor,
        goal_frame: torch.Tensor,
        *,
        return_cost: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, float]:
        current_frame, goal_frame, squeeze = self._normalize_frames(
            current_frame, goal_frame,
        )

        z_0 = self._encoder(current_frame).mean(dim=0, keepdim=True)
        z_g = self._encoder(goal_frame).mean(dim=0, keepdim=True)

        def cost_fn(actions: torch.Tensor) -> torch.Tensor:
            pop = actions.shape[0]
            z_0_exp = z_0.expand(pop, -1)
            z_g_exp = z_g.expand(pop, -1)
            trajectory = self._world_model.rollout(z_0_exp, actions)
            z_H = trajectory[:, -1, :]
            return self._cost_fn(z_H, z_g_exp)

        init_mean = None
        if self._config.warm_start and self._warm_start_mean is not None:
            init_mean = self._warm_start_mean

        cem_actions, _ = self._cem.optimize(
            cost_fn,
            horizon=self._cem_config.horizon,
            action_dim=self._action_dim,
            population_size=self._cem_config.population_size,
            elite_frac=self._cem_config.elite_frac,
            n_iterations=self._cem_config.n_iterations,
            action_low=self._config.action_low,
            action_high=self._config.action_high,
            init_mean=init_mean,
            device=self._device,
        )

        refined_actions, final_cost = self.refine_actions(z_0, z_g, cem_actions)

        if squeeze:
            if refined_actions.dim() == 3:
                refined_actions = refined_actions.squeeze(0)

        if return_cost:
            return refined_actions, final_cost
        return refined_actions

    def _normalize_frames(
        self,
        current_frame: torch.Tensor,
        goal_frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        if current_frame.dim() == 3:
            current_frame = current_frame.unsqueeze(0)
        if goal_frame.dim() == 3:
            goal_frame = goal_frame.unsqueeze(0)
        squeeze = current_frame.shape[0] == 1
        return current_frame, goal_frame, squeeze
