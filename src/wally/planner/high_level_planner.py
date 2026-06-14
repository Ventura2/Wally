from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import yaml
from pydantic import BaseModel, field_validator

from wally.planner.cem import CEMOptimizer
from wally.planner.protocols import WorldModelProtocol
from wally.planner.rollout import LatentRollout


class HighLevelPlannerConfig(BaseModel):
    macro_horizon: int = 5
    macro_action_dim: int = 25
    population_size: int = 32
    elite_frac: float = 0.1
    n_iterations: int = 5
    subgoal_timeout: int = 50
    max_replans: int = 3
    action_low: float = -1.0
    action_high: float = 1.0

    @field_validator("macro_horizon")
    @classmethod
    def _check_macro_horizon(cls, v: int) -> int:
        if not (5 <= v <= 10):
            raise ValueError("macro_horizon must be in the range [5, 10]")
        return v

    @field_validator("population_size")
    @classmethod
    def _check_population_size(cls, v: int) -> int:
        if v <= 1:
            raise ValueError("population_size must be greater than 1")
        return v

    @field_validator("elite_frac")
    @classmethod
    def _check_elite_frac(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("elite_frac must be in the open interval (0, 1)")
        return v

    @field_validator("n_iterations")
    @classmethod
    def _check_n_iterations(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_iterations must be at least 1")
        return v

    @field_validator("subgoal_timeout")
    @classmethod
    def _check_subgoal_timeout(cls, v: int) -> int:
        if v < 1:
            raise ValueError("subgoal_timeout must be at least 1")
        return v

    @field_validator("max_replans")
    @classmethod
    def _check_max_replans(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_replans must be non-negative")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> HighLevelPlannerConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    @classmethod
    def default(cls) -> HighLevelPlannerConfig:
        return cls()


class HighLevelWorldModel(nn.Module):
    # Consumes the encoder's projected output (via the encoder callable),
    # not the LeWorldModel predictor's output. Unaffected by the
    # residual-loss contract change.

    def __init__(
        self,
        encoder: Callable[[torch.Tensor], torch.Tensor],
        latent_dim: int,
        action_dim: int = 25,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self._encoder_fn = encoder
        self._predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        return self._encoder_fn(frame)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, action], dim=-1)
        return self._predictor(x)


class SubgoalExecutionResult:
    def __init__(
        self,
        success: bool,
        completed_subgoals: int,
        total_subgoals: int,
        steps_per_subgoal: list[int],
        replan_count: int = 0,
        failed: bool = False,
    ) -> None:
        self.success = success
        self.completed_subgoals = completed_subgoals
        self.total_subgoals = total_subgoals
        self.steps_per_subgoal = steps_per_subgoal
        self.replan_count = replan_count
        self.failed = failed


def train_high_level_model(
    encoder: Callable[[torch.Tensor], torch.Tensor],
    start_latents: torch.Tensor,
    macro_actions: torch.Tensor,
    end_latents: torch.Tensor,
    *,
    latent_dim: int | None = None,
    action_dim: int = 25,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    epochs: int = 100,
) -> HighLevelWorldModel:
    if latent_dim is None:
        latent_dim = start_latents.shape[-1]

    model = HighLevelWorldModel(
        encoder=encoder,
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model._predictor.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        predicted = model._predictor(
            torch.cat([start_latents, macro_actions], dim=-1)
        )
        loss = loss_fn(predicted, end_latents)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


class HighLevelPlanner:
    def __init__(
        self,
        high_level_model: WorldModelProtocol,
        encoder: Callable[[torch.Tensor], torch.Tensor],
        config: HighLevelPlannerConfig,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self._model = high_level_model
        self._encoder = encoder
        self._config = config
        self._device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._cem = CEMOptimizer()
        self._replan_count = 0

    @property
    def config(self) -> HighLevelPlannerConfig:
        return self._config

    @property
    def device(self) -> torch.device:
        return self._device

    def plan_subgoals(
        self,
        current_frame: torch.Tensor,
        goal_frame: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if current_frame.dim() == 3:
            current_frame = current_frame.unsqueeze(0)
        if goal_frame.dim() == 3:
            goal_frame = goal_frame.unsqueeze(0)

        z_0 = self._encoder(current_frame).mean(dim=0, keepdim=True)
        z_g = self._encoder(goal_frame).mean(dim=0, keepdim=True)

        rollout = LatentRollout(model=self._model, device=self._device)

        def cost_fn(actions: torch.Tensor) -> torch.Tensor:
            pop = actions.shape[0]
            z_0_exp = z_0.expand(pop, -1)
            z_g_exp = z_g.expand(pop, -1)
            trajectory = rollout.rollout(z_0_exp, actions)
            z_final = trajectory[:, -1, :]
            return ((z_final - z_g_exp) ** 2).sum(dim=-1)

        actions, cost_history = self._cem.optimize(
            cost_fn,
            horizon=self._config.macro_horizon,
            action_dim=self._config.macro_action_dim,
            population_size=self._config.population_size,
            elite_frac=self._config.elite_frac,
            n_iterations=self._config.n_iterations,
            action_low=self._config.action_low,
            action_high=self._config.action_high,
            device=self._device,
        )

        z_0_single = z_0
        full_trajectory = rollout.rollout(
            z_0_single, actions.unsqueeze(0)
        )
        subgoal_latents = full_trajectory[0, 1:, :]

        cost = cost_history[-1] if cost_history else 0.0
        return subgoal_latents, cost

    def subgoals_to_targets(
        self, subgoal_latents: torch.Tensor
    ) -> list[torch.Tensor]:
        return [subgoal_latents[i] for i in range(subgoal_latents.shape[0])]

    def execute_subgoals(
        self,
        subgoal_targets: list[torch.Tensor],
        low_level_planner: Callable[..., torch.Tensor],
        current_frame: torch.Tensor,
        *,
        reach_threshold: float = 1.0,
        encode_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> SubgoalExecutionResult:
        steps_per_subgoal: list[int] = []
        completed = 0

        for i, target in enumerate(subgoal_targets):
            steps = 0
            reached = False

            while steps < self._config.subgoal_timeout:
                low_level_planner(current_frame, target)

                if encode_fn is not None:
                    z_current = encode_fn(current_frame.unsqueeze(0)).squeeze(0)
                    dist = ((z_current - target) ** 2).sum().item()
                else:
                    dist = 0.0

                steps += 1

                if dist < reach_threshold:
                    reached = True
                    break

            steps_per_subgoal.append(steps)

            if reached:
                completed += 1
            else:
                return SubgoalExecutionResult(
                    success=False,
                    completed_subgoals=completed,
                    total_subgoals=len(subgoal_targets),
                    steps_per_subgoal=steps_per_subgoal,
                    replan_count=self._replan_count,
                    failed=True,
                )

        return SubgoalExecutionResult(
            success=True,
            completed_subgoals=completed,
            total_subgoals=len(subgoal_targets),
            steps_per_subgoal=steps_per_subgoal,
            replan_count=self._replan_count,
        )

    def replan(
        self,
        current_frame: torch.Tensor,
        goal_frame: torch.Tensor,
    ) -> tuple[torch.Tensor, float] | None:
        if self._replan_count >= self._config.max_replans:
            return None

        self._replan_count += 1
        return self.plan_subgoals(current_frame, goal_frame)

    def reset_replan_count(self) -> None:
        self._replan_count = 0
