from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, field_validator

from wally.planner.config import CEMConfig
from wally.planner.gradient_mpc import GradientMPC, GradientMPCConfig
from wally.planner.high_level_planner import HighLevelPlanner, HighLevelPlannerConfig
from wally.planner.plan import GoalConditionedPlanner


@dataclass
class HierarchicalPlanResult:
    actions: torch.Tensor
    subgoals: torch.Tensor | None
    success: bool
    replan_count: int
    cost: float
    low_confidence: bool = False


class HierarchicalPlannerConfig(BaseModel):
    cem_config: CEMConfig = CEMConfig.default()
    high_level_config: HighLevelPlannerConfig = HighLevelPlannerConfig.default()
    gradient_mpc_config: GradientMPCConfig = GradientMPCConfig.default()
    subgoal_timeout: int = 50
    max_replans: int = 3
    reach_threshold: float = 1.0

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

    @field_validator("reach_threshold")
    @classmethod
    def _check_reach_threshold(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("reach_threshold must be positive")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> HierarchicalPlannerConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    @classmethod
    def default(cls) -> HierarchicalPlannerConfig:
        return cls()


class HierarchicalPlanner:
    def __init__(
        self,
        high_level_planner: HighLevelPlanner,
        low_level_planner: GoalConditionedPlanner | GradientMPC,
        config: HierarchicalPlannerConfig,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self._high_level = high_level_planner
        self._low_level = low_level_planner
        self._config = config
        self._device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    @property
    def config(self) -> HierarchicalPlannerConfig:
        return self._config

    @property
    def device(self) -> torch.device:
        return self._device

    def plan(
        self,
        current_frame: torch.Tensor,
        goal_frame: torch.Tensor,
    ) -> HierarchicalPlanResult:
        subgoal_latents, cost = self._high_level.plan_subgoals(
            current_frame, goal_frame,
        )
        targets = self._high_level.subgoals_to_targets(subgoal_latents)

        all_actions: list[torch.Tensor] = []
        total_cost = cost
        replan_count = 0

        i = 0
        while i < len(targets):
            target = targets[i]
            try:
                actions = self._low_level.plan_to_latent(
                    current_frame, target, return_cost=False,
                )
                all_actions.append(actions)
                i += 1
            except Exception:
                replan_result = self._high_level.replan(current_frame, goal_frame)
                if replan_result is None:
                    return HierarchicalPlanResult(
                        actions=(
                            torch.cat(all_actions, dim=0)
                            if all_actions
                            else torch.tensor([])
                        ),
                        subgoals=subgoal_latents,
                        success=False,
                        replan_count=replan_count,
                        cost=total_cost,
                        low_confidence=True,
                    )
                replan_count += 1
                subgoal_latents, new_cost = replan_result
                total_cost += new_cost
                targets = self._high_level.subgoals_to_targets(subgoal_latents)
                i = 0

        final_actions = (
            torch.cat(all_actions, dim=0) if all_actions else torch.tensor([])
        )
        return HierarchicalPlanResult(
            actions=final_actions,
            subgoals=subgoal_latents,
            success=True,
            replan_count=replan_count,
            cost=total_cost,
        )
