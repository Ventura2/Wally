from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from wally.planner.gradient_mpc import GradientMPC
from wally.planner.hierarchical_planner import HierarchicalPlanner
from wally.planner.plan import GoalConditionedPlanner


@dataclass(frozen=True)
class PlanResult:
    actions: torch.Tensor
    subgoals: torch.Tensor | None = None
    success: bool = True
    cost: float = 0.0
    replan_count: int = 0
    low_confidence: bool = False


@dataclass
class EpisodeResult:
    steps: int
    final_cost: float
    duration_seconds: float
    trajectory: dict[str, np.ndarray] | None = None
    interrupted: bool = False


@runtime_checkable
class PlannerProtocol(Protocol):
    def plan(self, current_frame: torch.Tensor, goal_frame: torch.Tensor) -> PlanResult:
        ...


class FlatPlannerAdapter:
    def __init__(self, planner: GoalConditionedPlanner | GradientMPC) -> None:
        self._planner = planner

    def plan(self, current_frame: torch.Tensor, goal_frame: torch.Tensor) -> PlanResult:
        actions, cost = self._planner.plan(current_frame, goal_frame, return_cost=True)
        return PlanResult(actions=actions, cost=cost)

    def set_warm_start_mean(self, mean: torch.Tensor) -> None:
        self._planner.set_warm_start_mean(mean)


class HierarchicalPlannerAdapter:
    def __init__(self, planner: HierarchicalPlanner) -> None:
        self._planner = planner

    def plan(self, current_frame: torch.Tensor, goal_frame: torch.Tensor) -> PlanResult:
        result = self._planner.plan(current_frame, goal_frame)
        return PlanResult(
            actions=result.actions,
            subgoals=result.subgoals,
            success=result.success,
            cost=result.cost,
            replan_count=result.replan_count,
            low_confidence=result.low_confidence,
        )
