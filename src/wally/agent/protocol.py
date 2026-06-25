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


class HierarchicalEmbeddingPlannerAdapter:
    """Adapter for :class:`HierarchicalEmbeddingPlanner`.

    Implements the streaming-embedding protocol: on every
    :meth:`push_l0_state` call, the agent loop notifies the planner of
    the latest L0 state embedding so the upper layers' runtimes can
    compute drift. On every :meth:`plan` call, the planner reads the
    latest L1 target embedding and uses it as the L0 planner's
    ``target_embedding``.

    The class also exposes ``set_target_embedding`` so the agent loop
    can pass a top-level goal vector to the hierarchy.
    """

    def __init__(self, planner: "HierarchicalEmbeddingPlanner") -> None:  # noqa: F821
        self._planner = planner

    def plan(
        self,
        current_frame: torch.Tensor,
        goal_frame: torch.Tensor | None = None,
    ) -> PlanResult:
        result = self._planner.plan(current_frame)
        return PlanResult(
            actions=result.actions,
            subgoals=(
                result.subgoals if isinstance(result.subgoals, torch.Tensor) else None
            ),
            success=result.success,
            cost=result.cost,
            replan_count=result.replan_count,
            low_confidence=result.low_confidence,
        )

    def plan_with_target(
        self,
        current_frame: torch.Tensor,
        target_embedding: torch.Tensor,
    ) -> PlanResult:
        result = self._planner.plan(current_frame, target_embedding=target_embedding)
        return PlanResult(
            actions=result.actions,
            subgoals=(
                result.subgoals if isinstance(result.subgoals, torch.Tensor) else None
            ),
            success=result.success,
            cost=result.cost,
            replan_count=result.replan_count,
            low_confidence=result.low_confidence,
        )

    def set_target_embedding(self, target_embedding: torch.Tensor) -> None:
        self._planner.set_goal(target_embedding)

    def tick_with_frame(self, frame: torch.Tensor) -> None:
        """Encode a frame through the lowest layer's encoder and push.

        The agent loop calls this every step when a hierarchical
        planner is in use. The encoding is done in the planner so the
        agent loop does not need to know which encoder corresponds to
        the lowest layer.
        """
        state = self._planner.encode_for_lowest_layer(frame)
        self._planner.push_l0_state(state)

    def push_l0_state(self, state_embedding: torch.Tensor) -> None:
        """Direct push of a pre-encoded state embedding.

        Most callers should use :meth:`tick_with_frame`; this entry
        point is for tests and special cases.
        """
        self._planner.push_l0_state(state_embedding)
