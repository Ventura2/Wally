"""Shared planner factory used by ``wally-play`` and ``wally-deploy``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from wally.planner.config import CEMConfig
from wally.planner.gradient_mpc import GradientMPC, GradientMPCConfig
from wally.planner.hierarchical_planner import (
    HierarchicalPlanner,
    HierarchicalPlannerConfig,
)
from wally.planner.high_level_planner import (
    HighLevelPlanner,
    HighLevelPlannerConfig,
)
from wally.planner.plan import GoalConditionedPlanner
from wally.planner.rollout import LatentRollout

if TYPE_CHECKING:
    from wally.agent.protocol import PlannerProtocol


def build_planner(
    planner_kind: str,
    rollout: LatentRollout,
    encoder: torch.nn.Module,
) -> "PlannerProtocol":
    """Build a planner matching ``planner_kind``.

    Args:
        planner_kind: One of ``"cem"``, ``"gradient"``, ``"hierarchical"``.
        rollout: A :class:`LatentRollout` for the trained LeWorldModel.
        encoder: The encoder module (typically ``rollout._model.encode``).

    Returns:
        An object implementing :class:`agent.protocol.PlannerProtocol`.

    Raises:
        ValueError: If ``planner_kind`` is not one of the three supported kinds.
    """
    cem_config = CEMConfig.default()
    gradient_mpc_config = GradientMPCConfig.default()
    high_level_config = HighLevelPlannerConfig.default()
    hier_config = HierarchicalPlannerConfig.default()

    if planner_kind == "cem":
        from wally.agent.protocol import FlatPlannerAdapter

        cem_config = cem_config.model_copy(
            update={
                "inventory_stall_penalty": 0.25,
                "diversity_penalty": 1.0e-3,
                "camera_still_penalty": 1.0e-3,
            }
        )
        planner = GoalConditionedPlanner(rollout, encoder, cem_config)
        return FlatPlannerAdapter(planner)

    if planner_kind == "gradient":
        from wally.agent.protocol import FlatPlannerAdapter

        planner = GradientMPC(rollout, encoder, gradient_mpc_config)
        return FlatPlannerAdapter(planner)

    if planner_kind == "hierarchical":
        from wally.agent.protocol import HierarchicalPlannerAdapter

        high_level = HighLevelPlanner(rollout._model, encoder, high_level_config)
        cem_config = cem_config.model_copy(
            update={"inventory_stall_penalty": 0.25}
        )
        low_level = GoalConditionedPlanner(rollout, encoder, cem_config)
        planner = HierarchicalPlanner(high_level, low_level, hier_config)
        return HierarchicalPlannerAdapter(planner)

    raise ValueError(
        f"Unknown planner kind: {planner_kind!r}. "
        f"Expected one of: 'cem', 'gradient', 'hierarchical'."
    )
