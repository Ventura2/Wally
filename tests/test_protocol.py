from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest
import torch

from wally.agent.protocol import (
    EpisodeResult,
    FlatPlannerAdapter,
    HierarchicalPlannerAdapter,
    PlannerProtocol,
    PlanResult,
)
from wally.planner.hierarchical_planner import HierarchicalPlanResult


class TestPlanResult:
    def test_defaults(self):
        actions = torch.randn(8, 25)
        result = PlanResult(actions=actions, cost=1.5)
        assert result.actions is actions
        assert result.cost == 1.5
        assert result.subgoals is None
        assert result.success is True
        assert result.replan_count == 0
        assert result.low_confidence is False

    def test_immutability(self):
        result = PlanResult(actions=torch.randn(8, 25))
        with pytest.raises(FrozenInstanceError):
            result.success = False


class TestEpisodeResult:
    def test_defaults(self):
        result = EpisodeResult(steps=100, final_cost=2.5, duration_seconds=10.0)
        assert result.steps == 100
        assert result.final_cost == 2.5
        assert result.duration_seconds == 10.0
        assert result.trajectory is None
        assert result.interrupted is False


class TestFlatPlannerAdapter:
    def test_with_goal_conditioned_planner(self):
        mock_planner = MagicMock()
        expected_actions = torch.randn(8, 25)
        mock_planner.plan.return_value = (expected_actions, 0.5)

        adapter = FlatPlannerAdapter(mock_planner)
        current = torch.randn(3, 64, 64)
        goal = torch.randn(3, 64, 64)
        result = adapter.plan(current, goal)

        assert isinstance(result, PlanResult)
        assert result.actions is expected_actions
        assert result.actions.shape == (8, 25)
        assert result.cost == 0.5
        assert result.subgoals is None
        assert result.success is True
        assert result.replan_count == 0
        assert result.low_confidence is False
        mock_planner.plan.assert_called_once_with(current, goal, return_cost=True)

    def test_with_gradient_mpc(self):
        mock_planner = MagicMock()
        expected_actions = torch.randn(8, 25)
        mock_planner.plan.return_value = (expected_actions, 1.2)

        adapter = FlatPlannerAdapter(mock_planner)
        current = torch.randn(3, 64, 64)
        goal = torch.randn(3, 64, 64)
        result = adapter.plan(current, goal)

        assert isinstance(result, PlanResult)
        assert result.actions is expected_actions
        assert result.cost == 1.2
        mock_planner.plan.assert_called_once_with(current, goal, return_cost=True)

    def test_set_warm_start_mean(self):
        mock_planner = MagicMock()
        adapter = FlatPlannerAdapter(mock_planner)
        mean = torch.randn(8, 25)

        adapter.set_warm_start_mean(mean)
        mock_planner.set_warm_start_mean.assert_called_once_with(mean)


class TestHierarchicalPlannerAdapter:
    def test_maps_all_fields(self):
        mock_planner = MagicMock()
        actions = torch.randn(8, 25)
        subgoals = torch.randn(3, 8)
        mock_planner.plan.return_value = HierarchicalPlanResult(
            actions=actions,
            subgoals=subgoals,
            success=True,
            replan_count=2,
            cost=3.5,
            low_confidence=True,
        )

        adapter = HierarchicalPlannerAdapter(mock_planner)
        current = torch.randn(3, 64, 64)
        goal = torch.randn(3, 64, 64)
        result = adapter.plan(current, goal)

        assert isinstance(result, PlanResult)
        assert result.actions is actions
        assert result.subgoals is subgoals
        assert result.success is True
        assert result.replan_count == 2
        assert result.cost == 3.5
        assert result.low_confidence is True


class TestPlannerProtocol:
    def test_flat_adapter_satisfies_protocol(self):
        mock_planner = MagicMock()
        adapter = FlatPlannerAdapter(mock_planner)
        assert isinstance(adapter, PlannerProtocol)

    def test_hierarchical_adapter_satisfies_protocol(self):
        mock_planner = MagicMock()
        adapter = HierarchicalPlannerAdapter(mock_planner)
        assert isinstance(adapter, PlannerProtocol)
