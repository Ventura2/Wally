"""Tests for the shared ``build_planner`` factory used by both CLIs."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.planner_factory import build_planner
from agent.protocol import (
    FlatPlannerAdapter,
    HierarchicalPlannerAdapter,
    PlannerProtocol,
)


def _make_rollout() -> MagicMock:
    rollout = MagicMock()
    rollout._model = MagicMock()
    rollout._model.encode = MagicMock()
    return rollout


class TestBuildPlanner:
    def test_cem_returns_flat_adapter(self) -> None:
        rollout = _make_rollout()
        encoder = MagicMock()
        planner = build_planner("cem", rollout, encoder)
        assert isinstance(planner, FlatPlannerAdapter)
        assert isinstance(planner, PlannerProtocol)

    def test_gradient_returns_flat_adapter(self) -> None:
        rollout = _make_rollout()
        encoder = MagicMock()
        planner = build_planner("gradient", rollout, encoder)
        assert isinstance(planner, FlatPlannerAdapter)
        assert isinstance(planner, PlannerProtocol)

    def test_hierarchical_returns_hierarchical_adapter(self) -> None:
        rollout = _make_rollout()
        encoder = MagicMock()
        planner = build_planner("hierarchical", rollout, encoder)
        assert isinstance(planner, HierarchicalPlannerAdapter)
        assert isinstance(planner, PlannerProtocol)

    def test_unknown_kind_raises_value_error(self) -> None:
        rollout = _make_rollout()
        encoder = MagicMock()
        with pytest.raises(ValueError, match="Unknown planner kind"):
            build_planner("random", rollout, encoder)

    def test_unknown_kind_message_lists_choices(self) -> None:
        rollout = _make_rollout()
        encoder = MagicMock()
        with pytest.raises(ValueError) as exc_info:
            build_planner("bogus", rollout, encoder)
        assert "cem" in str(exc_info.value)
        assert "gradient" in str(exc_info.value)
        assert "hierarchical" in str(exc_info.value)

    @pytest.mark.parametrize("planner_kind", ["cem", "gradient", "hierarchical"])
    def test_satisfies_planner_protocol(self, planner_kind: str) -> None:
        rollout = _make_rollout()
        encoder = MagicMock()
        planner = build_planner(planner_kind, rollout, encoder)
        assert isinstance(planner, PlannerProtocol)
