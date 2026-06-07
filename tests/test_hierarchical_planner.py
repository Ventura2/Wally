from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from pydantic import ValidationError

from wally.planner.config import CEMConfig
from wally.planner.hierarchical_planner import (
    HierarchicalPlanner,
    HierarchicalPlannerConfig,
    HierarchicalPlanResult,
)
from wally.planner.high_level_planner import HighLevelPlanner, HighLevelPlannerConfig
from wally.planner.plan import GoalConditionedPlanner


def _make_mock_rollout(latent_dim: int = 8) -> MagicMock:
    mock = MagicMock()

    def rollout(z_0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, H, _ = actions.shape
        trajectory = torch.randn(B, H + 1, latent_dim)
        trajectory[:, 0, :] = z_0
        return trajectory

    mock.rollout = MagicMock(side_effect=rollout)
    return mock


def _make_encoder(latent_dim: int = 8) -> MagicMock:
    def encoder(frame: torch.Tensor) -> torch.Tensor:
        B = frame.shape[0]
        return torch.randn(B, latent_dim)

    return MagicMock(side_effect=encoder)


def _make_high_level_planner(
    latent_dim: int = 8,
    n_subgoals: int = 3,
    max_replans: int = 3,
) -> HighLevelPlanner:
    model = MagicMock()
    model.encode = MagicMock(return_value=torch.randn(1, latent_dim))

    def predict(z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(z)

    model.predict = MagicMock(side_effect=predict)
    encoder = _make_encoder(latent_dim)
    cfg = HighLevelPlannerConfig(
        macro_horizon=5,
        population_size=8,
        n_iterations=2,
        max_replans=max_replans,
    )
    return HighLevelPlanner(model, encoder, cfg, device="cpu")


def _make_low_level_planner(
    latent_dim: int = 8, horizon: int = 4,
) -> GoalConditionedPlanner:
    rollout = _make_mock_rollout(latent_dim)
    encoder = _make_encoder(latent_dim)
    cfg = CEMConfig(population_size=8, n_iterations=2, horizon=horizon)
    return GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")


class TestHierarchicalPlannerConfig:
    def test_default_values(self):
        cfg = HierarchicalPlannerConfig.default()
        assert cfg.subgoal_timeout == 50
        assert cfg.max_replans == 3
        assert cfg.reach_threshold == 1.0
        assert isinstance(cfg.cem_config, CEMConfig)
        assert isinstance(cfg.high_level_config, HighLevelPlannerConfig)

    def test_subgoal_timeout_zero_fails(self):
        with pytest.raises(ValidationError, match="subgoal_timeout"):
            HierarchicalPlannerConfig(subgoal_timeout=0)

    def test_subgoal_timeout_negative_fails(self):
        with pytest.raises(ValidationError, match="subgoal_timeout"):
            HierarchicalPlannerConfig(subgoal_timeout=-1)

    def test_max_replans_negative_fails(self):
        with pytest.raises(ValidationError, match="max_replans"):
            HierarchicalPlannerConfig(max_replans=-1)

    def test_reach_threshold_zero_fails(self):
        with pytest.raises(ValidationError, match="reach_threshold"):
            HierarchicalPlannerConfig(reach_threshold=0.0)

    def test_reach_threshold_negative_fails(self):
        with pytest.raises(ValidationError, match="reach_threshold"):
            HierarchicalPlannerConfig(reach_threshold=-0.5)

    def test_valid_custom_values(self):
        cfg = HierarchicalPlannerConfig(
            subgoal_timeout=100, max_replans=5, reach_threshold=2.0,
        )
        assert cfg.subgoal_timeout == 100
        assert cfg.max_replans == 5
        assert cfg.reach_threshold == 2.0

    def test_from_yaml(self, tmp_path: Path):
        yaml_file = tmp_path / "hier.yaml"
        yaml_file.write_text(
            "subgoal_timeout: 80\nmax_replans: 4\nreach_threshold: 1.5\n"
        )
        cfg = HierarchicalPlannerConfig.from_yaml(yaml_file)
        assert cfg.subgoal_timeout == 80
        assert cfg.max_replans == 4
        assert cfg.reach_threshold == 1.5

    def test_from_yaml_empty_uses_defaults(self, tmp_path: Path):
        yaml_file = tmp_path / "hier.yaml"
        yaml_file.write_text("")
        cfg = HierarchicalPlannerConfig.from_yaml(yaml_file)
        assert cfg == HierarchicalPlannerConfig.default()


class TestHierarchicalPlanResult:
    def test_all_fields(self):
        actions = torch.randn(4, 25)
        subgoals = torch.randn(3, 8)
        result = HierarchicalPlanResult(
            actions=actions,
            subgoals=subgoals,
            success=True,
            replan_count=0,
            cost=1.5,
            low_confidence=False,
        )
        assert result.actions is actions
        assert result.subgoals is subgoals
        assert result.success is True
        assert result.replan_count == 0
        assert result.cost == 1.5
        assert result.low_confidence is False

    def test_default_low_confidence(self):
        result = HierarchicalPlanResult(
            actions=torch.tensor([]),
            subgoals=None,
            success=False,
            replan_count=3,
            cost=10.0,
        )
        assert result.low_confidence is False

    def test_failure_result(self):
        result = HierarchicalPlanResult(
            actions=torch.tensor([]),
            subgoals=torch.randn(3, 8),
            success=False,
            replan_count=3,
            cost=50.0,
            low_confidence=True,
        )
        assert result.success is False
        assert result.low_confidence is True
        assert result.replan_count == 3


class TestHierarchicalPlannerPlan:
    def test_full_pipeline_success(self):
        hl_planner = _make_high_level_planner()
        ll_planner = _make_low_level_planner()
        cfg = HierarchicalPlannerConfig(max_replans=3)
        planner = HierarchicalPlanner(hl_planner, ll_planner, cfg, device="cpu")

        current_frame = torch.randn(3, 64, 64)
        goal_frame = torch.randn(3, 64, 64)

        result = planner.plan(current_frame, goal_frame)

        assert isinstance(result, HierarchicalPlanResult)
        assert result.success is True
        assert result.replan_count == 0
        assert isinstance(result.actions, torch.Tensor)
        assert result.actions.dim() == 2
        assert result.subgoals is not None

    def test_subgoal_failure_triggers_replan(self):
        hl_planner = _make_high_level_planner()
        ll_planner = _make_low_level_planner()
        cfg = HierarchicalPlannerConfig(max_replans=3)
        planner = HierarchicalPlanner(hl_planner, ll_planner, cfg, device="cpu")

        call_count = [0]
        original_plan_to_latent = ll_planner.plan_to_latent

        def failing_plan_to_latent(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("planner failure")
            return original_plan_to_latent(*args, **kwargs)

        ll_planner.plan_to_latent = MagicMock(side_effect=failing_plan_to_latent)

        current_frame = torch.randn(3, 64, 64)
        goal_frame = torch.randn(3, 64, 64)

        result = planner.plan(current_frame, goal_frame)

        assert result.success is True
        assert result.replan_count == 1

    def test_max_replans_exceeded_returns_failure(self):
        hl_planner = _make_high_level_planner(max_replans=2)
        ll_planner = _make_low_level_planner()
        cfg = HierarchicalPlannerConfig(max_replans=2)
        planner = HierarchicalPlanner(hl_planner, ll_planner, cfg, device="cpu")

        ll_planner.plan_to_latent = MagicMock(
            side_effect=RuntimeError("always fails"),
        )

        current_frame = torch.randn(3, 64, 64)
        goal_frame = torch.randn(3, 64, 64)

        result = planner.plan(current_frame, goal_frame)

        assert result.success is False
        assert result.low_confidence is True
        assert result.replan_count == 2

    def test_device_property(self):
        hl_planner = _make_high_level_planner()
        ll_planner = _make_low_level_planner()
        cfg = HierarchicalPlannerConfig.default()
        planner = HierarchicalPlanner(hl_planner, ll_planner, cfg, device="cpu")
        assert planner.device == torch.device("cpu")

    def test_config_property(self):
        hl_planner = _make_high_level_planner()
        ll_planner = _make_low_level_planner()
        cfg = HierarchicalPlannerConfig(subgoal_timeout=100)
        planner = HierarchicalPlanner(hl_planner, ll_planner, cfg, device="cpu")
        assert planner.config.subgoal_timeout == 100


class TestPlanToLatent:
    def test_skips_goal_encoding(self):
        rollout = _make_mock_rollout()
        encoder = _make_encoder()
        cfg = CEMConfig(population_size=8, n_iterations=2, horizon=4)
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")

        current_frame = torch.randn(3, 64, 64)
        goal_latent = torch.randn(8)

        actions = planner.plan_to_latent(current_frame, goal_latent)

        assert encoder.call_count == 1
        assert isinstance(actions, torch.Tensor)
        assert actions.dim() == 2

    def test_return_cost(self):
        rollout = _make_mock_rollout()
        encoder = _make_encoder()
        cfg = CEMConfig(population_size=8, n_iterations=2, horizon=4)
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")

        current_frame = torch.randn(3, 64, 64)
        goal_latent = torch.randn(8)

        result = planner.plan_to_latent(current_frame, goal_latent, return_cost=True)

        assert isinstance(result, tuple)
        actions, cost = result
        assert isinstance(actions, torch.Tensor)
        assert isinstance(cost, float)

    def test_2d_goal_latent(self):
        rollout = _make_mock_rollout()
        encoder = _make_encoder()
        cfg = CEMConfig(population_size=8, n_iterations=2, horizon=4)
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")

        current_frame = torch.randn(3, 64, 64)
        goal_latent = torch.randn(1, 8)

        actions = planner.plan_to_latent(current_frame, goal_latent)
        assert isinstance(actions, torch.Tensor)
        assert actions.dim() == 2

    def test_bounded_actions(self):
        rollout = _make_mock_rollout()
        encoder = _make_encoder()
        cfg = CEMConfig(
            population_size=16, n_iterations=2, horizon=4,
            action_low=-0.5, action_high=0.5,
        )
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")

        current_frame = torch.randn(3, 64, 64)
        goal_latent = torch.randn(8)

        actions = planner.plan_to_latent(current_frame, goal_latent)
        assert actions.min() >= -0.5
        assert actions.max() <= 0.5


class TestWarmStartMean:
    def test_set_warm_start_mean(self):
        rollout = _make_mock_rollout()
        encoder = _make_encoder()
        cfg = CEMConfig(population_size=8, n_iterations=2, horizon=4)
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")

        mean = torch.randn(4, 25)
        planner.set_warm_start_mean(mean)
        assert planner._warm_start_mean is mean

    def test_warm_start_passed_to_cem(self):
        rollout = _make_mock_rollout()
        encoder = _make_encoder()
        cfg = CEMConfig(population_size=8, n_iterations=2, horizon=4)
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")

        mean = torch.zeros(4, 25)
        planner.set_warm_start_mean(mean)

        mock_cem = MagicMock()
        mock_cem.optimize.return_value = (torch.randn(4, 25), [1.0])
        planner._cem = mock_cem

        current_frame = torch.randn(3, 64, 64)
        goal_latent = torch.randn(8)

        planner.plan_to_latent(current_frame, goal_latent)

        mock_cem.optimize.assert_called_once()
        call_kwargs = mock_cem.optimize.call_args.kwargs
        assert call_kwargs.get("init_mean") is mean

    def test_no_warm_start_by_default(self):
        rollout = _make_mock_rollout()
        encoder = _make_encoder()
        cfg = CEMConfig(population_size=8, n_iterations=2, horizon=4)
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")

        mock_cem = MagicMock()
        mock_cem.optimize.return_value = (torch.randn(4, 25), [1.0])
        planner._cem = mock_cem

        current_frame = torch.randn(3, 64, 64)
        goal_latent = torch.randn(8)

        planner.plan_to_latent(current_frame, goal_latent)

        mock_cem.optimize.assert_called_once()
        call_kwargs = mock_cem.optimize.call_args.kwargs
        assert call_kwargs.get("init_mean") is None
