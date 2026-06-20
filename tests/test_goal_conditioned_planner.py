from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from wally.planner.config import CEMConfig
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


def _make_constant_rollout(latent_dim: int = 8) -> MagicMock:
    mock = MagicMock()

    def rollout(z_0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, H, _ = actions.shape
        trajectory = torch.zeros(B, H + 1, latent_dim, device=actions.device)
        trajectory[:, 0, :] = z_0
        return trajectory

    mock.rollout = MagicMock(side_effect=rollout)
    return mock


class TestGoalConditionedPlanner:
    def test_default_inventory_stall_penalty_is_nonzero(self):
        cfg = CEMConfig.default()
        assert cfg.inventory_stall_penalty > 0

    def test_bounded_actions(self):
        cfg = CEMConfig(
            population_size=32, n_iterations=3, horizon=4,
            action_low=-0.5, action_high=0.5,
        )
        rollout = _make_mock_rollout()
        encoder = _make_encoder()
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")

        frame = torch.randn(3, 64, 64)
        actions = planner.plan(frame, frame)
        assert actions.min() >= -0.5
        assert actions.max() <= 0.5

    def test_encoder_reuse(self):
        cfg = CEMConfig(population_size=16, n_iterations=2, horizon=3)
        rollout = _make_mock_rollout()
        encoder = _make_encoder()
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")

        current = torch.randn(3, 64, 64)
        goal = torch.randn(3, 64, 64)
        planner.plan(current, goal)

        assert encoder.call_count == 2
        assert planner.encoder is encoder

    def test_default_cost_function(self):
        cfg = CEMConfig(population_size=16, n_iterations=2, horizon=3)
        rollout = _make_mock_rollout()
        encoder = _make_encoder()
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")

        frame = torch.randn(3, 64, 64)
        actions = planner.plan(frame, frame)
        assert actions.shape == (3, 25)

    def test_custom_cost_function(self):
        custom_called = []

        def custom_cost(z_H: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
            custom_called.append(True)
            return (z_H - z_g).abs().sum(dim=-1)

        cfg = CEMConfig(population_size=16, n_iterations=2, horizon=3)
        rollout = _make_mock_rollout()
        encoder = _make_encoder()
        planner = GoalConditionedPlanner(
            rollout, encoder, cfg, device="cpu", cost_fn=custom_cost
        )

        frame = torch.randn(3, 64, 64)
        planner.plan(frame, frame)
        assert len(custom_called) > 0

    def test_device_auto_select(self):
        cfg = CEMConfig()
        rollout = _make_mock_rollout()
        encoder = _make_encoder()
        planner = GoalConditionedPlanner(rollout, encoder, cfg)

        expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert planner.device == expected

    def test_device_explicit_override(self):
        cfg = CEMConfig()
        rollout = _make_mock_rollout()
        encoder = _make_encoder()
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")
        assert planner.device == torch.device("cpu")

    def test_return_cost_true(self):
        cfg = CEMConfig(population_size=16, n_iterations=2, horizon=3)
        rollout = _make_mock_rollout()
        encoder = _make_encoder()
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")

        frame = torch.randn(3, 64, 64)
        result = planner.plan(frame, frame, return_cost=True)
        assert isinstance(result, tuple)
        actions, cost = result
        assert actions.shape == (3, 25)
        assert isinstance(cost, float)

    def test_return_cost_false(self):
        cfg = CEMConfig(population_size=16, n_iterations=2, horizon=3)
        rollout = _make_mock_rollout()
        encoder = _make_encoder()
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")

        frame = torch.randn(3, 64, 64)
        result = planner.plan(frame, frame)
        assert isinstance(result, torch.Tensor)

    def test_4d_input(self):
        cfg = CEMConfig(population_size=16, n_iterations=2, horizon=3)
        rollout = _make_mock_rollout()
        encoder = _make_encoder()
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")

        frame = torch.randn(2, 3, 64, 64)
        actions = planner.plan(frame, frame)
        assert actions.shape == (3, 25)

    def test_inventory_stall_penalty_prefers_clean_sequence(self):
        cfg = CEMConfig(
            population_size=4,
            n_iterations=1,
            horizon=3,
            inventory_stall_penalty=0.25,
        )
        rollout = _make_constant_rollout()
        encoder = _make_encoder()
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")

        def fake_optimize(cost_fn, **kwargs):
            actions = torch.zeros(2, cfg.horizon, 25)
            actions[0, :, 12] = 1.0
            costs = cost_fn(actions)
            assert costs[0] > costs[1]
            return actions[costs.argmin().item()].clone(), [costs.min().item()]

        planner._cem.optimize = MagicMock(side_effect=fake_optimize)

        frame = torch.randn(3, 64, 64)
        actions = planner.plan(frame, frame)
        assert torch.count_nonzero(actions[:, 12]) == 0

    def test_inventory_stall_penalty_can_be_disabled(self):
        cfg = CEMConfig(
            population_size=4,
            n_iterations=1,
            horizon=3,
            inventory_stall_penalty=0.0,
        )
        rollout = _make_constant_rollout()
        encoder = _make_encoder()
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")

        def fake_optimize(cost_fn, **kwargs):
            actions = torch.zeros(2, cfg.horizon, 25)
            actions[0, :, 12] = 1.0
            costs = cost_fn(actions)
            assert torch.allclose(costs[0], costs[1])
            return actions[0].clone(), [costs[0].item()]

        planner._cem.optimize = MagicMock(side_effect=fake_optimize)

        frame = torch.randn(3, 64, 64)
        actions = planner.plan(frame, frame)
        assert actions.shape == (3, 25)


class TestGoalConditionedPlannerCPU:
    def test_plan_to_latent_end_to_end_on_cpu(self):
        latent_dim = 8
        cfg = CEMConfig(population_size=16, n_iterations=2, horizon=3)
        rollout = _make_mock_rollout(latent_dim=latent_dim)
        encoder = _make_encoder(latent_dim=latent_dim)
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cpu")

        current_frame = torch.randn(3, 64, 64)
        goal_latent = torch.randn(latent_dim)

        actions = planner.plan_to_latent(current_frame, goal_latent)

        assert isinstance(actions, torch.Tensor)
        assert torch.isfinite(actions).all()


class TestGoalConditionedPlannerCUDA:
    pytestmark = pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )

    def test_plan_to_latent_returns_cuda_tensor(self):
        latent_dim = 8

        def cuda_encoder(frame: torch.Tensor) -> torch.Tensor:
            frame = frame.to("cuda")
            B = frame.shape[0]
            return torch.randn(B, latent_dim, device="cuda")

        def cuda_rollout(z_0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            z_0 = z_0.to(actions.device)
            B, H, _ = actions.shape
            trajectory = torch.randn(B, H + 1, latent_dim, device=actions.device)
            trajectory[:, 0, :] = z_0
            return trajectory

        encoder = MagicMock(side_effect=cuda_encoder)
        rollout = MagicMock()
        rollout.rollout = MagicMock(side_effect=cuda_rollout)

        cfg = CEMConfig(population_size=16, n_iterations=2, horizon=3)
        planner = GoalConditionedPlanner(rollout, encoder, cfg, device="cuda")

        current_frame = torch.randn(3, 64, 64)
        goal_latent = torch.randn(latent_dim, device="cuda")

        actions = planner.plan_to_latent(current_frame, goal_latent)

        assert actions.device.type == "cuda"
