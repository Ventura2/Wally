from __future__ import annotations

import pytest
import torch

from wally.planner.cem import CEMOptimizer, RandomShooting
from wally.planner.config import CEMConfig
from wally.planner.plan import GoalConditionedPlanner
from wally.planner.rollout import LatentRollout

Z_DIM = 16
A_DIM = 4
HORIZON = 8
POP = 32
SEED = 42


class LinearDynamics:
    def __init__(self, z_dim: int = Z_DIM, a_dim: int = A_DIM, seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.z_dim = z_dim
        self.a_dim = a_dim
        self.W = torch.randn(z_dim, z_dim, generator=gen) * 0.1
        self.V = torch.randn(z_dim, a_dim, generator=gen) * 0.5
        self.bias = torch.zeros(z_dim)

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        b = frame.shape[0]
        return frame.view(b, -1)[:, : self.z_dim]

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return z @ self.W.T + action @ self.V.T + self.bias


def _make_encoder(z_dim: int = Z_DIM):
    def encoder(frame: torch.Tensor) -> torch.Tensor:
        b = frame.shape[0]
        return frame.view(b, -1)[:, :z_dim]

    return encoder


def _make_cost_fn(dynamics: LinearDynamics, z_goal: torch.Tensor):
    def cost_fn(actions: torch.Tensor) -> torch.Tensor:
        b = actions.shape[0]
        z = torch.zeros(b, dynamics.z_dim)
        for h in range(actions.shape[1]):
            z = dynamics.predict(z, actions[:, h, :])
        return ((z - z_goal) ** 2).sum(dim=-1)

    return cost_fn


@pytest.fixture
def dynamics() -> LinearDynamics:
    return LinearDynamics(seed=SEED)


@pytest.fixture
def z_goal(dynamics: LinearDynamics) -> torch.Tensor:
    gen = torch.Generator().manual_seed(SEED + 1)
    return torch.randn(1, dynamics.z_dim, generator=gen) * 2.0


class TestPlannerSmoke:
    @pytest.mark.smoke
    def test_planner_returns_bounded_actions(
        self, dynamics: LinearDynamics, z_goal: torch.Tensor
    ) -> None:
        cfg = CEMConfig(
            population_size=POP,
            elite_frac=0.2,
            n_iterations=3,
            horizon=HORIZON,
            action_low=-1.0,
            action_high=1.0,
        )
        rollout = LatentRollout(model=dynamics, gradient_policy="detach")
        encoder = _make_encoder()
        planner = GoalConditionedPlanner(
            rollout, encoder, cfg, action_dim=A_DIM, device="cpu",
        )

        current_frame = torch.zeros(1, 3, 8, 8)
        goal_frame = torch.zeros(1, 3, 8, 8)
        goal_frame.view(1, -1)[0, :Z_DIM] = z_goal.squeeze(0)

        actions = planner.plan(current_frame, goal_frame)

        assert actions.shape == (HORIZON, A_DIM)
        assert actions.min() >= cfg.action_low
        assert actions.max() <= cfg.action_high

    @pytest.mark.smoke
    def test_cem_decreases_cost(
        self, dynamics: LinearDynamics, z_goal: torch.Tensor
    ) -> None:
        cost_fn = _make_cost_fn(dynamics, z_goal)
        cem = CEMOptimizer()
        rng = torch.Generator().manual_seed(SEED)

        _, cost_history = cem.optimize(
            cost_fn,
            horizon=HORIZON,
            action_dim=A_DIM,
            population_size=POP,
            elite_frac=0.2,
            n_iterations=5,
            action_low=-1.0,
            action_high=1.0,
            rng=rng,
        )

        assert len(cost_history) == 5
        assert cost_history[-1] < cost_history[0]

    @pytest.mark.smoke
    def test_cem_beats_random_shooting(
        self, dynamics: LinearDynamics, z_goal: torch.Tensor
    ) -> None:
        cost_fn = _make_cost_fn(dynamics, z_goal)

        rng_cem = torch.Generator().manual_seed(SEED)
        cem = CEMOptimizer()
        _, cem_costs = cem.optimize(
            cost_fn,
            horizon=HORIZON,
            action_dim=A_DIM,
            population_size=POP,
            elite_frac=0.2,
            n_iterations=5,
            action_low=-1.0,
            action_high=1.0,
            rng=rng_cem,
        )

        rng_rs = torch.Generator().manual_seed(SEED)
        rs = RandomShooting()
        _, rs_costs = rs.optimize(
            cost_fn,
            horizon=HORIZON,
            action_dim=A_DIM,
            population_size=POP,
            action_low=-1.0,
            action_high=1.0,
            rng=rng_rs,
        )

        assert cem_costs[-1] < rs_costs[0]
