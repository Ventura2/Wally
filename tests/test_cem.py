from __future__ import annotations

import torch

from wally.planner.cem import CEMOptimizer, RandomShooting


def _quadratic_cost(actions: torch.Tensor) -> torch.Tensor:
    target = torch.zeros_like(actions)
    target[..., 0] = 0.5
    return ((actions - target) ** 2).sum(dim=(-2, -1))


class TestCEMOptimizer:
    def test_cost_decreases(self):
        opt = CEMOptimizer()
        rng = torch.Generator().manual_seed(42)
        _, history = opt.optimize(
            _quadratic_cost,
            horizon=5,
            action_dim=3,
            population_size=128,
            elite_frac=0.1,
            n_iterations=10,
            rng=rng,
        )
        assert history[-1] < history[0]

    def test_bound_enforcement(self):
        opt = CEMOptimizer()
        rng = torch.Generator().manual_seed(0)
        actions, _ = opt.optimize(
            _quadratic_cost,
            horizon=4,
            action_dim=2,
            population_size=64,
            n_iterations=3,
            action_low=-0.5,
            action_high=0.5,
            rng=rng,
        )
        assert actions.min() >= -0.5
        assert actions.max() <= 0.5

    def test_determinism_with_seed(self):
        opt = CEMOptimizer()

        def run(seed: int) -> tuple[torch.Tensor, list[float]]:
            rng = torch.Generator().manual_seed(seed)
            return opt.optimize(
                _quadratic_cost,
                horizon=3,
                action_dim=2,
                population_size=32,
                n_iterations=3,
                rng=rng,
            )

        a1, h1 = run(123)
        a2, h2 = run(123)
        assert torch.equal(a1, a2)
        assert h1 == h2

    def test_small_population(self):
        opt = CEMOptimizer()
        rng = torch.Generator().manual_seed(7)
        actions, history = opt.optimize(
            _quadratic_cost,
            horizon=2,
            action_dim=1,
            population_size=2,
            elite_frac=0.5,
            n_iterations=3,
            rng=rng,
        )
        assert actions.shape == (2, 1)
        assert len(history) == 3

    def test_returns_correct_shape(self):
        opt = CEMOptimizer()
        rng = torch.Generator().manual_seed(0)
        actions, history = opt.optimize(
            _quadratic_cost,
            horizon=8,
            action_dim=4,
            population_size=64,
            n_iterations=5,
            rng=rng,
        )
        assert actions.shape == (8, 4)
        assert len(history) == 5


class TestRandomShooting:
    def test_returns_correct_shape(self):
        rs = RandomShooting()
        rng = torch.Generator().manual_seed(0)
        actions, history = rs.optimize(
            _quadratic_cost,
            horizon=5,
            action_dim=3,
            population_size=64,
            rng=rng,
        )
        assert actions.shape == (5, 3)
        assert len(history) == 1

    def test_bound_enforcement(self):
        rs = RandomShooting()
        rng = torch.Generator().manual_seed(0)
        actions, _ = rs.optimize(
            _quadratic_cost,
            horizon=4,
            action_dim=2,
            population_size=128,
            action_low=-0.3,
            action_high=0.3,
            rng=rng,
        )
        assert actions.min() >= -0.3
        assert actions.max() <= 0.3

    def test_determinism_with_seed(self):
        rs = RandomShooting()

        def run(seed: int) -> tuple[torch.Tensor, list[float]]:
            rng = torch.Generator().manual_seed(seed)
            return rs.optimize(
                _quadratic_cost,
                horizon=3,
                action_dim=2,
                population_size=32,
                rng=rng,
            )

        a1, h1 = run(99)
        a2, h2 = run(99)
        assert torch.equal(a1, a2)
        assert h1 == h2
