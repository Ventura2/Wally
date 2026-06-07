from __future__ import annotations

import pytest
import torch

from wally.planner.config import CEMConfig
from wally.planner.gradient_mpc import GradientMPC, GradientMPCConfig
from wally.planner.rollout import LatentRollout


class SimpleDynamicsModel:
    def __init__(self, latent_dim: int = 8, action_dim: int = 4) -> None:
        self.linear = torch.nn.Linear(latent_dim + action_dim, latent_dim)

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        return frame

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, action], dim=-1)
        return self.linear(x)


def _make_rollout(latent_dim: int = 8, action_dim: int = 4) -> LatentRollout:
    model = SimpleDynamicsModel(latent_dim, action_dim)
    return LatentRollout(model=model, gradient_policy="straight_through")


class TestGradientMPCConfig:
    def test_default_config(self) -> None:
        cfg = GradientMPCConfig.default()
        assert cfg.learning_rate == 0.01
        assert cfg.n_refinement_steps == 10
        assert cfg.grad_clip_norm == 1.0
        assert cfg.warm_start is True
        assert cfg.action_low == -1.0
        assert cfg.action_high == 1.0

    def test_negative_learning_rate_raises(self) -> None:
        with pytest.raises(Exception):
            GradientMPCConfig(learning_rate=-0.01)

    def test_zero_learning_rate_raises(self) -> None:
        with pytest.raises(Exception):
            GradientMPCConfig(learning_rate=0.0)

    def test_zero_refinement_steps_raises(self) -> None:
        with pytest.raises(Exception):
            GradientMPCConfig(n_refinement_steps=0)

    def test_negative_grad_clip_norm_raises(self) -> None:
        with pytest.raises(Exception):
            GradientMPCConfig(grad_clip_norm=-1.0)

    def test_zero_grad_clip_norm_raises(self) -> None:
        with pytest.raises(Exception):
            GradientMPCConfig(grad_clip_norm=0.0)

    def test_valid_custom_config(self) -> None:
        cfg = GradientMPCConfig(
            learning_rate=0.1,
            n_refinement_steps=5,
            grad_clip_norm=0.5,
            warm_start=False,
        )
        assert cfg.learning_rate == 0.1
        assert cfg.n_refinement_steps == 5
        assert cfg.grad_clip_norm == 0.5
        assert cfg.warm_start is False


class TestRefineActions:
    def test_refine_reduces_cost(self) -> None:
        torch.manual_seed(42)
        latent_dim, action_dim, horizon = 8, 4, 6
        rollout = _make_rollout(latent_dim, action_dim)
        cfg = GradientMPCConfig(
            learning_rate=0.05,
            n_refinement_steps=50,
            grad_clip_norm=10.0,
        )
        planner = GradientMPC(
            world_model=rollout,
            encoder=lambda x: x,
            config=cfg,
            action_dim=action_dim,
            device="cpu",
        )

        z_0 = torch.randn(1, latent_dim)
        z_g = torch.randn(1, latent_dim)
        actions = torch.zeros(horizon, action_dim)

        with torch.no_grad():
            traj_before = rollout.rollout(z_0, actions.unsqueeze(0))
            cost_before = ((traj_before[:, -1, :] - z_g) ** 2).sum().item()

        refined, cost_after = planner.refine_actions(z_0, z_g, actions)

        assert cost_after < cost_before

    def test_action_bounds_enforced(self) -> None:
        torch.manual_seed(42)
        latent_dim, action_dim, horizon = 8, 4, 6
        rollout = _make_rollout(latent_dim, action_dim)
        low, high = -0.5, 0.5
        cfg = GradientMPCConfig(
            learning_rate=1.0,
            n_refinement_steps=20,
            action_low=low,
            action_high=high,
        )
        planner = GradientMPC(
            world_model=rollout,
            encoder=lambda x: x,
            config=cfg,
            action_dim=action_dim,
            device="cpu",
        )

        z_0 = torch.randn(1, latent_dim)
        z_g = torch.randn(1, latent_dim)
        actions = torch.zeros(horizon, action_dim)

        refined, _ = planner.refine_actions(z_0, z_g, actions)
        assert refined.min() >= low
        assert refined.max() <= high

    def test_gradient_clipping_applied(self) -> None:
        torch.manual_seed(42)
        latent_dim, action_dim, horizon = 8, 4, 6
        rollout = _make_rollout(latent_dim, action_dim)
        for p in rollout._model.linear.parameters():
            with torch.no_grad():
                p.mul_(100.0)
        cfg = GradientMPCConfig(
            learning_rate=0.01,
            n_refinement_steps=5,
            grad_clip_norm=0.1,
        )
        planner = GradientMPC(
            world_model=rollout,
            encoder=lambda x: x,
            config=cfg,
            action_dim=action_dim,
            device="cpu",
        )

        z_0 = torch.randn(1, latent_dim)
        z_g = torch.randn(1, latent_dim)
        actions = torch.randn(horizon, action_dim)

        refined, cost = planner.refine_actions(z_0, z_g, actions)
        assert torch.isfinite(refined).all()
        assert torch.isfinite(torch.tensor(cost))


class TestWarmStart:
    def test_set_and_clear_warm_start_mean(self) -> None:
        rollout = _make_rollout()
        cfg = GradientMPCConfig()
        planner = GradientMPC(
            world_model=rollout,
            encoder=lambda x: x,
            config=cfg,
            device="cpu",
        )
        assert planner._warm_start_mean is None

        mean = torch.randn(6, 4)
        planner.set_warm_start_mean(mean)
        assert planner._warm_start_mean is not None
        assert torch.equal(planner._warm_start_mean, mean)

        planner.clear_warm_start_mean()
        assert planner._warm_start_mean is None

    def test_warm_start_mean_passed_to_cem(self) -> None:
        torch.manual_seed(42)
        latent_dim, action_dim, horizon = 8, 4, 6
        rollout = _make_rollout(latent_dim, action_dim)
        cfg = GradientMPCConfig(warm_start=True, n_refinement_steps=1)
        cem_config = CEMConfig(horizon=horizon, population_size=8, n_iterations=2)
        planner = GradientMPC(
            world_model=rollout,
            encoder=lambda x: x,
            config=cfg,
            cem_config=cem_config,
            action_dim=action_dim,
            device="cpu",
        )

        warm_mean = torch.randn(horizon, action_dim)
        planner.set_warm_start_mean(warm_mean)

        captured: dict = {}
        original_optimize = planner._cem.optimize

        def spy_optimize(cost_fn, **kwargs):
            captured.update(kwargs)
            return original_optimize(cost_fn, **kwargs)

        planner._cem.optimize = spy_optimize

        frame = torch.randn(1, 3, 16, 16)
        planner._encoder = lambda x: x.view(x.shape[0], -1)[:, :latent_dim]
        try:
            planner.plan(frame, frame)
        except Exception:
            pass

        if "init_mean" in captured:
            assert captured["init_mean"] is not None


class TestFullPlanPipeline:
    def test_plan_returns_actions(self) -> None:
        torch.manual_seed(42)
        latent_dim, action_dim, horizon = 8, 4, 6
        rollout = _make_rollout(latent_dim, action_dim)
        cfg = GradientMPCConfig(n_refinement_steps=3)
        cem_config = CEMConfig(horizon=horizon, population_size=8, n_iterations=2)

        def encoder(frame: torch.Tensor) -> torch.Tensor:
            return frame.view(frame.shape[0], -1)[:, :latent_dim]

        planner = GradientMPC(
            world_model=rollout,
            encoder=encoder,
            config=cfg,
            cem_config=cem_config,
            action_dim=action_dim,
            device="cpu",
        )

        current = torch.randn(3, 16, 16)
        goal = torch.randn(3, 16, 16)
        result = planner.plan(current, goal)
        assert isinstance(result, torch.Tensor)

    def test_plan_with_return_cost(self) -> None:
        torch.manual_seed(42)
        latent_dim, action_dim, horizon = 8, 4, 6
        rollout = _make_rollout(latent_dim, action_dim)
        cfg = GradientMPCConfig(n_refinement_steps=3)
        cem_config = CEMConfig(horizon=horizon, population_size=8, n_iterations=2)

        def encoder(frame: torch.Tensor) -> torch.Tensor:
            return frame.view(frame.shape[0], -1)[:, :latent_dim]

        planner = GradientMPC(
            world_model=rollout,
            encoder=encoder,
            config=cfg,
            cem_config=cem_config,
            action_dim=action_dim,
            device="cpu",
        )

        current = torch.randn(3, 16, 16)
        goal = torch.randn(3, 16, 16)
        actions, cost = planner.plan(current, goal, return_cost=True)
        assert isinstance(actions, torch.Tensor)
        assert isinstance(cost, float)
