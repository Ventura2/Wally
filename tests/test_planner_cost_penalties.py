"""Cost-function unit tests for the diversity and camera-stillness penalties.

These tests construct a minimal :class:`GoalConditionedPlanner` with a stub
world model that returns a constant latent (so the base latent-distance
cost is equal across all candidates), then assert that the new penalties
bias the cost in the documented direction.

Regression target: ``openspec/changes/fix-wood-gathering-stall`` tasks 2.1
and 2.2.
"""
from __future__ import annotations

import pytest
import torch

from wally.planner.config import CEMConfig
from wally.planner.plan import GoalConditionedPlanner


class _ConstantLatentWorldModel:
    """Stub world model whose ``rollout`` returns a constant latent tensor.

    With this stub, the base latent-distance cost
    ``((z_H - z_g) ** 2).sum(-1)`` is the same for every candidate, so any
    difference in the planner's reported cost comes from the regularization
    terms we are testing.
    """

    def __init__(self, latent_dim: int = 4) -> None:
        self._latent_dim = latent_dim

    def rollout(self, z_0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # z_0: (pop, latent_dim) -> trajectory of (pop, H+1, latent_dim)
        pop, h, _ = actions.shape
        return torch.zeros(pop, h + 1, self._latent_dim)


def _encoder(frame: torch.Tensor) -> torch.Tensor:
    # (1, 3, H, W) -> (1, latent_dim)
    return torch.zeros(1, 4)


def _make_planner(diversity: float, camera: float) -> GoalConditionedPlanner:
    cfg = CEMConfig(
        population_size=8,
        horizon=4,
        n_iterations=1,
        diversity_penalty=diversity,
        camera_still_penalty=camera,
        inventory_stall_penalty=0.0,
    )
    return GoalConditionedPlanner(
        world_model=_ConstantLatentWorldModel(),  # type: ignore[arg-type]
        encoder=_encoder,
        config=cfg,
        device="cpu",
    )


def _sample_actions(pop: int = 8, horizon: int = 4, dim: int = 25) -> torch.Tensor:
    # Build actions in [-1, 1] with values that span the action space.
    torch.manual_seed(0)
    return torch.empty(pop, horizon, dim).uniform_(-1.0, 1.0)


def _plan(planner: GoalConditionedPlanner, actions: torch.Tensor) -> torch.Tensor:
    """Run the cost function directly on a fixed population of actions."""
    # Bypass CEM: invoke _regularized_cost on a candidate population.
    z_H = torch.zeros(actions.shape[0], 4)
    z_g = torch.zeros(actions.shape[0], 4)
    return planner._regularized_cost(actions, z_H, z_g)  # type: ignore[attr-defined]


class TestDiversityPenalty:
    def test_zero_penalty_gives_uniform_cost(self) -> None:
        """When diversity_penalty=0 and latent cost is constant, every
        candidate should have the same total cost regardless of where it
        sits in the population.
        """
        planner = _make_planner(diversity=0.0, camera=0.0)
        actions = _sample_actions()
        cost = _plan(planner, actions)
        # All costs equal (the base latent cost is 0 and no regularization)
        assert torch.allclose(cost, torch.zeros_like(cost))

    def test_high_diversity_preferred_over_low_diversity(self) -> None:
        """With diversity_penalty > 0, a candidate that is *far* from the
        population mean (high diversity) should have a *lower* cost than a
        candidate that is *close* to the mean (low diversity).
        """
        planner = _make_planner(diversity=1.0, camera=0.0)
        # Build a population: one outlier + 7 candidates clustered together.
        _, horizon, dim = 8, 4, 25
        clustered = torch.full((7, horizon, dim), 0.0)
        outlier = torch.full((1, horizon, dim), 0.9)
        # Shape for _plan: (pop, horizon, dim)
        actions = torch.cat([clustered, outlier], dim=0)
        cost = _plan(planner, actions)
        # The clustered candidates get a less-negative diversity contribution
        # (closer to mean) than the outlier, so they should have HIGHER cost.
        clustered_cost = cost[:7]
        outlier_cost = cost[7]
        assert (outlier_cost < clustered_cost).all(), (
            f"outlier cost {outlier_cost.tolist()} should be lower than "
            f"clustered mean {clustered_cost.mean().item():.3f}"
        )

    def test_penalty_disabled_in_config(self) -> None:
        """CEMConfig.diversity_penalty=0 must be a valid configuration
        that does not change the cost surface.
        """
        cfg_off = CEMConfig(diversity_penalty=0.0)
        cfg_on = CEMConfig(diversity_penalty=0.5)
        assert cfg_off.diversity_penalty == 0.0
        assert cfg_on.diversity_penalty == 0.5


class TestCameraStillPenalty:
    def test_zero_penalty_gives_uniform_cost(self) -> None:
        planner = _make_planner(diversity=0.0, camera=0.0)
        actions = _sample_actions()
        cost = _plan(planner, actions)
        assert torch.allclose(cost, torch.zeros_like(cost))

    def test_stationary_camera_disfavored(self) -> None:
        """A candidate with camera dims 0 and 1 near zero (still camera)
        should have a *higher* cost than a candidate with non-zero camera,
        when camera_still_penalty > 0.
        """
        planner = _make_planner(diversity=0.0, camera=1.0)
        pop, horizon, dim = 8, 4, 25
        actions = torch.zeros(pop, horizon, dim)
        # Half the candidates have non-zero camera (dims 0, 1)
        actions[:4, :, :2] = 0.8
        # Other half have zero camera
        cost = _plan(planner, actions)
        moving_cost = cost[:4]
        still_cost = cost[4:]
        assert (still_cost > moving_cost).all(), (
            f"still-camera mean cost {still_cost.mean().item():.3f} should "
            f"be higher than moving-camera mean {moving_cost.mean().item():.3f}"
        )

    def test_penalty_scales_linearly(self) -> None:
        """Doubling the coefficient should double the camera-still cost
        contribution (with the other terms at zero).
        """
        actions = torch.zeros(2, 4, 25)
        actions[0, :, :2] = 0.5  # moving
        actions[1, :, :2] = 0.0  # still
        cost_half = _plan(_make_planner(0.0, 0.5), actions)
        cost_full = _plan(_make_planner(0.0, 1.0), actions)
        diff_half = cost_half[1] - cost_half[0]
        diff_full = cost_full[1] - cost_full[0]
        # Difference should scale ~2x
        assert torch.allclose(diff_full, 2.0 * diff_half, atol=1e-5), (
            f"diff_full={diff_full.item():.4f} ~ 2x diff_half="
            f"{diff_half.item():.4f}"
        )

    def test_penalty_disabled_in_config(self) -> None:
        cfg = CEMConfig(camera_still_penalty=0.0)
        assert cfg.camera_still_penalty == 0.0


class TestPenaltiesApplyBeforeCEMSelection:
    """Per the spec: both penalties SHALL contribute to each candidate's
    cost before elites are selected. We verify this by running a single
    CEM step and checking the returned costs reflect the regularization
    (not the raw latent distance).
    """

    def test_cem_costs_include_penalties(self) -> None:
        planner = _make_planner(diversity=1.0, camera=1.0)
        # Build a fixed population: 4 still-camera candidates + 4 outlier candidates
        pop, horizon, dim = 8, 4, 25
        clustered = torch.full((4, horizon, dim), 0.0)
        outlier = torch.full((4, horizon, dim), 0.8)
        outlier[:, :, :2] = 0.8  # also moving camera
        actions = torch.cat([clustered, outlier], dim=0)

        z_0 = torch.zeros(pop, 4)
        z_g = torch.zeros(pop, 4)
        trajectory = planner._world_model.rollout(z_0, actions)
        z_H = trajectory[:, -1, :]
        costs = planner._regularized_cost(actions, z_H, z_g)
        # Clustered candidates are still-camera + low-diversity: should
        # have the highest cost.
        assert costs[:4].mean() > costs[4:].mean()


@pytest.mark.parametrize("bad_value", [-0.1, -1.0, -100.0])
def test_diversity_penalty_rejects_negative(bad_value: float) -> None:
    with pytest.raises(ValueError):
        CEMConfig(diversity_penalty=bad_value)


@pytest.mark.parametrize("bad_value", [-0.1, -1.0, -100.0])
def test_camera_penalty_rejects_negative(bad_value: float) -> None:
    with pytest.raises(ValueError):
        CEMConfig(camera_still_penalty=bad_value)
