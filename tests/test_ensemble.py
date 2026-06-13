from __future__ import annotations

import pytest
import torch

from wally.training.ensemble import EnsembleConfig, EnsembleWorldModel


class TestEnsembleConfig:
    def test_default_config(self) -> None:
        cfg = EnsembleConfig.default()
        assert cfg.ensemble_size == 3
        assert cfg.embed_dim == 192
        assert cfg.action_dim == 25
        assert cfg.depth == 4
        assert cfg.num_heads == 4
        assert cfg.mlp_ratio == 4.0
        assert cfg.dropout == 0.1
        assert cfg.uncertainty_threshold == 1.0

    def test_ensemble_size_below_3_raises(self) -> None:
        with pytest.raises(Exception):
            EnsembleConfig(ensemble_size=2)

    def test_ensemble_size_above_5_raises(self) -> None:
        with pytest.raises(Exception):
            EnsembleConfig(ensemble_size=6)

    def test_negative_uncertainty_threshold_raises(self) -> None:
        with pytest.raises(Exception):
            EnsembleConfig(uncertainty_threshold=-1.0)

    def test_zero_uncertainty_threshold_raises(self) -> None:
        with pytest.raises(Exception):
            EnsembleConfig(uncertainty_threshold=0.0)

    def test_valid_custom_config(self) -> None:
        cfg = EnsembleConfig(ensemble_size=5, uncertainty_threshold=2.0)
        assert cfg.ensemble_size == 5
        assert cfg.uncertainty_threshold == 2.0


def _make_ensemble(
    ensemble_size: int = 3, embed_dim: int = 16, action_dim: int | None = None
) -> EnsembleWorldModel:
    if action_dim is None:
        action_dim = embed_dim
    cfg = EnsembleConfig(
        ensemble_size=ensemble_size, embed_dim=embed_dim, action_dim=action_dim
    )
    return EnsembleWorldModel(cfg)


class TestEnsembleTraining:
    def test_train_step_returns_per_member_and_average(self) -> None:
        torch.manual_seed(0)
        model = _make_ensemble(embed_dim=16)
        B = 4
        latents = torch.randn(B, 16)
        actions = torch.randn(B, 16)
        targets = torch.randn(B, 16)
        losses = model.train_step(latents, actions, targets)
        for i in range(3):
            assert f"member_{i}" in losses
        assert "average" in losses

    def test_training_reduces_loss(self) -> None:
        torch.manual_seed(42)
        model = _make_ensemble(embed_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        B = 32
        latents = torch.randn(B, 16)
        actions = torch.randn(B, 16)
        targets = latents * 0.5 + actions * 0.3

        initial_loss = model.train_step(latents, actions, targets)["average"]

        for _ in range(100):
            optimizer.zero_grad()
            a_emb = model._embed_actions(actions)
            total = torch.tensor(0.0)
            for member in model.members:
                pred = member(latents, a_emb)
                total = total + torch.nn.functional.mse_loss(pred, targets)
            total.backward()
            optimizer.step()

        final_loss = model.train_step(latents, actions, targets)["average"]
        assert final_loss < initial_loss


class TestPredictWithUncertainty:
    def test_returns_mean_and_variance(self) -> None:
        torch.manual_seed(0)
        model = _make_ensemble(embed_dim=16)
        z = torch.randn(4, 16)
        action = torch.randn(4, 16)
        mean, var = model.predict_with_uncertainty(z, action)
        assert mean.shape == (4, 16)
        assert var.shape == (4,)

    def test_variance_zero_when_members_agree(self) -> None:
        torch.manual_seed(0)
        model = _make_ensemble(embed_dim=16)
        state_dict = model.members[0].state_dict()
        for member in model.members:
            member.load_state_dict(state_dict)
        z = torch.randn(4, 16)
        action = torch.randn(4, 16)
        _, var = model.predict_with_uncertainty(z, action)
        assert torch.allclose(var, torch.zeros_like(var), atol=1e-6)


class TestRolloutWithUncertainty:
    def test_returns_trajectory_and_cumulative_uncertainty(self) -> None:
        torch.manual_seed(0)
        model = _make_ensemble(embed_dim=16)
        H = 5
        z_0 = torch.randn(2, 16)
        actions = torch.randn(2, H, 16)
        trajectory, cum_unc = model.rollout_with_uncertainty(z_0, actions)
        assert trajectory.shape == (2, H + 1, 16)
        assert cum_unc.shape == (2,)
        assert (cum_unc >= 0).all()


class TestSafePlanSelection:
    def test_picks_lowest_cost_among_safe(self) -> None:
        model = _make_ensemble()
        candidates = torch.randn(5, 8, 4)
        costs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        uncertainties = torch.tensor([0.5, 0.3, 2.0, 0.1, 0.8])
        best, low_conf = model.select_safe_plan(candidates, costs, uncertainties)
        assert not low_conf
        assert torch.equal(best, candidates[0])

    def test_fallback_when_all_uncertain(self) -> None:
        model = _make_ensemble()
        candidates = torch.randn(3, 8, 4)
        costs = torch.tensor([1.0, 2.0, 3.0])
        uncertainties = torch.tensor([5.0, 3.0, 4.0])
        best, low_conf = model.select_safe_plan(candidates, costs, uncertainties)
        assert low_conf
        assert torch.equal(best, candidates[1])


class TestConstraints:
    def test_register_and_check_constraints(self) -> None:
        model = _make_ensemble()
        model.register_constraint("non_negative", lambda t: (t >= 0).all().item())
        safe_traj = torch.ones(10, 16)
        assert model.check_constraints(safe_traj) is True
        unsafe_traj = -torch.ones(10, 16)
        assert model.check_constraints(unsafe_traj) is False

    def test_filter_by_constraints(self) -> None:
        model = _make_ensemble()
        model.register_constraint("non_negative", lambda t: (t >= 0).all().item())
        trajectories = torch.stack([
            torch.ones(10, 16),
            -torch.ones(10, 16),
            torch.ones(10, 16) * 2,
        ])
        filtered = model.filter_by_constraints(trajectories)
        assert filtered.shape[0] == 2

    def test_no_constraints_passes_all(self) -> None:
        model = _make_ensemble()
        trajectories = torch.randn(5, 10, 16)
        filtered = model.filter_by_constraints(trajectories)
        assert filtered.shape[0] == 5
