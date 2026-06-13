from __future__ import annotations

import torch

from wally.training.ensemble import EnsembleConfig, EnsembleWorldModel

EMBED_DIM = 16
ACTION_DIM = 16
ENSEMBLE_SIZE = 3


def _make_small_ensemble() -> EnsembleWorldModel:
    cfg = EnsembleConfig(
        ensemble_size=ENSEMBLE_SIZE,
        embed_dim=EMBED_DIM,
        action_dim=ACTION_DIM,
    )
    return EnsembleWorldModel(cfg)


class TestEnsembleUncertaintySafePlanE2E:
    def test_train_and_rollout_with_uncertainty(self) -> None:
        torch.manual_seed(42)
        model = _make_small_ensemble()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        B = 32
        latents = torch.randn(B, EMBED_DIM)
        actions = torch.randn(B, ACTION_DIM)
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

        H = 5
        z_0 = torch.randn(2, EMBED_DIM)
        rollout_actions = torch.randn(2, H, ACTION_DIM)
        trajectory, cum_unc = model.rollout_with_uncertainty(z_0, rollout_actions)

        assert trajectory.shape == (2, H + 1, EMBED_DIM)
        assert cum_unc.shape == (2,)
        assert (cum_unc >= 0).all()

    def test_safe_plan_selection(self) -> None:
        torch.manual_seed(0)
        model = _make_small_ensemble()

        n_candidates = 5
        candidates = torch.randn(n_candidates, 8, 4)
        costs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        uncertainties = torch.tensor([0.5, 0.3, 2.0, 0.1, 0.8])

        best, low_conf = model.select_safe_plan(candidates, costs, uncertainties)
        assert low_conf is False
        assert torch.equal(best, candidates[0])

    def test_fallback_when_all_uncertain(self) -> None:
        torch.manual_seed(0)
        model = _make_small_ensemble()

        n_candidates = 3
        candidates = torch.randn(n_candidates, 8, 4)
        costs = torch.tensor([1.0, 2.0, 3.0])
        uncertainties = torch.tensor([5.0, 3.0, 4.0])

        best, low_conf = model.select_safe_plan(candidates, costs, uncertainties)
        assert low_conf is True
        assert torch.equal(best, candidates[1])

    def test_constraint_registration_and_filtering(self) -> None:
        model = _make_small_ensemble()

        model.register_constraint(
            "non_negative", lambda t: (t >= 0).all().item()
        )

        safe_traj = torch.ones(10, EMBED_DIM)
        assert model.check_constraints(safe_traj) is True

        unsafe_traj = -torch.ones(10, EMBED_DIM)
        assert model.check_constraints(unsafe_traj) is False

        trajectories = torch.stack([
            torch.ones(10, EMBED_DIM),
            -torch.ones(10, EMBED_DIM),
            torch.ones(10, EMBED_DIM) * 2,
        ])
        filtered = model.filter_by_constraints(trajectories)
        assert filtered.shape[0] == 2

    def test_full_pipeline_train_rollout_select(self) -> None:
        torch.manual_seed(99)
        model = _make_small_ensemble()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        B = 16
        latents = torch.randn(B, EMBED_DIM)
        actions = torch.randn(B, ACTION_DIM)
        targets = latents * 0.5 + actions * 0.3

        for _ in range(50):
            optimizer.zero_grad()
            a_emb = model._embed_actions(actions)
            total = torch.tensor(0.0)
            for member in model.members:
                pred = member(latents, a_emb)
                total = total + torch.nn.functional.mse_loss(pred, targets)
            total.backward()
            optimizer.step()

        z_0 = torch.randn(4, EMBED_DIM)
        H = 3
        candidate_actions = torch.randn(4, H, ACTION_DIM)
        trajectory, cum_unc = model.rollout_with_uncertainty(z_0, candidate_actions)

        assert trajectory.shape == (4, H + 1, EMBED_DIM)
        assert cum_unc.shape == (4,)

        costs = torch.norm(trajectory[:, -1, :], p=2, dim=-1)
        uncertainties = cum_unc

        best_plan, low_conf = model.select_safe_plan(
            candidate_actions, costs, uncertainties
        )
        assert best_plan.shape == (H, ACTION_DIM)
        assert isinstance(low_conf, bool)

        model.register_constraint(
            "bounded_trajectory",
            lambda t: (t.abs() < 100).all().item(),
        )
        filtered = model.filter_by_constraints(trajectory)
        assert filtered.shape[0] <= trajectory.shape[0]
