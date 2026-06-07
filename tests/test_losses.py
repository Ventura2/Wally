from __future__ import annotations

import torch
import pytest

from wally.training.losses import prediction_loss, combined_loss
from wally.training.sigreg import SIGRegCritic, sigreg_loss


class TestPredictionLoss:
    def test_identical_inputs_zero_loss(self):
        x = torch.randn(4, 15, 192)
        loss = prediction_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_is_scalar(self):
        pred = torch.randn(2, 10, 64)
        target = torch.randn(2, 10, 64)
        loss = prediction_loss(pred, target)
        assert loss.ndim == 0

    def test_loss_is_differentiable(self):
        pred = torch.randn(2, 10, 64, requires_grad=True)
        target = torch.randn(2, 10, 64)
        loss = prediction_loss(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_loss_increases_with_distance(self):
        target = torch.zeros(2, 10, 64)
        close = torch.ones(2, 10, 64) * 0.1
        far = torch.ones(2, 10, 64) * 1.0
        loss_close = prediction_loss(close, target)
        loss_far = prediction_loss(far, target)
        assert loss_far.item() > loss_close.item()


class TestSIGRegCritic:
    def test_output_shape(self):
        critic = SIGRegCritic(embed_dim=192, hidden_dim=256)
        pred = torch.randn(4, 15, 192)
        target = torch.randn(4, 15, 192)
        out = critic(pred, target)
        assert out.shape[-1] == 1

    def test_different_embed_dims(self):
        critic = SIGRegCritic(embed_dim=64, hidden_dim=128)
        pred = torch.randn(2, 8, 64)
        target = torch.randn(2, 8, 64)
        out = critic(pred, target)
        assert out.shape == (2, 8, 1)


class TestSigregLoss:
    def test_loss_is_scalar(self):
        critic = SIGRegCritic(embed_dim=64, hidden_dim=128)
        pred = torch.randn(4, 15, 64)
        target = torch.randn(4, 15, 64)
        loss = sigreg_loss(critic, pred, target)
        assert loss.ndim == 0

    def test_loss_is_differentiable(self):
        critic = SIGRegCritic(embed_dim=64, hidden_dim=128)
        pred = torch.randn(4, 15, 64, requires_grad=True)
        target = torch.randn(4, 15, 64)
        loss = sigreg_loss(critic, pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_loss_finite(self):
        critic = SIGRegCritic(embed_dim=64, hidden_dim=128)
        pred = torch.randn(4, 15, 64)
        target = torch.randn(4, 15, 64)
        loss = sigreg_loss(critic, pred, target)
        assert torch.isfinite(loss)


class TestCombinedLoss:
    def test_returns_total_and_metrics(self):
        critic = SIGRegCritic(embed_dim=64, hidden_dim=128)
        pred = torch.randn(4, 15, 64)
        target = torch.randn(4, 15, 64)
        total, metrics = combined_loss(pred, target, critic, alpha=0.1)
        assert total.ndim == 0
        assert "prediction_loss" in metrics
        assert "sigreg_loss" in metrics
        assert "total_loss" in metrics

    def test_total_equals_pred_plus_alpha_sigreg(self):
        critic = SIGRegCritic(embed_dim=64, hidden_dim=128)
        pred = torch.randn(4, 15, 64)
        target = torch.randn(4, 15, 64)
        alpha = 0.5
        total, metrics = combined_loss(pred, target, critic, alpha=alpha)
        expected = metrics["prediction_loss"] + alpha * metrics["sigreg_loss"]
        assert metrics["total_loss"] == pytest.approx(expected, abs=1e-4)

    def test_alpha_zero_ignores_sigreg(self):
        critic = SIGRegCritic(embed_dim=64, hidden_dim=128)
        pred = torch.randn(4, 15, 64)
        target = torch.randn(4, 15, 64)
        total, metrics = combined_loss(pred, target, critic, alpha=0.0)
        assert total.item() == pytest.approx(metrics["prediction_loss"], abs=1e-6)

    def test_total_is_differentiable(self):
        critic = SIGRegCritic(embed_dim=64, hidden_dim=128)
        pred = torch.randn(4, 15, 64, requires_grad=True)
        target = torch.randn(4, 15, 64)
        total, _ = combined_loss(pred, target, critic, alpha=0.1)
        total.backward()
        assert pred.grad is not None
