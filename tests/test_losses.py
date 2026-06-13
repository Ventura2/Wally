from __future__ import annotations

import pytest
import torch

from wally.training.losses import combined_loss, prediction_loss
from wally.training.sigreg import SIGReg


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


class TestSIGReg:
    def test_output_is_scalar(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        proj = torch.randn(8, 4, 32)
        out = sigreg(proj)
        assert out.ndim == 0

    def test_loss_finite(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        proj = torch.randn(8, 4, 32)
        out = sigreg(proj)
        assert torch.isfinite(out)

    def test_loss_non_negative(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        proj = torch.randn(8, 4, 32)
        out = sigreg(proj)
        assert out.item() >= 0.0

    def test_constant_embedding_finite(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        proj = torch.zeros(8, 4, 32)
        out = sigreg(proj)
        assert torch.isfinite(out)
        assert out.item() >= 0.0

    def test_gradient_flows_to_embedding(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        proj = torch.randn(8, 4, 32, requires_grad=True)
        out = sigreg(proj)
        out.backward()
        assert proj.grad is not None


class TestCombinedLoss:
    def test_returns_total_and_metrics(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        pred = torch.randn(4, 15, 64)
        target = torch.randn(4, 15, 64)
        embeddings = torch.randn(4, 16, 64)
        total, metrics = combined_loss(pred, target, embeddings, 0.01, sigreg)
        assert total.ndim == 0
        assert "prediction_loss" in metrics
        assert "sigreg_loss" in metrics
        assert "total_loss" in metrics

    def test_total_equals_pred_plus_alpha_sigreg(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        pred = torch.randn(4, 15, 64)
        target = torch.randn(4, 15, 64)
        embeddings = torch.randn(4, 16, 64)
        alpha = 0.5
        total, metrics = combined_loss(pred, target, embeddings, alpha, sigreg)
        expected = metrics["prediction_loss"] + alpha * metrics["sigreg_loss"]
        assert metrics["total_loss"] == pytest.approx(expected, abs=1e-4)

    def test_alpha_zero_ignores_sigreg(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        pred = torch.randn(4, 15, 64)
        target = torch.randn(4, 15, 64)
        embeddings = torch.randn(4, 16, 64)
        total, metrics = combined_loss(pred, target, embeddings, 0.0, sigreg)
        assert total.item() == pytest.approx(metrics["prediction_loss"], abs=1e-6)

    def test_total_is_differentiable(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        pred = torch.randn(4, 15, 64, requires_grad=True)
        target = torch.randn(4, 15, 64)
        embeddings = torch.randn(4, 16, 64)
        total, _ = combined_loss(pred, target, embeddings, 0.01, sigreg)
        total.backward()
        assert pred.grad is not None

    def test_accepts_tbd_embeddings(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        pred = torch.randn(4, 15, 64)
        target = torch.randn(4, 15, 64)
        embeddings_tbd = torch.randn(16, 4, 64)
        total, _ = combined_loss(pred, target, embeddings_tbd, 0.01, sigreg)
        assert torch.isfinite(total)
