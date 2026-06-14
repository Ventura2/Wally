from __future__ import annotations

import pytest
import torch

from wally.training.losses import combined_loss, prediction_loss
from wally.training.sigreg import SIGReg


class TestPredictionLoss:
    def test_zero_when_predicted_equals_true_change(self):
        emb = torch.randn(4, 16, 192)
        true_change = emb[:, 1:] - emb[:, :-1]
        loss = prediction_loss(emb, true_change)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_is_scalar(self):
        emb = torch.randn(2, 11, 64)
        predicted_change = torch.randn(2, 10, 64)
        loss = prediction_loss(emb, predicted_change)
        assert loss.ndim == 0

    def test_loss_is_differentiable(self):
        emb = torch.randn(2, 11, 64, requires_grad=True)
        predicted_change = torch.randn(2, 10, 64)
        loss = prediction_loss(emb, predicted_change)
        loss.backward()
        assert emb.grad is not None

    def test_loss_increases_with_distance(self):
        emb = torch.randn(2, 11, 64)
        true_change = emb[:, 1:] - emb[:, :-1]
        close = true_change + 0.1
        far = true_change + 1.0
        loss_close = prediction_loss(emb, close)
        loss_far = prediction_loss(emb, far)
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

    def test_rejects_non_3d_input(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        with pytest.raises(AssertionError, match=r"SIGReg expects \(T, B, D\)"):
            sigreg(torch.randn(4, 32))


class TestCombinedLoss:
    def test_returns_total_and_metrics(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        emb = torch.randn(4, 16, 64)
        predicted_change = torch.randn(4, 15, 64)
        embeddings_tbd = emb.transpose(0, 1)
        total, metrics = combined_loss(
            emb, predicted_change, embeddings_tbd, 0.01, sigreg
        )
        assert total.ndim == 0
        assert "prediction_loss" in metrics
        assert "sigreg_loss" in metrics
        assert "total_loss" in metrics

    def test_total_equals_pred_plus_alpha_sigreg(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        emb = torch.randn(4, 16, 64)
        predicted_change = torch.randn(4, 15, 64)
        embeddings_tbd = emb.transpose(0, 1)
        alpha = 0.5
        total, metrics = combined_loss(
            emb, predicted_change, embeddings_tbd, alpha, sigreg
        )
        expected = metrics["prediction_loss"] + alpha * metrics["sigreg_loss"]
        assert metrics["total_loss"] == pytest.approx(expected, abs=1e-4)

    def test_alpha_zero_ignores_sigreg(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        emb = torch.randn(4, 16, 64)
        predicted_change = torch.randn(4, 15, 64)
        embeddings_tbd = emb.transpose(0, 1)
        total, metrics = combined_loss(
            emb, predicted_change, embeddings_tbd, 0.0, sigreg
        )
        assert total.item() == pytest.approx(metrics["prediction_loss"], abs=1e-6)

    def test_total_is_differentiable(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        emb = torch.randn(4, 16, 64, requires_grad=True)
        predicted_change = torch.randn(4, 15, 64)
        embeddings_tbd = emb.transpose(0, 1)
        total, _ = combined_loss(
            emb, predicted_change, embeddings_tbd, 0.01, sigreg
        )
        total.backward()
        assert emb.grad is not None

    def test_pred_loss_is_residual(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        emb = torch.randn(4, 16, 64)
        predicted_change = torch.randn(4, 15, 64)
        embeddings_tbd = emb.transpose(0, 1)
        _, metrics = combined_loss(
            emb, predicted_change, embeddings_tbd, 0.0, sigreg
        )
        expected = (emb[:, 1:] - emb[:, :-1] - predicted_change).pow(2).mean()
        assert metrics["prediction_loss"] == pytest.approx(expected.item(), abs=1e-5)

    def test_sigreg_input_is_tbd(self):
        """SIGReg receives the (T, B, D) tensor directly, not a re-transposed one."""
        sigreg = SIGReg(knots=17, num_proj=64)
        emb = torch.randn(4, 16, 64)
        predicted_change = torch.randn(4, 15, 64)
        embeddings_tbd = emb.transpose(0, 1)
        recorded: dict[str, tuple[int, ...]] = {}
        original_forward = sigreg.forward

        def recording_forward(proj: torch.Tensor) -> torch.Tensor:
            recorded["shape"] = tuple(proj.shape)
            return original_forward(proj)

        sigreg.forward = recording_forward  # type: ignore[method-assign]
        combined_loss(emb, predicted_change, embeddings_tbd, 0.0, sigreg)
        assert recorded["shape"] == (16, 4, 64)
