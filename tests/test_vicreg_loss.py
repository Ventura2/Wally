from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from wally.training.losses import combined_loss, vicreg_loss
from wally.training.sigreg import SIGReg


class TestVicregLoss:
    @pytest.mark.smoke
    def test_vicreg_loss_returns_finite_value_for_random_input(self):
        z = torch.randn(16, 4)
        out = vicreg_loss(z)
        assert out.ndim == 0
        assert torch.isfinite(out)

    @pytest.mark.smoke
    def test_std_term_is_hinge_on_std(self):
        z = torch.ones(16, 4)
        out = vicreg_loss(z, std_target=1.0, cov_weight=0.0)
        assert out.item() == pytest.approx(1.0, abs=1e-6)

    def test_std_term_zero_when_std_above_target(self):
        torch.manual_seed(0)
        z = torch.randn(16, 4) * 2.0
        out = vicreg_loss(z, std_target=1.0, cov_weight=0.0)
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.smoke
    def test_cov_term_zero_for_uncorrelated_input(self):
        # 16 samples × 4 dims has ~12 off-diag entries; with B=16 the sampling
        # noise in each entry is ~1/sqrt(B)=0.25, so the sum of squares is
        # ~12·0.25² / D ≈ 0.19. Use a larger batch to get the off-diag term
        # down to "sampling noise only" (< 0.1).
        torch.manual_seed(0)
        z = torch.randn(1024, 4)
        out_full = vicreg_loss(z, std_target=0.0, cov_weight=1.0)
        out_std = vicreg_loss(z, std_target=0.0, cov_weight=0.0)
        cov_only = (out_full - out_std).item()
        assert cov_only < 0.1

    @pytest.mark.smoke
    def test_cov_term_penalizes_perfectly_correlated_columns(self):
        torch.manual_seed(0)
        base = torch.randn(16, 1)
        z = base.expand(16, 4).contiguous()
        out_cov_zero = vicreg_loss(z, std_target=0.0, cov_weight=0.0)
        out_cov_one = vicreg_loss(z, std_target=0.0, cov_weight=1.0)
        cov_term = (out_cov_one - out_cov_zero).item()
        assert cov_term > 0.0

    def test_cov_weight_zero_disables_cov_term(self):
        torch.manual_seed(0)
        base = torch.randn(16, 1)
        z = base.expand(16, 4).contiguous()
        out = vicreg_loss(z, std_target=0.0, cov_weight=0.0)
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.smoke
    def test_gradients_flow_through_both_terms(self):
        z = torch.randn(16, 4, requires_grad=True)
        out = vicreg_loss(z, std_target=1.0, cov_weight=1.0)
        out.backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()
        assert (z.grad != 0).any()

    def test_batch_size_one_produces_nan(self):
        z = torch.randn(1, 4)
        out = vicreg_loss(z, std_target=1.0, cov_weight=1.0)
        assert torch.isnan(out).any() or not torch.isfinite(out)


class TestCombinedLossVicreg:
    def _make_args(self):
        sigreg = SIGReg(knots=17, num_proj=64)
        emb = torch.randn(4, 16, 64)
        predicted_change = torch.randn(4, 15, 64)
        embeddings_tbd = emb.transpose(0, 1)
        return emb, predicted_change, embeddings_tbd, sigreg

    @pytest.mark.smoke
    def test_combined_loss_vicreg_off_is_bit_identical(self):
        """With vicreg_weight=0 the metrics dict must NOT contain a vicreg_loss key,
        and the total must equal pred + alpha*sigreg (pre-VICReg behavior)."""
        emb, pc, emb_tbd, sigreg = self._make_args()
        total, metrics = combined_loss(
            emb, pc, emb_tbd, 0.1, sigreg, vicreg_weight=0.0
        )
        assert "vicreg_loss" not in metrics
        assert set(metrics.keys()) == {"prediction_loss", "sigreg_loss", "total_loss"}
        expected = (
            metrics["prediction_loss"] + 0.1 * metrics["sigreg_loss"]
        )
        assert metrics["total_loss"] == pytest.approx(expected, abs=1e-5)

    @pytest.mark.smoke
    def test_combined_loss_vicreg_on_includes_metric(self):
        emb, pc, emb_tbd, sigreg = self._make_args()
        total, metrics = combined_loss(
            emb, pc, emb_tbd, 0.1, sigreg, vicreg_weight=1.0
        )
        assert "vicreg_loss" in metrics
        assert metrics["vicreg_loss"] >= 0.0
        assert torch.isfinite(
            torch.tensor(metrics["vicreg_loss"])
        )
        # total includes the (unweighted) vicreg term, mirrored from sigreg pattern
        expected = (
            metrics["prediction_loss"]
            + 0.1 * metrics["sigreg_loss"]
            + 1.0 * metrics["vicreg_loss"]
        )
        assert metrics["total_loss"] == pytest.approx(expected, abs=1e-5)

    def test_combined_loss_vicreg_weight_scales_total(self):
        emb, pc, emb_tbd, sigreg = self._make_args()
        _, m_off = combined_loss(
            emb, pc, emb_tbd, 0.1, sigreg, vicreg_weight=0.0
        )
        _, m_on = combined_loss(
            emb, pc, emb_tbd, 0.1, sigreg, vicreg_weight=2.0
        )
        # The two calls are deterministic given the same inputs, but the
        # .item() conversions and the separate reassociation of the
        # weighted-add introduce fp reordering error. 1e-3 is well above
        # epsilon for tensors of this magnitude.
        assert m_off["total_loss"] + 2.0 * m_on["vicreg_loss"] == pytest.approx(
            m_on["total_loss"], abs=1e-3
        )

    @pytest.mark.smoke
    def test_combined_loss_vicreg_off_does_not_call_vicreg_loss(self):
        """Pin the bit-identical-on-disable contract: when vicreg_weight == 0,
        vicreg_loss must never be called (no extra std/cov tensor allocations)."""
        emb, pc, emb_tbd, sigreg = self._make_args()
        with patch("wally.training.losses.vicreg_loss") as mocked:
            combined_loss(emb, pc, emb_tbd, 0.1, sigreg, vicreg_weight=0.0)
            assert mocked.call_count == 0

    def test_combined_loss_vicreg_on_calls_vicreg_loss(self):
        emb, pc, emb_tbd, sigreg = self._make_args()
        with patch(
            "wally.training.losses.vicreg_loss",
            return_value=torch.tensor(0.42),
        ) as mocked:
            combined_loss(emb, pc, emb_tbd, 0.1, sigreg, vicreg_weight=1.0)
            assert mocked.call_count == 1

    @pytest.mark.smoke
    def test_combined_loss_metrics_shape_vicreg_off(self):
        emb, pc, emb_tbd, sigreg = self._make_args()
        _, metrics = combined_loss(
            emb, pc, emb_tbd, 0.1, sigreg, vicreg_weight=0.0
        )
        assert set(metrics.keys()) == {"prediction_loss", "sigreg_loss", "total_loss"}

    @pytest.mark.smoke
    def test_combined_loss_metrics_shape_vicreg_on(self):
        emb, pc, emb_tbd, sigreg = self._make_args()
        _, metrics = combined_loss(
            emb, pc, emb_tbd, 0.1, sigreg, vicreg_weight=1.0
        )
        assert set(metrics.keys()) == {
            "prediction_loss",
            "sigreg_loss",
            "vicreg_loss",
            "total_loss",
        }

    def test_combined_loss_vicreg_std_target_changes_std_term(self):
        emb, pc, emb_tbd, sigreg = self._make_args()
        _, m_low = combined_loss(
            emb, pc, emb_tbd, 0.0, sigreg,
            vicreg_weight=1.0, vicreg_std_target=0.1, vicreg_cov_weight=0.0,
        )
        _, m_high = combined_loss(
            emb, pc, emb_tbd, 0.0, sigreg,
            vicreg_weight=1.0, vicreg_std_target=2.0, vicreg_cov_weight=0.0,
        )
        assert m_high["vicreg_loss"] != pytest.approx(m_low["vicreg_loss"], abs=1e-3)

    def test_combined_loss_vicreg_cov_weight_scales_cov_term(self):
        emb, pc, emb_tbd, sigreg = self._make_args()
        _, m_c0 = combined_loss(
            emb, pc, emb_tbd, 0.0, sigreg,
            vicreg_weight=1.0, vicreg_std_target=0.0, vicreg_cov_weight=0.0,
        )
        _, m_c5 = combined_loss(
            emb, pc, emb_tbd, 0.0, sigreg,
            vicreg_weight=1.0, vicreg_std_target=0.0, vicreg_cov_weight=5.0,
        )
        assert m_c5["vicreg_loss"] > m_c0["vicreg_loss"]
