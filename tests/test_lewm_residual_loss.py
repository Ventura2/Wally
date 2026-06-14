"""Regression tests for the residual-loss contract (fix-lewm-training-collapse).

These tests pin the LeWorldModel + combined_loss + SIGReg contract that
aligns with the LeWM paper (Algorithm 1, line 303):

    pred_loss = F.mse_loss(emb[:, 1:] - emb[:, :-1], predicted_change)

where ``predicted_change`` is the predictor's output (the frame-to-frame
delta in latent space), NOT the absolute next-frame latent.

The SIGReg input is the projected encoder output, transposed to (T, B, D)
exactly once at the model boundary. The SIGReg module does NOT re-transpose.

These tests are expected to FAIL on the pre-change code:
  - ``test_combined_loss_is_residual`` — current loss is MSE(predicted, emb[:, 1:]).
  - ``test_sigreg_input_shape_is_TBD`` — current code double-transposes to (B, T, D).
  - ``test_combined_loss_sigreg_no_double_transpose`` — same as above.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from wally.models.lewm import LeWorldModel
from wally.training.losses import combined_loss
from wally.training.sigreg import SIGReg

BATCH_SIZE = 2
SEQ_LENGTH = 4
EMBED_DIM = 64
ACTION_DIM = 25
IMAGE_SIZE = 224


class _SyntheticIterableDataset(IterableDataset):
    """Synthetic dataset yielding one batch of the required shape."""

    def __init__(self, num_batches: int = 1) -> None:
        self.num_batches = num_batches

    def __iter__(self):
        torch.manual_seed(0)
        for _ in range(self.num_batches):
            yield {
                "frames": torch.rand(
                    BATCH_SIZE, SEQ_LENGTH, 3, IMAGE_SIZE, IMAGE_SIZE
                ),
                "actions": torch.clamp(
                    torch.randn(BATCH_SIZE, SEQ_LENGTH, ACTION_DIM), -1.0, 1.0
                ),
            }


def _identity_collate(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return batch


def _make_model() -> LeWorldModel:
    torch.manual_seed(0)
    return LeWorldModel(
        encoder_type="cnn",
        embed_dim=EMBED_DIM,
        depth=2,
        num_heads=4,
        action_dim=ACTION_DIM,
        num_frames=SEQ_LENGTH,
    )


def _make_loader(num_batches: int = 1) -> DataLoader:
    return DataLoader(
        _SyntheticIterableDataset(num_batches),
        batch_size=None,
        collate_fn=_identity_collate,
    )


class TestResidualLossContract:
    @pytest.mark.smoke
    def test_combined_loss_is_residual(self) -> None:
        """prediction_loss is the MSE between true and predicted change."""
        torch.manual_seed(0)
        model = _make_model()
        sigreg = SIGReg(knots=9, num_proj=32)
        loader = _make_loader(num_batches=1)
        batch = next(iter(loader))

        out = model(batch["frames"], batch["actions"], return_embeddings=True)
        # New contract: model returns (predicted_change, emb). The third
        # element is the SIGReg input in (T, B, D).
        if len(out) == 2:
            predicted_change, emb_T_B_D = out
        else:
            # Pre-fix: (predicted, target, emb_T_B_D) — the predicted tensor
            # is the absolute next latent, not the residual change.
            predicted_abs, _target, emb_T_B_D = out
            predicted_change = predicted_abs
        emb_B_T_D = emb_T_B_D.transpose(0, 1).contiguous()

        total_loss, metrics = combined_loss(
            emb_B_T_D, predicted_change, emb_T_B_D, alpha=0.1, sigreg_module=sigreg
        )

        expected = F.mse_loss(
            emb_B_T_D[:, 1:] - emb_B_T_D[:, :-1], predicted_change
        )

        assert torch.isfinite(total_loss)
        assert metrics["prediction_loss"] == pytest.approx(expected.item(), abs=1e-5), (
            "combined_loss is not computing the residual loss: "
            f"got {metrics['prediction_loss']}, expected {expected.item()}"
        )
        assert metrics["prediction_loss"] > 0.0, (
            "prediction_loss must be strictly positive — a zero loss means the "
            "predictor is exploiting frame-to-frame smoothness and not learning "
            "dynamics (the regression observed on 2026-06-14)."
        )

    @pytest.mark.smoke
    def test_sigreg_input_shape_is_TBD(self) -> None:
        """SIGReg.forward receives a (T, B, D) tensor where T == seq_length."""
        torch.manual_seed(0)
        model = _make_model()
        sigreg = SIGReg(knots=9, num_proj=32)
        loader = _make_loader(num_batches=1)
        batch = next(iter(loader))

        recorded: dict[str, tuple[int, ...]] = {}
        original_forward = sigreg.forward

        def recording_forward(proj: torch.Tensor) -> torch.Tensor:
            recorded["shape"] = tuple(proj.shape)
            return original_forward(proj)

        sigreg.forward = recording_forward  # type: ignore[method-assign]

        out = model(batch["frames"], batch["actions"], return_embeddings=True)
        if len(out) == 2:
            predicted_change, emb_T_B_D = out
        else:
            predicted_abs, _target, emb_T_B_D = out
            predicted_change = predicted_abs
        emb_B_T_D = emb_T_B_D.transpose(0, 1).contiguous()

        combined_loss(
            emb_B_T_D, predicted_change, emb_T_B_D, 0.1, sigreg_module=sigreg
        )

        assert "shape" in recorded, "SIGReg.forward was not called"
        sigreg_shape = recorded["shape"]
        assert sigreg_shape[0] == SEQ_LENGTH, (
            f"SIGReg input first dim must be seq_length={SEQ_LENGTH} (time axis), "
            f"got shape {sigreg_shape} — the SIGReg input is not in (T, B, D)"
        )
        assert sigreg_shape[1] == BATCH_SIZE, (
            f"SIGReg input second dim must be batch_size={BATCH_SIZE}, "
            f"got shape {sigreg_shape}"
        )
        assert sigreg_shape[2] == EMBED_DIM, (
            f"SIGReg input third dim must be embed_dim={EMBED_DIM}, "
            f"got shape {sigreg_shape}"
        )

    @pytest.mark.smoke
    def test_combined_loss_sigreg_no_double_transpose(self) -> None:
        """The combined_loss call site must not re-transpose the SIGReg input.

        Pre-fix code re-transposes embeddings inside combined_loss, so SIGReg
        receives (B, T, D) instead of (T, B, D). The new contract locks the
        shape so SIGReg.forward sees the time axis as the first dimension.
        """
        torch.manual_seed(0)
        model = _make_model()
        sigreg = SIGReg(knots=9, num_proj=32)
        loader = _make_loader(num_batches=1)
        batch = next(iter(loader))

        original_forward = sigreg.forward

        def shape_asserting_forward(proj: torch.Tensor) -> torch.Tensor:
            assert proj.dim() == 3, f"SIGReg expected 3D input, got {proj.dim()}D"
            assert proj.shape[0] == SEQ_LENGTH, (
                f"SIGReg first dim must be seq_length={SEQ_LENGTH}, "
                f"got {proj.shape}"
            )
            assert proj.shape[1] == BATCH_SIZE, (
                f"SIGReg second dim must be batch_size={BATCH_SIZE}, "
                f"got {proj.shape}"
            )
            return original_forward(proj)

        sigreg.forward = shape_asserting_forward  # type: ignore[method-assign]

        out = model(batch["frames"], batch["actions"], return_embeddings=True)
        if len(out) == 2:
            predicted_change, emb_T_B_D = out
        else:
            predicted_abs, _target, emb_T_B_D = out
            predicted_change = predicted_abs
        emb_B_T_D = emb_T_B_D.transpose(0, 1).contiguous()

        # This call must not raise — if it does, the double-transpose is still in place.
        total_loss, _ = combined_loss(
            emb_B_T_D, predicted_change, emb_T_B_D, alpha=0.1, sigreg_module=sigreg
        )
        assert torch.isfinite(total_loss)
