from __future__ import annotations

import torch
from torch import Tensor

from wally.training.sigreg import SIGReg


def prediction_loss(emb: Tensor, predicted_change: Tensor) -> Tensor:
    """Residual prediction loss: MSE between the true and predicted
    frame-to-frame change in projected latent space.

    Matches the LeWorldModel paper (Algorithm 1, line 303):
        ``pred_loss = F.mse_loss(emb[:, 1:] - next_emb[:, :-1])``

    Args:
        emb:              (B, T, embed_dim) projected encoder embeddings
        predicted_change: (B, T-1, embed_dim) — the predictor's output

    Returns:
        Scalar MSE loss between the true change (emb[:, 1:] - emb[:, :-1])
        and the predicted change.
    """
    target_change = emb[:, 1:] - emb[:, :-1]
    return torch.nn.functional.mse_loss(target_change, predicted_change)


def combined_loss(
    emb: Tensor,
    predicted_change: Tensor,
    embeddings: Tensor,
    alpha: float,
    sigreg_module: SIGReg,
) -> tuple[Tensor, dict[str, float]]:
    """Combined prediction + SIGReg loss.

    Args:
        emb:              (B, T, embed_dim) projected encoder embeddings.
            Used to compute the residual prediction target
            ``emb[:, 1:] - emb[:, :-1]``.
        predicted_change: (B, T-1, embed_dim) — the predictor's output
            (frame-to-frame delta in latent space).
        embeddings:       (T, B, embed_dim) — the projected encoder
            embeddings, transposed to time-first shape at the model
            boundary. The SIGReg input contract is (T, B, D); this function
            does NOT re-transpose its input.
        alpha:            weight for SIGReg regularization term
        sigreg_module:    stateless SIGReg module (Epps-Pulley statistic)

    Returns:
        (total_loss, metrics_dict) where total_loss = pred_loss + alpha * sigreg_loss.
    """
    pred_loss = prediction_loss(emb, predicted_change)
    s_loss = sigreg_module(embeddings)
    total = pred_loss + alpha * s_loss

    metrics = {
        "prediction_loss": pred_loss.item(),
        "sigreg_loss": s_loss.item(),
        "total_loss": total.item(),
    }
    return total, metrics
