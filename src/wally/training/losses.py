from __future__ import annotations

import torch
from torch import Tensor

from wally.training.sigreg import SIGReg


def prediction_loss(predicted: Tensor, target: Tensor) -> Tensor:
    """MSE loss between predicted and target latents.

    Args:
        predicted: (B, T-1, embed_dim)
        target:    (B, T-1, embed_dim)

    Returns:
        Scalar MSE loss.
    """
    return torch.nn.functional.mse_loss(predicted, target)


def combined_loss(
    predicted: Tensor,
    target: Tensor,
    embeddings: Tensor,
    alpha: float,
    sigreg_module: SIGReg,
) -> tuple[Tensor, dict[str, float]]:
    """Combined prediction + SIGReg loss.

    Args:
        predicted:     (B, T-1, embed_dim) predicted latents
        target:        (B, T-1, embed_dim) target latents
        embeddings:    projected encoder embeddings — the output of the
            ``projector`` MLP in ``LeWorldModel``, not the raw encoder output.
            Shape is (B, T, D) as returned by
            ``LeWorldModel.forward(..., return_embeddings=True)``; it is
            transposed to (T, B, D) before being passed to the SIGReg module.
        alpha:         weight for SIGReg regularization term
        sigreg_module: stateless SIGReg module (Epps-Pulley statistic)

    Returns:
        (total_loss, metrics_dict) where total_loss = pred_loss + alpha * sigreg_loss.
    """
    pred_loss = prediction_loss(predicted, target)
    s_loss = sigreg_module(
        embeddings.transpose(0, 1) if embeddings.dim() == 3 else embeddings
    )
    total = pred_loss + alpha * s_loss

    metrics = {
        "prediction_loss": pred_loss.item(),
        "sigreg_loss": s_loss.item(),
        "total_loss": total.item(),
    }
    return total, metrics
