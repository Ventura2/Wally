from __future__ import annotations

import torch
from torch import Tensor

from wally.training.sigreg import SIGRegCritic, sigreg_loss


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
    critic: SIGRegCritic,
    alpha: float = 0.1,
) -> tuple[Tensor, dict[str, float]]:
    """Combined prediction + SIGReg loss.

    Args:
        predicted: (B, T-1, embed_dim) predicted latents
        target:    (B, T-1, embed_dim) target latents
        critic:    SIGReg critic network
        alpha:     weight for SIGReg regularization term

    Returns:
        (total_loss, metrics_dict)
    """
    pred_loss = prediction_loss(predicted, target)
    s_loss = sigreg_loss(critic, predicted, target)
    total = pred_loss + alpha * s_loss

    metrics = {
        "prediction_loss": pred_loss.item(),
        "sigreg_loss": s_loss.item(),
        "total_loss": total.item(),
    }
    return total, metrics
