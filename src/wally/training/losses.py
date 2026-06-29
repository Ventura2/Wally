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


def vicreg_loss(z: Tensor, std_target: float = 1.0, cov_weight: float = 1.0) -> Tensor:
    """VICReg auxiliary loss (Variance-Invariance-Covariance Regularization,
    Bardes et al. 2022), minus the invariance (similarity) term — the
    prediction loss is computed separately by the caller.

    Computes two terms applied to the projected encoder output ``z``:
      - std_loss: ``mean(relu(std_target - z.std(dim=0)))`` — hinge that
        pushes per-dim std toward ``std_target`` (default 1.0). Forces every
        latent dim to carry non-redundant information.
      - cov_loss: ``(off_diag(cov(z)) ** 2).sum() / D`` — squared
        off-diagonal covariance, penalizes correlation between dims.

    Args:
        z:          (B, D) tensor of latent vectors (B = batch size, D = dim).
        std_target: per-dim std target (the hinge ``gamma``).
        cov_weight: weight of the covariance term relative to the std term.

    Returns:
        Scalar tensor ``mean(relu(std_target - z.std(dim=0))) + cov_weight * cov_loss``.
        Batch size 1 produces NaN (the std is undefined for a single sample);
        the L0 training pipeline guarantees ``batch_size >= 4``.
    """
    std = z.std(dim=0)
    std_loss = torch.nn.functional.relu(std_target - std).mean()

    z_centered = z - z.mean(dim=0)
    B = z.shape[0]
    cov = (z_centered.T @ z_centered) / (B - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    D = z.shape[1]
    cov_loss = (off_diag ** 2).sum() / D

    return std_loss + cov_weight * cov_loss


def combined_loss(
    emb: Tensor,
    predicted_change: Tensor,
    embeddings: Tensor,
    alpha: float,
    sigreg_module: SIGReg,
    *,
    vicreg_weight: float = 0.0,
    vicreg_std_target: float = 1.0,
    vicreg_cov_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Combined prediction + SIGReg (+ optional VICReg) loss.

    Args:
        emb:              (B, T, embed_dim) projected encoder embeddings.
            Used to compute the residual prediction target
            ``emb[:, 1:] - emb[:, :-1]`` and the VICReg regularization
            (when ``vicreg_weight > 0``).
        predicted_change: (B, T-1, embed_dim) — the predictor's output
            (frame-to-frame delta in latent space).
        embeddings:       (T, B, embed_dim) — the projected encoder
            embeddings, transposed to time-first shape at the model
            boundary. The SIGReg input contract is (T, B, D); this function
            does NOT re-transpose its input.
        alpha:            weight for SIGReg regularization term
        sigreg_module:    stateless SIGReg module (Epps-Pulley statistic)
        vicreg_weight:    weight for the VICReg auxiliary term. Default 0
            disables VICReg entirely and keeps the metrics dict bit-identical
            to the pre-VICReg output (no ``vicreg_loss`` key).
        vicreg_std_target: per-dim std target (the hinge ``gamma``).
        vicreg_cov_weight: weight of the VICReg covariance term.

    Returns:
        (total_loss, metrics_dict) where
        ``total_loss = pred_loss + alpha * sigreg_loss + vicreg_weight * vicreg(z)``.
    """
    pred_loss = prediction_loss(emb, predicted_change)
    s_loss = sigreg_module(embeddings)
    total = pred_loss + alpha * s_loss

    metrics = {
        "prediction_loss": pred_loss.item(),
        "sigreg_loss": s_loss.item(),
    }

    if vicreg_weight > 0.0:
        B, _, D = emb.shape
        flat = emb.reshape(B * emb.shape[1], D)
        v_loss = vicreg_loss(flat, vicreg_std_target, vicreg_cov_weight)
        total = total + vicreg_weight * v_loss
        metrics["vicreg_loss"] = v_loss.item()

    metrics["total_loss"] = total.item()
    return total, metrics
