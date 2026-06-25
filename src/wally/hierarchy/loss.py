"""Loss functions for the hierarchical world-model stack.

The V1 hierarchy uses a single temporal-coherence self-supervised
objective: predict the L_n-embedding of a state K_n frames in the future,
conditioned on the actual future embedding as the target. This is the
JEPA training objective (Bardes et al, 2024) applied to the semantic
level.
"""

from __future__ import annotations

from torch import Tensor

from wally.training.sigreg import SIGReg


def temporal_coherence_loss(
    predicted: Tensor,
    target: Tensor,
) -> Tensor:
    """L2 distance between predicted and actual L_n-embeddings.

    Both inputs have shape ``(B, D)`` — single-vector embeddings (no
    pixel reconstruction; the L_n space is the prediction target).
    """
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted and target must have the same shape, got "
            f"{tuple(predicted.shape)} vs {tuple(target.shape)}"
        )
    return ((predicted - target) ** 2).sum(dim=-1).mean()


def combined_hierarchy_loss(
    predicted: Tensor,
    target: Tensor,
    projected_embeddings: Tensor,
    alpha: float,
    sigreg: SIGReg,
) -> tuple[Tensor, dict[str, float]]:
    """Temporal-coherence loss + SIGReg regulariser on the embedding.

    Args:
        predicted: ``(B, D)`` predicted L_n embedding.
        target: ``(B, D)`` actual L_n embedding.
        projected_embeddings: ``(B, D)`` L_n embeddings (the SIGReg
            input). Must be batch-first; the function transposes to
            ``(1, B, D)`` (a single timestep) before passing to SIGReg so
            the contract is satisfied without fabricating a time axis.
        alpha: Weight on the SIGReg term.
        sigreg: The SIGReg module.
    """
    pred_loss = temporal_coherence_loss(predicted, target)
    s_input = projected_embeddings.unsqueeze(0).contiguous()
    s_loss = sigreg(s_input)
    total = pred_loss + alpha * s_loss
    return total, {
        "prediction_loss": float(pred_loss.detach().item()),
        "sigreg_loss": float(s_loss.detach().item()),
        "total_loss": float(total.detach().item()),
    }
