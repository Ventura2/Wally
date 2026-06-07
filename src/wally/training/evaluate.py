from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wally.training.logging import log_metrics
from wally.training.losses import prediction_loss

logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plot_latent_comparison(
    predicted: torch.Tensor,
    target: torch.Tensor,
    output_path: Path,
    step: int,
) -> None:
    """Save a visualization comparing predicted vs target latents.

    Creates two subplots:
    1. Per-dimension mean of predicted vs target latents (line plot)
    2. Cosine similarity heatmap between predicted and target timesteps

    Args:
        predicted: (B, T-1, embed_dim) predicted latents
        target:    (B, T-1, embed_dim) target latents
        output_path: Path to save the PNG.
        step: Current training step (for title).
    """
    pred = predicted.detach().float().cpu()
    tgt = target.detach().float().cpu()

    # Per-dimension mean across batch and time
    pred_mean = pred.mean(dim=(0, 1))  # (embed_dim,)
    tgt_mean = tgt.mean(dim=(0, 1))  # (embed_dim,)

    # Cosine similarity between predicted and target at each timestep
    # Average over batch, then compute pairwise cos-sim across timesteps
    pred_bt = pred.mean(dim=0)  # (T-1, embed_dim)
    tgt_bt = tgt.mean(dim=0)  # (T-1, embed_dim)
    pred_norm = torch.nn.functional.normalize(pred_bt, dim=-1)
    tgt_norm = torch.nn.functional.normalize(tgt_bt, dim=-1)
    cos_sim = (pred_norm * tgt_norm).sum(dim=-1)  # (T-1,)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: per-dimension mean latent magnitudes
    dims = range(len(pred_mean))
    ax1.plot(dims, pred_mean.numpy(), label="Predicted", alpha=0.8)
    ax1.plot(dims, tgt_mean.numpy(), label="Target", alpha=0.8)
    ax1.set_xlabel("Latent dimension")
    ax1.set_ylabel("Mean magnitude")
    ax1.set_title(f"Per-dimension mean latents (step {step})")
    ax1.legend()

    # Subplot 2: cosine similarity per timestep
    timesteps = range(len(cos_sim))
    ax2.bar(timesteps, cos_sim.numpy(), color="steelblue")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Cosine similarity")
    ax2.set_title(f"Predicted vs Target cosine similarity (step {step})")
    ax2.set_ylim(-1.0, 1.0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader[Any],
    device: torch.device,
    output_dir: str | Path,
    step: int,
) -> float:
    """Run model on a single validation batch and return loss.

    Args:
        model: LeWorldModel instance.
        val_loader: Validation dataloader.
        device: Device to run on.
        output_dir: Directory to save visualizations.
        step: Current training step (for naming artifacts).

    Returns:
        Validation prediction loss as a float.
    """
    model.eval()
    output_dir = Path(output_dir)

    batch = next(iter(val_loader))
    frames = batch["frames"].to(device)
    actions = batch["actions"].to(device)

    predicted, target = model(frames, actions)
    val_loss = prediction_loss(predicted, target)

    # Save visualization
    img_path = output_dir / f"eval_step_{step}.png"
    _plot_latent_comparison(predicted, target, img_path, step)

    model.train()
    return val_loss.item()


def evaluate_and_log(
    model: nn.Module,
    val_loader: DataLoader[Any],
    device: torch.device,
    output_dir: str | Path,
    step: int,
) -> float:
    """Evaluate and log results to wandb.

    Args:
        model: LeWorldModel instance.
        val_loader: Validation dataloader.
        device: Device to run on.
        output_dir: Directory to save visualizations.
        step: Current training step.

    Returns:
        Validation prediction loss as a float.
    """
    import wandb

    val_loss = evaluate(model, val_loader, device, output_dir, step)

    log_metrics({"val_prediction_loss": val_loss}, step)

    img_path = Path(output_dir) / f"eval_step_{step}.png"
    if img_path.exists() and wandb.run is not None:
        wandb.log({"val_latent_viz": wandb.Image(str(img_path))}, step=step)

    return val_loss
