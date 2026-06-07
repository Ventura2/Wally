from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    critic_optimizer: Optimizer,
    global_step: int,
    config: dict[str, Any],
) -> None:
    """Save training checkpoint.

    Args:
        path: File path for the checkpoint.
        model: Main model.
        optimizer: Main model optimizer.
        critic_optimizer: SIGReg critic optimizer.
        global_step: Current training step.
        config: Training configuration dict to embed in checkpoint.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "critic_optimizer_state_dict": critic_optimizer.state_dict(),
            "global_step": global_step,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    critic_optimizer: Optimizer | None = None,
) -> int:
    """Load training checkpoint and restore states.

    Args:
        path: Checkpoint file path.
        model: Main model to restore.
        optimizer: Main model optimizer to restore (optional).
        critic_optimizer: SIGReg critic optimizer to restore (optional).

    Returns:
        The saved global_step.
    """
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if critic_optimizer is not None and "critic_optimizer_state_dict" in checkpoint:
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

    return checkpoint["global_step"]
