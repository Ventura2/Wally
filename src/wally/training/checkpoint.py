from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    global_step: int,
    config: dict[str, Any],
    *,
    scheduler: LRScheduler | None = None,
    model_config: dict[str, Any] | None = None,
) -> None:
    """Save training checkpoint.

    Args:
        path: File path for the checkpoint.
        model: Main model.
        optimizer: Main model optimizer.
        global_step: Current training step.
        config: Training configuration dict to embed in checkpoint.
        scheduler: Optional LR scheduler to persist.
        model_config: Optional JSON-serializable dict of the model
            architecture configuration (e.g. ``asdict(ModelConfig())``).
            Stored under the ``model_config`` key in the payload so
            downstream code can reconstruct the model without re-reading
            the YAML config. Must be a plain dict; convert dataclasses
            and ``Path`` objects at the call site.
    """
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "config": config,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if model_config is not None:
        if not isinstance(model_config, dict):
            raise TypeError(
                "model_config must be a plain dict; convert dataclasses "
                "and Path objects at the call site (e.g. asdict(ModelConfig))."
            )
        payload["model_config"] = model_config
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    *,
    scheduler: LRScheduler | None = None,
) -> int:
    """Load training checkpoint and restore states.

    Args:
        path: Checkpoint file path.
        model: Main model to restore.
        optimizer: Main model optimizer to restore (optional).
        scheduler: Optional LR scheduler to restore.

    Returns:
        The saved global_step.
    """
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            global_step = int(checkpoint.get("global_step", 0))
            scheduler.last_epoch = global_step - 1
            logger.info(
                "legacy checkpoint: scheduler state not found, "
                "initializing at last_epoch=%d",
                global_step - 1,
            )

    return int(checkpoint["global_step"])
