from __future__ import annotations

from typing import Any

import wandb


def init_wandb(config: dict[str, Any], project_name: str = "wally") -> None:
    """Initialize a wandb run.

    Args:
        config: Configuration dict to log as run config.
        project_name: wandb project name.
    """
    wandb.init(project=project_name, config=config)


def log_metrics(metrics: dict[str, Any], step: int) -> None:
    """Log metrics to wandb at the given step.

    Args:
        metrics: Dict of metric name → value.
        step: Global training step.
    """
    wandb.log(metrics, step=step)
