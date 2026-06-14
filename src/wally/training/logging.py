from __future__ import annotations

from typing import Any

import wandb


def init_wandb(
    config: dict[str, Any],
    project_name: str = "wally",
    *,
    name: str | None = None,
) -> None:
    """Initialize a wandb run.

    Args:
        config: Configuration dict to log as run config.
        project_name: wandb project name.
        name: Optional display name for the wandb run. When provided, this
            is forwarded to ``wandb.init(name=...)`` so resumed runs can be
            distinguished in the dashboard by their starting step.
    """
    wandb.init(project=project_name, config=config, name=name)


def log_metrics(metrics: dict[str, Any], step: int) -> None:
    """Log metrics to wandb at the given step.

    Args:
        metrics: Dict of metric name → value.
        step: Global training step.
    """
    wandb.log(metrics, step=step)
