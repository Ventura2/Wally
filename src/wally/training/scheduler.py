from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


def create_scheduler(
    optimizer: Optimizer,
    warmup_steps: int = 1000,
    max_steps: int = 100_000,
) -> LambdaLR:
    """Cosine annealing LR scheduler with linear warmup.

    Linear warmup from 0 to base_lr over ``warmup_steps``, then cosine decay
    from base_lr to 0 over the remaining steps.

    Args:
        optimizer: Optimizer to schedule.
        warmup_steps: Number of linear warmup steps.
        max_steps: Total training steps.

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
