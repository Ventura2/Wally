from __future__ import annotations

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
) -> Optimizer:
    """Create AdamW optimizer with separate param groups for weight decay.

    Bias and LayerNorm parameters are excluded from weight decay.

    Args:
        model: The model to optimize.
        lr: Learning rate.
        weight_decay: Weight decay coefficient.

    Returns:
        Configured AdamW optimizer.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim <= 1
            or "bias" in name
            or "LayerNorm" in name
            or "layernorm" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(param_groups, lr=lr)
