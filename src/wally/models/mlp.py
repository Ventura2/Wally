from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Two-layer MLP with optional normalization and activation (LeWM official).

    When ``norm_fn=nn.BatchNorm1d``, the forward reshapes a ``(B, T, D)`` input
    to ``(B*T, D)`` for the norm, then reshapes the output back to ``(B, T, D')``
    so the rest of the pipeline keeps a clean batch-major shape. For all other
    norms (LayerNorm, Identity) the input is fed through ``self.net`` as-is.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        norm_fn: type[nn.Module] = nn.LayerNorm,
        act_fn: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self._norm_kind = norm_fn
        norm_layer: nn.Module
        if norm_fn is nn.Identity or norm_fn is None:
            norm_layer = nn.Identity()
        else:
            norm_layer = norm_fn(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer,
            act_fn(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._norm_kind is nn.BatchNorm1d and x.dim() == 3:
            shape = x.shape
            x = x.reshape(-1, shape[-1])
            x = self.net(x)
            x = x.reshape(*shape[:-1], -1)
            return x
        return self.net(x)  # type: ignore[no-any-return]
