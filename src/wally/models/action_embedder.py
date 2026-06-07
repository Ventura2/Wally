from __future__ import annotations

import torch
import torch.nn as nn


class ActionEmbedder(nn.Module):
    """Linear projection from action space to embedding space."""

    def __init__(self, action_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(action_dim, embed_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: (B, T, action_dim)

        Returns:
            (B, T, embed_dim)
        """
        return self.proj(actions)
