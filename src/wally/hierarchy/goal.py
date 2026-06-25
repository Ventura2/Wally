"""Learned goal embeddings for the hierarchy layers.

A :class:`LearnedGoalEmbedding` is a single ``nn.Parameter`` of shape
``(D,)`` per task. For V1 it is optimised end-to-end on the same
temporal-coherence loss as the JEPA world model — i.e. the goal is
trained to be a target the predictor can hit, which in turn shapes the
predictor toward a useful representation.

For V2+ the parameter can be replaced by a projection from a language
embedding; the public interface stays the same.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedGoalEmbedding(nn.Module):
    """A single learned ``(D,)`` goal vector.

    Args:
        D: Embedding dimension.
        init: Initial value. ``None`` samples from ``N(0, 1e-2)`` so the
            predictor's first pass is roughly the embedding of an
            "average" state.
        task_id: Optional human-readable task name (purely metadata for
            logging/checkpointing; the runtime path never reads it).
    """

    def __init__(
        self,
        D: int,
        *,
        init: torch.Tensor | None = None,
        task_id: str | None = None,
    ) -> None:
        super().__init__()
        if D < 1:
            raise ValueError(f"D must be >= 1, got {D}")
        if init is None:
            init = torch.randn(D) * 1e-2
        else:
            if init.shape != (D,):
                raise ValueError(
                    f"init must have shape ({D},), got {tuple(init.shape)}"
                )
        self.g = nn.Parameter(init.clone())
        self.D = D
        self.task_id = task_id

    def forward(self) -> torch.Tensor:
        return self.g

    def as_target(self) -> torch.Tensor:
        """Return the goal as a 1D ``Tensor[D]`` (the conventional target shape)."""
        return self.g.detach().clone().squeeze()
