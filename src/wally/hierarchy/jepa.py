"""JEPA-style hierarchical world-model predictor.

A :class:`JEPAWorldModel` predicts the L_n-embedding of a state K_n
frames in the future, conditioned on a target embedding ``g_n``. It is
deliberately simple — AdaLN-style Transformer predictor over a single
state vector, no pixel reconstruction, no action conditioning. Higher
layers (L1, L2, L3) all instantiate the same class with different
hyperparameters; the L0 LeWorldModel is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wally.models.lewm_blocks import ConditionalBlock, Transformer


class JEPAWorldModel(nn.Module):
    """A single-layer JEPA world model.

    Forward signature is ``predict(s_t, g) -> s_{t+K}``: given the current
    state embedding and a target embedding (typically the future state
    itself during training, or a planned target during inference), predict
    the next state embedding K frames later.

    Args:
        state_dim: Embedding dimension of the input state ``s_t`` (and the
            predicted output). Usually equals ``D`` for the layer.
        target_dim: Embedding dimension of the target ``g``. Typically
            also ``D``; can differ if the layer is conditioned on an
            embedding from a different space.
        hidden_dim: Transformer hidden width.
        depth: Number of ``ConditionalBlock`` layers.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio in the feed-forward blocks.
        dropout: Dropout probability in attention/MLP blocks.
    """

    def __init__(
        self,
        state_dim: int,
        target_dim: int,
        hidden_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {state_dim}")
        if target_dim < 1:
            raise ValueError(f"target_dim must be >= 1, got {target_dim}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")

        self.state_dim = state_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.num_heads = num_heads

        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.target_proj = nn.Linear(target_dim, hidden_dim)
        self.transformer = Transformer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=state_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            block_class=ConditionalBlock,
        )

    def predict(self, s_t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Predict the L_n-embedding K frames in the future.

        Args:
            s_t: Current state embedding, shape ``(B, state_dim)``.
            g: Target embedding, shape ``(B, target_dim)``.

        Returns:
            Predicted next state embedding, shape ``(B, state_dim)``.
        """
        if s_t.dim() != 2:
            raise ValueError(
                f"s_t must be 2D (B, state_dim), got shape {tuple(s_t.shape)}"
            )
        if g.dim() != 2:
            raise ValueError(
                f"g must be 2D (B, target_dim), got shape {tuple(g.shape)}"
            )
        if s_t.shape[0] != g.shape[0]:
            raise ValueError(
                f"Batch sizes must match: s_t={s_t.shape[0]}, g={g.shape[0]}"
            )

        x = self.state_proj(s_t).unsqueeze(1)
        c = self.target_proj(g).unsqueeze(1)
        out = self.transformer(x, c).squeeze(1)
        return out

    def forward(self, s_t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.predict(s_t, g)
