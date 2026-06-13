from __future__ import annotations

import torch
import torch.nn as nn

from wally.models.lewm_blocks import ConditionalBlock, Transformer


class ARPredictor(nn.Module):
    """Action-conditioned causal Transformer predictor (LeWM official AdaLN-Zero).

    Wraps the official ``Transformer`` with ``ConditionalBlock`` and a learnable
    positional embedding. The forward signature is ``forward(x, c) -> (B, T, D)``
    where:
      - ``x``: projected encoder output of shape ``(B, T, input_dim)``
      - ``c``: action-embedding sequence of shape ``(B, T, c_dim)``

    The action-embedding sequence is consumed exclusively through AdaLN-Zero
    modulation in the ``ConditionalBlock`` (NOT interleaved into ``x``). The
    positional embedding is added to ``x`` before the first block; the
    conditioning ``c`` is added separately via a learned projection inside the
    ``Transformer`` (``cond_proj``).

    The predictor runs in the same autocast context as the rest of the model
    (bf16 by default). The internal ``nn.LayerNorm`` is ``elementwise_affine=False``
    and the AdaLN-Zero modulation is zero-initialized, so the entire block is
    a strict identity at step 0 — gradients are well-behaved from the first
    step and the predictor contributes zero to the residual until the
    modulation weights learn to grow.
    """

    def __init__(
        self,
        input_dim: int = 192,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        c_dim: int = 192,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_frames: int = 16,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.c_dim = c_dim
        self.num_frames = num_frames

        # Learnable positional embedding added to x before the Transformer.
        # Shape (1, num_frames, input_dim) so it broadcasts over the batch.
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_frames, input_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.transformer = Transformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            block_class=ConditionalBlock,
        )

        # The Transformer created its own input_proj and cond_proj. We need
        # to rebuild cond_proj to use c_dim instead of input_dim (they may
        # differ). The simplest way is to re-create the Transformer with
        # c_dim-aware projections — but the official Transformer assumes
        # cond_dim == input_dim. So we add an explicit c_proj that maps
        # c_dim -> input_dim, and pass c_proj(c) as the conditioning.
        self.c_proj = nn.Linear(c_dim, input_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) projected encoder output.
            c: (B, T, c_dim) action-embedding sequence.

        Returns:
            (B, T, output_dim) predicted latents.
        """
        T = x.size(1)
        if T > self.num_frames:
            raise ValueError(
                f"input sequence length {T} exceeds configured num_frames "
                f"{self.num_frames}; increase num_frames in the ARPredictor "
                f"__init__ or truncate the input"
            )
        x = x + self.pos_embedding[:, :T, :]
        c = self.c_proj(c)
        return self.transformer(x, c)  # type: ignore[no-any-return]
