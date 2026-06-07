from __future__ import annotations

import torch
import torch.nn as nn


class CausalTransformerPredictor(nn.Module):
    """Decoder-only Transformer with causal masking for next-latent prediction."""

    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # causal mask: upper-triangular True means "blocked"
        mask = nn.Transformer.generate_square_subsequent_mask(1)  # placeholder
        self.register_buffer("causal_mask", mask)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2*T, embed_dim) interleaved latent/action sequence.

        Returns:
            (B, T, embed_dim) predicted latents at even (latent) positions.
        """
        seq_len = x.size(1)
        mask = self._get_causal_mask(seq_len, x.device)

        # self-attention with causal mask (no separate memory)
        out = self.decoder(x, memory=x, tgt_mask=mask, memory_mask=mask)
        out = self.norm(out)

        # extract predictions at even indices (latent positions)
        return out[:, 0::2, :]
