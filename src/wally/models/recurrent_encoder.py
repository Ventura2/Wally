from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from wally.models.encoder import ViTEncoder


class RecurrentEncoder(nn.Module):
    """ViTEncoder + single-layer LSTM for memory-augmented frame encoding."""

    def __init__(
        self,
        vit_variant: str = "vit_tiny_patch16_224",
        pretrained: bool = True,
        hidden_size: int = 192,
        memory_length: int = 16,
        recurrence: bool = True,
    ) -> None:
        super().__init__()
        self.vit_encoder = ViTEncoder(variant=vit_variant, pretrained=pretrained)
        embed_dim = self.vit_encoder.embed_dim
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_size, embed_dim)
        self.hidden_size: int = hidden_size
        self.memory_length: int = memory_length
        self.recurrence: bool = recurrence
        self._hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def embed_dim(self) -> int:
        return self.vit_encoder.embed_dim

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a single frame with optional LSTM memory.

        Args:
            frames: (B, 3, H, W) single RGB frame.

        Returns:
            (B, embed_dim) context-augmented latent.
        """
        tokens = self.vit_encoder(frames)
        pooled = tokens.mean(dim=1)

        if not self.recurrence:
            return pooled

        seq = pooled.unsqueeze(1)
        lstm_out, self._hidden = self.lstm(seq, self._hidden)
        return self.output_proj(lstm_out.squeeze(1))

    def forward_sequence(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Encode a sequence of T frames through the LSTM.

        Args:
            frames: (B, T, 3, H, W) sequence of RGB frames.

        Returns:
            latents: (B, T, embed_dim) context-augmented latents.
            hidden: final (h, c) hidden state tuple.
        """
        B, T, C, H, W = frames.shape
        flat = rearrange(frames, "b t c h w -> (b t) c h w")
        tokens = self.vit_encoder(flat)
        pooled = rearrange(tokens, "(b t) n d -> b t n d", b=B, t=T)
        pooled_frames = pooled.mean(dim=2)

        if not self.recurrence:
            return pooled_frames, None  # type: ignore[return-value]

        lstm_out, hidden = self.lstm(pooled_frames, self._hidden)
        self._hidden = hidden
        latents = self.output_proj(lstm_out)
        return latents, hidden

    def reset_hidden(
        self,
        batch_size: int = 1,
        device: torch.device | str | None = None,
    ) -> None:
        """Reset the LSTM hidden state."""
        self._hidden = None

    def get_hidden(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the current hidden state without modifying it."""
        return self._hidden

    def set_hidden(self, hidden: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Set the hidden state to a provided (h, c) tuple."""
        self._hidden = hidden
