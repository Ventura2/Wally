from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from wally.models.action_embedder import ActionEmbedder
from wally.models.encoder import ViTEncoder
from wally.models.predictor import CausalTransformerPredictor


class LeWorldModel(nn.Module):
    """Latent Embedding World Model — composes encoder, action embedder, predictor."""

    def __init__(
        self,
        vit_variant: str = "vit_tiny_patch16_224",
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        action_dim: int = 25,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = ViTEncoder(variant=vit_variant, pretrained=pretrained)
        self.action_embedder = ActionEmbedder(action_dim, embed_dim)
        self.predictor = CausalTransformerPredictor(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(
        self, frames: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frames:  (B, T, 3, 224, 224)
            actions: (B, T, action_dim)

        Returns:
            predicted_latents: (B, T-1, embed_dim)
            target_latents:    (B, T-1, embed_dim)
        """
        B, T, C, H, W = frames.shape

        # encode all frames → (B, T, N, embed_dim) then pool patches → (B, T, embed_dim)
        flat = rearrange(frames, "b t c h w -> (b t) c h w")
        tokens = self.encoder(flat)  # (B*T, 196, embed_dim)
        latents = tokens.mean(dim=1)  # (B*T, embed_dim) — mean-pool patches
        latents = rearrange(latents, "(b t) d -> b t d", b=B, t=T)

        # targets: next-frame latents
        target_latents = latents[:, 1:]  # (B, T-1, embed_dim)

        # predictor input: interleave current latents + actions
        current_latents = latents[:, :-1]  # (B, T-1, embed_dim)
        current_actions = self.action_embedder(actions[:, :-1])  # (B, T-1, embed_dim)

        # interleave: even=latent, odd=action → (B, 2*(T-1), embed_dim)
        interleaved = torch.stack([current_latents, current_actions], dim=2)
        interleaved = rearrange(interleaved, "b t two d -> b (t two) d")

        # predict → (B, T-1, embed_dim) at even positions
        predicted_latents = self.predictor(interleaved)

        return predicted_latents, target_latents
