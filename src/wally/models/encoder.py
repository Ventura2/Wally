from __future__ import annotations

import timm
import torch
import torch.nn as nn
from einops import rearrange


class ViTEncoder(nn.Module):
    """ViT Tiny encoder wrapper that outputs latent patch tokens."""

    def __init__(
        self, variant: str = "vit_tiny_patch16_224", pretrained: bool = True
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(variant, pretrained=pretrained, num_classes=0)
        self.embed_dim: int = self.backbone.embed_dim  # 192 for vit_tiny

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, 3, 224, 224) RGB frames.

        Returns:
            (B, 196, 192) patch token latents (CLS token stripped).
        """
        # forward_features returns CLS + patch tokens: (B, 1+N, embed_dim)
        features = self.backbone.forward_features(frames)
        # strip CLS token at index 0 → (B, N, embed_dim)
        patch_tokens = features[:, 1:, :]
        return rearrange(patch_tokens, "b n d -> b n d")
