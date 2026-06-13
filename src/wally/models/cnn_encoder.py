from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp


class SimpleCNNEncoder(nn.Module):
    """Simple CNN encoder for image frames.

    More numerically stable than ViT on ROCm (no LayerNorm backward issues).
    Produces (B, embed_dim) latents per frame.
    """

    def __init__(self, embed_dim: int = 192) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Standard CNN: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2)  # 224 -> 56
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 56 -> 28
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 28 -> 14
        self.conv4 = nn.Conv2d(
            128, embed_dim, kernel_size=4, stride=2, padding=1
        )  # 14 -> 7

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # No BN on last conv - it goes directly to global average pool

    @amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)  # type: ignore[attr-defined,untyped-decorator]
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, 3, 224, 224) RGB frames normalized to [0, 1].

        Returns:
            (B, embed_dim) latent vector per frame.
        """
        x = F.relu(self.bn1(self.conv1(frames)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv4(x))
        # Global average pool
        x = x.mean(dim=[2, 3])  # (B, embed_dim)
        return x
