from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class WorldModelProtocol(Protocol):
    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode a frame to latent. (B, C, H, W) -> (B, Z)"""
        ...

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next latent. z: (B, Z), action: (B, A) -> (B, Z)"""
        ...
