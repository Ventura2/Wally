from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vit_variant: str = "vit_tiny_patch16_224"
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    action_dim: int = 25
    pretrained: bool = True
    encoder_type: str = "vit"  # "vit" or "cnn"
