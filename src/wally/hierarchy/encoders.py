"""Encoders for the hierarchy layers.

Each L_n encoder wraps a frozen lower-layer encoder (the L0 LeWorldModel
encoder for L1, the L1 encoder for L2, etc.) and adds a learned linear
projection to the layer's own embedding dimension ``D_n``. All encoders
share the same interface: ``encode(frames) -> Tensor[..., D_n]`` so the
training code can swap them transparently.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from wally.models.lewm import LeWorldModel
from wally.training.checkpoint import load_checkpoint


class BaseLayerEncoder(nn.Module):
    """Shared scaffold for L1/L2/L3 encoders.

    The lower-layer encoder is held as a frozen reference (it is
    registered as a submodule but its parameters are frozen after
    initialisation). The learned linear projection ``self.proj`` maps
    from the lower layer's embedding dim to this layer's ``D``.

    Subclasses must implement :meth:`_encode_lower` to call the
    lower-layer encoder and return a ``(B, D_lower)`` tensor.
    """

    def __init__(self, lower_dim: int, D: int) -> None:
        super().__init__()
        if lower_dim < 1:
            raise ValueError(f"lower_dim must be >= 1, got {lower_dim}")
        if D < 1:
            raise ValueError(f"D must be >= 1, got {D}")
        self.lower_dim = lower_dim
        self.D = D
        self.proj = nn.Linear(lower_dim, D)

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.proj.weight.requires_grad = True
        self.proj.bias.requires_grad = True

    def _encode_lower(self, frames: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        lower = self._encode_lower(frames)
        if lower.dim() == 3:
            lower = lower.mean(dim=1)
        return self.proj(lower)


class L1Encoder(BaseLayerEncoder):
    """L1 encoder: frozen L0 encoder + linear projection to ``D1``.

    Wraps a :class:`LeWorldModel` (or its encoder submodule directly).
    Pass either an already-loaded :class:`LeWorldModel` or a checkpoint
    path (the model is then loaded and frozen).

    Args:
        model_or_path: Either a :class:`LeWorldModel` instance or a path
            to a checkpoint to load. When given a path, the model is
            loaded in ``eval()`` mode with all parameters frozen.
        D1: L1 embedding dimension.
    """

    def __init__(self, model_or_path: LeWorldModel | str, D1: int = 64) -> None:
        if isinstance(model_or_path, str):
            ck = torch.load(model_or_path, map_location="cpu", weights_only=False)
            model_config = ck.get("model_config", {}) or {}
            embed_dim = int(model_config.get("embed_dim", 192))
            depth = int(model_config.get("depth", 4))
            num_heads = int(model_config.get("num_heads", 4))
            mlp_ratio = float(model_config.get("mlp_ratio", 4.0))
            dropout = float(model_config.get("dropout", 0.1))
            encoder_type = model_config.get("encoder_type", "cnn")
            pretrained = bool(model_config.get("pretrained", False))
            model = LeWorldModel(
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                encoder_type=encoder_type,
                pretrained=pretrained,
            )
            load_checkpoint(model_or_path, model)
            lower_dim = embed_dim
        else:
            model = model_or_path
            lower_dim = model.projector.net[3].out_features
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        super().__init__(lower_dim=lower_dim, D=D1)
        self.l0_model = model
        self._freeze()

    @torch.no_grad()
    def _encode_lower(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() == 4:
            x = frames.unsqueeze(1)
        elif frames.dim() == 5:
            x = frames
        else:
            raise ValueError(
                "frames must be 4D (B, C, H, W) or 5D (B, T, C, H, W); "
                f"got {frames.dim()}D"
            )

        is_cnn = getattr(self.l0_model, "_is_cnn", False)
        flat = rearrange(x, "b t c h w -> (b t) c h w")
        if is_cnn:
            latents = self.l0_model.encoder(flat)
        else:
            tokens = self.l0_model.encoder(flat)
            latents = tokens.mean(dim=1)
        latents = latents.unsqueeze(1) if latents.dim() == 2 else latents
        projected = self.l0_model._projector_fp32(latents)
        return projected.squeeze(1) if projected.dim() == 3 else projected

    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a 5D frame tensor to a sequence of L1 embeddings.

        Args:
            frames: ``(B, T, 3, H, W)`` uint8/float batch of frames.
        Returns:
            ``(B, T, D1)`` L1 embeddings.
        """
        B, T, C, H, W = frames.shape
        flat = rearrange(frames, "b t c h w -> (b t) c h w")
        with torch.no_grad():
            is_cnn = getattr(self.l0_model, "_is_cnn", False)
            if is_cnn:
                latents_flat = self.l0_model.encoder(flat)
            else:
                tokens = self.l0_model.encoder(flat)
                latents_flat = tokens.mean(dim=1)
            latents = rearrange(latents_flat, "(b t) d -> b t d", b=B, t=T)
            projected = self.l0_model._projector_fp32(latents)
        return self.proj(projected)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        l0_model: LeWorldModel,
        D1: int = 64,
    ) -> "L1Encoder":
        """Build an :class:`L1Encoder` from a previously saved L1 checkpoint.

        Args:
            checkpoint_path: Path to an L1 checkpoint saved by
                :class:`HierarchyTrainer._save_checkpoint`. Must contain
                ``encoder_state_dict``.
            l0_model: The frozen L0 :class:`LeWorldModel` instance (or
                its checkpoint path) used to build the encoder. This must
                match the L0 that was used when training the L1 layer.
            D1: L1 embedding dimension (must match the saved projection).
        """
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        enc_sd = ck.get("encoder_state_dict")
        if enc_sd is None:
            raise ValueError(
                f"{checkpoint_path} has no encoder_state_dict — not an L1 checkpoint"
            )
        if isinstance(l0_model, str):
            l0_instance = cls(l0_model, D1=D1).l0_model
        else:
            l0_instance = l0_model
        enc = cls(l0_instance, D1=D1)
        enc.load_state_dict(enc_sd, strict=False)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        enc.proj.weight.requires_grad = True
        enc.proj.bias.requires_grad = True
        return enc


class L2Encoder(BaseLayerEncoder):
    """L2 encoder: frozen L1 encoder + linear projection to ``D2``.

    Holds a reference to an :class:`L1Encoder` instance; the L1 encoder's
    parameters are frozen on construction and only the linear projection
    to D2 is trainable.

    Args:
        l1_encoder: A frozen :class:`L1Encoder` instance.
        D2: L2 embedding dimension.
    """

    def __init__(self, l1_encoder: L1Encoder, D2: int = 32) -> None:
        super().__init__(lower_dim=l1_encoder.D, D=D2)
        self.l1_encoder = l1_encoder
        for p in self.l1_encoder.parameters():
            p.requires_grad = False
        self._freeze()

    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a 5D frame tensor to a sequence of L2 embeddings.

        Args:
            frames: ``(B, T, 3, H, W)`` batch of frames.
        Returns:
            ``(B, T, D2)`` L2 embeddings.
        """
        l1 = self.l1_encoder.encode_sequence(frames)
        return self.proj(l1)

    def _encode_lower(self, frames: torch.Tensor) -> torch.Tensor:
        return self.l1_encoder.encode_sequence(frames).mean(dim=1)

    @classmethod
    def from_l1_checkpoint(
        cls,
        l1_checkpoint: str,
        l0_model: LeWorldModel,
        D1: int = 64,
        D2: int = 32,
    ) -> "L2Encoder":
        """Build an :class:`L2Encoder` from a saved L1 checkpoint + L0 model.

        Loads the L1 encoder state from ``l1_checkpoint`` and stacks a
        fresh L2 projection on top.
        """
        l1 = L1Encoder.from_checkpoint(l1_checkpoint, l0_model, D1=D1)
        return cls(l1, D2=D2)


class L3Encoder(BaseLayerEncoder):
    """L3 encoder: frozen L2 encoder + linear projection to ``D3``.

    Args:
        l2_encoder: A frozen :class:`L2Encoder` instance.
        D3: L3 embedding dimension.
    """

    def __init__(self, l2_encoder: L2Encoder, D3: int = 16) -> None:
        super().__init__(lower_dim=l2_encoder.D, D=D3)
        self.l2_encoder = l2_encoder
        for p in self.l2_encoder.parameters():
            p.requires_grad = False
        self._freeze()

    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a 5D frame tensor to a sequence of L3 embeddings.

        Args:
            frames: ``(B, T, 3, H, W)`` batch of frames.
        Returns:
            ``(B, T, D3)`` L3 embeddings.
        """
        l2 = self.l2_encoder.encode_sequence(frames)
        return self.proj(l2)

    def _encode_lower(self, frames: torch.Tensor) -> torch.Tensor:
        return self.l2_encoder.encode_sequence(frames).mean(dim=1)

    @classmethod
    def from_l2_checkpoint(
        cls,
        l2_checkpoint: str,
        l0_model: LeWorldModel,
        D1: int = 64,
        D2: int = 32,
        D3: int = 16,
    ) -> "L3Encoder":
        """Build an :class:`L3Encoder` from a saved L2 checkpoint + L0 model."""
        l2 = L2Encoder.from_l1_checkpoint(l2_checkpoint, l0_model, D1=D1, D2=D2)
        return cls(l2, D3=D3)
