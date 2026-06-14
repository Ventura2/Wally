from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from torch.amp import custom_fwd  # type: ignore[attr-defined]

from wally.models.cnn_encoder import SimpleCNNEncoder
from wally.models.embedder import Embedder
from wally.models.encoder import ViTEncoder
from wally.models.mlp import MLP
from wally.models.predictor import ARPredictor


class LeWorldModel(nn.Module):
    """Latent Embedding World Model — encoder + projector + AdaLN predictor + pred_proj.

    Data flow:
        frames → encoder → projector → emb (B, T, hidden_dim)
        actions → action_embedder → act_emb (B, T-1, c_dim)
        pred_emb = predictor(emb[:, :-1], act_emb) → (B, T-1, hidden_dim)
        predicted_change = pred_proj(pred_emb) → (B, T-1, output_dim)

    The first returned tensor is the **predicted change** Δ — the frame-to-frame
    delta in latent space. The next-frame latent is reconstructed by the loss
    as ``projected_embeddings[:, :-1] + predicted_change``. This is the
    LeWorldModel paper formulation (Algorithm 1, line 303:
    ``pred_loss = F.mse_loss(emb[:, 1:] - next_emb[:, :-1])``).

    SIGReg is applied to ``emb`` (the projected encoder output), matching
    the official LeWM paper.
    """

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
        encoder_type: str = "vit",
        num_frames: int = 16,
    ) -> None:
        super().__init__()
        if encoder_type == "cnn":
            self.encoder: nn.Module = SimpleCNNEncoder(embed_dim=embed_dim)
            self._is_cnn = True
        else:
            self.encoder = ViTEncoder(variant=vit_variant, pretrained=pretrained)
            self._is_cnn = False

        # Projector: encoder output → predictor hidden dim
        # Uses BatchNorm1d in fp32 (matches official LeWM stability pattern).
        self.projector = MLP(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            norm_fn=nn.BatchNorm1d,
            act_fn=nn.GELU,
        )

        # Action embedder: official Conv1d + 2-layer MLP with SiLU
        self.action_embedder = Embedder(
            input_dim=action_dim,
            smoothed_dim=embed_dim,
            emb_dim=embed_dim,
            mlp_scale=4,
        )

        # AdaLN-Zero causal Transformer predictor
        self.predictor = ARPredictor(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            c_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_frames=num_frames,
        )

        # pred_proj: predictor output → target dim (same as input dim here)
        self.pred_proj = MLP(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            norm_fn=nn.BatchNorm1d,
            act_fn=nn.GELU,
        )

    @custom_fwd(device_type="cuda", cast_inputs=torch.float32)  # type: ignore[untyped-decorator]
    def _projector_fp32(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)  # type: ignore[no-any-return]

    def forward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        return_embeddings: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        Args:
            frames:  (B, T, 3, 224, 224)
            actions: (B, T, action_dim)
            return_embeddings: If True, also return the projected encoder
                embeddings (the SIGReg input), transposed to (T, B, D).

        Returns:
            predicted_change: (B, T-1, embed_dim) — the per-step change in
                latent space (the predictor's output). The next-frame latent
                is reconstructed as ``projected_embeddings[:, :-1] + predicted_change``.
            embeddings: (T, B, embed_dim) — only if return_embeddings=True;
                the SIGReg input in time-first shape.
        """
        B, T, C, H, W = frames.shape

        # encode all frames → (B, T, embed_dim)
        flat = rearrange(frames, "b t c h w -> (b t) c h w")
        if self._is_cnn:
            latents_flat = self.encoder(flat)
        else:
            tokens = self.encoder(flat)
            latents_flat = tokens.mean(dim=1)
        latents = rearrange(latents_flat, "(b t) d -> b t d", b=B, t=T)

        # project latents through the projector (BatchNorm1d in fp32)
        emb = self._projector_fp32(latents)

        # predictor input + conditioning
        current_emb = emb[:, :-1]                       # (B, T-1, embed_dim)
        act_emb = self.action_embedder(actions[:, :-1])  # (B, T-1, embed_dim)

        # AdaLN-Zero conditioned causal prediction — output is the per-step change
        pred_emb = self.predictor(current_emb, act_emb)  # (B, T-1, embed_dim)
        predicted_change = self.pred_proj(pred_emb)

        if return_embeddings:
            return (
                predicted_change,
                emb.transpose(0, 1).contiguous(),
            )
        return (predicted_change,)
