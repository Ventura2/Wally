from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-Zero modulation: x * (1 + scale) + shift.

    ``shift``/``scale`` may be 2D ``(B, dim)`` (per-batch) or 3D ``(B, T, dim)``
    (per-frame). 2D inputs are broadcast over the sequence dim.
    """
    if shift.dim() == 2 and x.dim() == 3:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1.0 + scale) + shift


class FeedForward(nn.Module):
    """Pre-LN GELU feed-forward block (LeWM official)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # type: ignore[no-any-return]


class Attention(nn.Module):
    """Multi-head causal self-attention with fused SDPA (LeWM official)."""

    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv
        )
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)  # type: ignore[no-any-return]


class Block(nn.Module):
    """Pre-LN Transformer block without conditioning (LeWM official).

    No extra ``norm1``/``norm2`` here on purpose: ``Attention`` and
    ``FeedForward`` each apply their own internal pre-LN, so adding a
    second pair would double-normalize. Only ``ConditionalBlock`` adds
    the AdaLN pre-norms (for the scale/shift modulation).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=num_heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class ConditionalBlock(nn.Module):
    """AdaLN-Zero conditioned Transformer block (LeWM official, wally LN fix)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        c_dim: int | None = None,
        dropout: float = 0.0,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        if c_dim is None:
            c_dim = dim
        self.attn = Attention(dim, heads=num_heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Linear(c_dim, 6 * dim, bias=True)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Accept per-batch (B, c_dim) or per-frame (B, T, c_dim) conditioning.
        if c.dim() == 2:
            c = c.unsqueeze(1).expand(-1, x.size(1), -1)
        chunks = self.modulation(F.silu(c)).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class Transformer(nn.Module):
    """Stack of Block/ConditionalBlock with optional input/cond/output projections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
        dim_head: int = 64,
        block_class: type[nn.Module] = Block,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(
                    hidden_dim, num_heads, mlp_ratio, dropout=dropout, dim_head=dim_head
                )
            )

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if hasattr(self, "input_proj"):
            x = self.input_proj(x)
        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)
        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x
