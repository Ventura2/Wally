from __future__ import annotations

import torch
from torch import nn


def pairwise_features(z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
    return torch.cat([z_i, z_j, z_i - z_j, (z_i - z_j).abs()], dim=-1)


class TRMHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        feat_dim = 4 * latent_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        feats = pairwise_features(z_i, z_j)
        return self.net(feats).squeeze(-1)


def hybrid_cost(
    lat_cost: torch.Tensor,
    trm_cost: torch.Tensor,
    lam: float = 0.5,
) -> torch.Tensor:
    a = lat_cost - lat_cost.mean()
    b = trm_cost - trm_cost.mean()
    a_std = a.std()
    b_std = b.std()
    a_term = a / (a_std + 1e-6) if a_std > 1e-9 else torch.zeros_like(a)
    b_term = (b / (b_std + 1e-6)) * lam if b_std > 1e-9 else torch.zeros_like(b)
    return a_term + b_term
