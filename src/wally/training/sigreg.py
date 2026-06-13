from __future__ import annotations

import torch
from torch import Tensor, nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularization (Epps-Pulley statistic).

    Computes a non-negative statistic measuring deviation of an embedding
    distribution from an isotropic Gaussian. The statistic is estimated by
    applying random unit-norm projections to the input and comparing the
    empirical characteristic function of the projected scalars against the
    Gaussian target phi(t) = exp(-t^2 / 2).

    Ported from lucas-maes/le-wm (module.py:8-37). The module exposes no
    learnable parameters: the projection matrix is resampled on every call,
    so gradients flow through the encoder embeddings but never into the
    projection itself.

    The output is non-negative and finite for any finite input embedding,
    including degenerate cases (all-zeros, constant vectors).
    """

    t: Tensor
    phi: Tensor
    weights: Tensor

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.knots = knots
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: Tensor) -> Tensor:
        """Compute the Epps-Pulley statistic on input embeddings.

        Args:
            proj: Tensor of shape (T, B, D) (time, batch, dimension) per the
                le-wm convention.

        Returns:
            Scalar non-negative tensor measuring deviation from an isotropic
            Gaussian.
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
