from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SIGRegCritic(nn.Module):
    """MLP critic for SIGReg mutual information estimation.

    Takes concatenated (predicted, target) latents and outputs a scalar score.
    Architecture: embed_dim*2 → hidden_dim → hidden_dim → 1
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            predicted: (..., embed_dim)
            target:    (..., embed_dim)

        Returns:
            Scalar scores (..., 1)
        """
        x = torch.cat([predicted, target], dim=-1)
        return self.net(x)


def sigreg_loss(
    critic: SIGRegCritic,
    predicted: Tensor,
    target: Tensor,
) -> Tensor:
    """SIGReg loss for mutual information estimation.

    Estimates MI between predicted and target latents. The loss is formulated
    so that minimizing it encourages the model to produce predictions that
    share high mutual information with targets.

    Joint score: critic on real (predicted, target) pairs
    Marginal score: critic on (predicted, shuffled_target) pairs
    loss = mean(marginal_scores) - mean(joint_scores)

    Args:
        critic:    SIGRegCritic network
        predicted: (B, T-1, embed_dim)
        target:    (B, T-1, embed_dim)

    Returns:
        Scalar SIGReg loss.
    """
    # joint: real pairs
    joint_scores = critic(predicted, target)

    # marginal: shuffle target across batch dimension (dim=0)
    perm = torch.randperm(target.size(0), device=target.device)
    shuffled_target = target[perm]
    marginal_scores = critic(predicted, shuffled_target)

    # loss = mean(marginal) - mean(joint)
    # minimizing this encourages high MI (large joint - marginal)
    return marginal_scores.mean() - joint_scores.mean()
