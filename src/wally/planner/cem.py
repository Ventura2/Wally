from __future__ import annotations

from typing import Callable

import torch


class CEMOptimizer:
    def optimize(
        self,
        cost_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        horizon: int,
        action_dim: int,
        population_size: int = 64,
        elite_frac: float = 0.1,
        n_iterations: int = 5,
        action_low: float = -1.0,
        action_high: float = 1.0,
        init_mean: torch.Tensor | None = None,
        init_std: float = 1.0,
        rng: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, list[float]]:
        elite_size = max(1, int(population_size * elite_frac))

        if init_mean is not None:
            mean = init_mean.clone()
        else:
            mean = torch.zeros(horizon, action_dim)

        std = torch.full((horizon, action_dim), init_std)
        cost_history: list[float] = []
        best_actions: torch.Tensor | None = None
        best_cost = float("inf")

        for _ in range(n_iterations):
            candidates = self._sample_truncated_normal(
                mean, std, population_size, action_low, action_high, rng,
            )
            costs = cost_fn(candidates)
            iter_best_idx = costs.argmin().item()
            iter_best_cost = costs[iter_best_idx].item()
            cost_history.append(iter_best_cost)

            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_actions = candidates[iter_best_idx].clone()

            _, elite_idx = torch.topk(costs, elite_size, largest=False)
            elites = candidates[elite_idx]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0, unbiased=False).clamp(min=1e-6)

        assert best_actions is not None
        return best_actions, cost_history

    def _sample_truncated_normal(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        population_size: int,
        low: float,
        high: float,
        rng: torch.Generator | None,
    ) -> torch.Tensor:
        shape = (population_size, *mean.shape)
        samples = torch.randn(shape, generator=rng) * std + mean
        return samples.clamp(low, high)


class RandomShooting:
    def optimize(
        self,
        cost_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        horizon: int,
        action_dim: int,
        population_size: int = 64,
        action_low: float = -1.0,
        action_high: float = 1.0,
        rng: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, list[float]]:
        shape = (population_size, horizon, action_dim)
        mid = (action_low + action_high) / 2.0
        scale = (action_high - action_low) / 2.0
        samples = torch.randn(shape, generator=rng) * scale + mid
        samples = samples.clamp(action_low, action_high)

        costs = cost_fn(samples)
        best_idx = costs.argmin().item()
        best_cost = costs[best_idx].item()
        return samples[best_idx].clone(), [best_cost]
