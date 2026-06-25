from __future__ import annotations

from typing import Callable, Literal

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
        device: torch.device | str | None = None,
        search_space: Literal["action", "embedding"] = "action",
    ) -> tuple[torch.Tensor, list[float]]:
        """Run the Cross-Entropy Method over a population of action sequences.

        When ``device`` is non-``None``, every candidate tensor (initial
        ``mean``/``std``, the sampled population, and the returned
        ``best_action_sequence``) lives on that device. When ``device`` is
        ``None``, candidates are created on CPU and the historical behaviour
        is preserved.

        When ``search_space='embedding'``, the search variable is a single
        continuous vector of shape ``(D,)`` (a target embedding) rather than
        an action sequence. ``horizon`` must be 1 in this mode; the returned
        ``best_sequence`` is a 1D tensor of shape ``(D,)`` instead of
        ``(H, A)``.
        """
        if search_space == "embedding":
            if horizon != 1:
                raise ValueError("search_space='embedding' requires horizon=1")
            return self._optimize_embedding(
                cost_fn,
                action_dim=action_dim,
                population_size=population_size,
                elite_frac=elite_frac,
                n_iterations=n_iterations,
                action_low=action_low,
                action_high=action_high,
                init_mean=init_mean,
                init_std=init_std,
                rng=rng,
                device=device,
            )

        elite_size = max(1, int(population_size * elite_frac))

        if init_mean is not None:
            mean = init_mean.clone()
        elif device is not None:
            mean = torch.zeros(horizon, action_dim, device=device)
        else:
            mean = torch.zeros(horizon, action_dim)

        if device is not None:
            std = torch.full((horizon, action_dim), init_std, device=device)
        else:
            std = torch.full((horizon, action_dim), init_std)
        cost_history: list[float] = []
        best_actions: torch.Tensor | None = None
        best_cost = float("inf")

        for _ in range(n_iterations):
            candidates = self._sample_truncated_normal(
                mean, std, population_size, action_low, action_high, rng, device,
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
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        shape = (population_size, *mean.shape)
        if device is not None:
            target = torch.device(device)
            if rng is not None and rng.device.type == target.type:
                samples = torch.randn(shape, generator=rng, device=device) * std + mean
            else:
                samples = torch.randn(shape, device=device) * std + mean
        else:
            samples = torch.randn(shape, generator=rng) * std + mean
        return samples.clamp(low, high)

    def _optimize_embedding(
        self,
        cost_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        action_dim: int,
        population_size: int,
        elite_frac: float,
        n_iterations: int,
        action_low: float,
        action_high: float,
        init_mean: torch.Tensor | None,
        init_std: float,
        rng: torch.Generator | None,
        device: torch.device | str | None,
    ) -> tuple[torch.Tensor, list[float]]:
        """CEM over a 1D ``(D,)`` candidate vector.

        Returns ``(Tensor[D], list[float])``.
        """
        elite_size = max(1, int(population_size * elite_frac))

        if init_mean is not None:
            if init_mean.dim() == 2 and init_mean.shape[0] == 1:
                mean = init_mean.squeeze(0).clone()
            else:
                mean = init_mean.clone()
        elif device is not None:
            mean = torch.zeros(action_dim, device=device)
        else:
            mean = torch.zeros(action_dim)

        if device is not None:
            std = torch.full((action_dim,), init_std, device=device)
        else:
            std = torch.full((action_dim,), init_std)
        cost_history: list[float] = []
        best_vector: torch.Tensor | None = None
        best_cost = float("inf")

        for _ in range(n_iterations):
            candidates = self._sample_truncated_normal(
                mean, std, population_size, action_low, action_high, rng, device,
            )
            costs = cost_fn(candidates)
            iter_best_idx = costs.argmin().item()
            iter_best_cost = costs[iter_best_idx].item()
            cost_history.append(iter_best_cost)

            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_vector = candidates[iter_best_idx].clone()

            _, elite_idx = torch.topk(costs, elite_size, largest=False)
            elites = candidates[elite_idx]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0, unbiased=False).clamp(min=1e-6)

        assert best_vector is not None
        return best_vector, cost_history


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
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, list[float]]:
        """Sample a population of action sequences uniformly and pick the best.

        When ``device`` is non-``None``, the candidate tensor is created on
        that device. When ``device`` is ``None``, the historical CPU-only
        behaviour is preserved.
        """
        shape = (population_size, horizon, action_dim)
        mid = (action_low + action_high) / 2.0
        scale = (action_high - action_low) / 2.0
        if device is not None:
            target = torch.device(device)
            if rng is not None and rng.device.type == target.type:
                samples = torch.randn(shape, generator=rng, device=device) * scale + mid
            else:
                samples = torch.randn(shape, device=device) * scale + mid
        else:
            samples = torch.randn(shape, generator=rng) * scale + mid
        samples = samples.clamp(action_low, action_high)

        costs = cost_fn(samples)
        best_idx = costs.argmin().item()
        best_cost = costs[best_idx].item()
        return samples[best_idx].clone(), [best_cost]
