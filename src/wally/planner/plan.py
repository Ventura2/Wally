from __future__ import annotations

from typing import Callable

import torch

from wally.planner.cem import CEMOptimizer
from wally.planner.config import CEMConfig
from wally.planner.rollout import LatentRollout
from wally.planner.trm_head import TRMHead, hybrid_cost


def _default_cost(z_H: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
    """Cosine-distance cost: ignores latent magnitude, compares direction.

    The 1k-step L0's latent is dominated by frame brightness (PC1 ≈ 84% of
    variance; `‖z‖` correlates with brightness at +0.97 per the
    `tools/experiments/REPORT.md` 04_latent_geometry probe). The raw L2 cost
    `‖z_H − z_g‖²` therefore mostly measures a brightness mismatch, not a
    content mismatch — e.g. a dim cave and a dim tree score as "close" even
    if they share no content. Normalising both latents to unit length before
    the L2 makes the cost a *direction* comparison (bounded in [0, 4],
    equivalent to 2·(1 − cosine_similarity)). The brightness dim is still
    present, but the cost is no longer dominated by the magnitude
    difference; content dims (PC2+) get equal weight per dim.

    Note: the report's literal formula has a typo (`(z_H - z_g) / ‖z_H‖`
    instead of `z_H / ‖z_H‖`); this is the standard cosine distance form.
    """
    z_H_unit = z_H / (z_H.norm(dim=-1, keepdim=True) + 1e-6)
    z_g_unit = z_g / (z_g.norm(dim=-1, keepdim=True) + 1e-6)
    return ((z_H_unit - z_g_unit) ** 2).sum(dim=-1)


class GoalConditionedPlanner:
    def __init__(
        self,
        world_model: LatentRollout,
        encoder: Callable[[torch.Tensor], torch.Tensor],
        config: CEMConfig,
        *,
        device: torch.device | str | None = None,
        cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        action_dim: int = 25,
        on_replan: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None] | None = None,
        trm_head: TRMHead | None = None,
        trm_lambda: float = 0.5,
    ) -> None:
        self._world_model = world_model
        self._encoder = encoder
        self._config = config
        self._device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._cost_fn = cost_fn if cost_fn is not None else _default_cost
        self._action_dim = action_dim
        self._cem = CEMOptimizer()
        self._warm_start_mean: torch.Tensor | None = None
        self._on_replan = on_replan
        self._trm_head = trm_head
        self._trm_lambda = trm_lambda
        if trm_head is not None:
            trm_head.to(self._device)
            trm_head.eval()

    @property
    def encoder(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self._encoder

    @property
    def device(self) -> torch.device:
        return self._device

    def plan(
        self,
        current_frame: torch.Tensor,
        goal_frame: torch.Tensor | None = None,
        *,
        target_embedding: torch.Tensor | None = None,
        return_cost: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, float]:
        if (goal_frame is None) == (target_embedding is None):
            raise ValueError(
                "Exactly one of goal_frame or target_embedding must be provided"
            )

        if current_frame.dim() == 3:
            current_frame = current_frame.unsqueeze(0)
        squeeze = current_frame.shape[0] == 1
        current_frame = current_frame.to(self._device)

        captured: dict[str, torch.Tensor | None] = {"z_H": None, "costs": None}

        z_0 = self._encoder(current_frame).mean(dim=0, keepdim=True)

        if target_embedding is not None:
            if target_embedding.device != self._device:
                target_embedding = target_embedding.to(self._device)
            if target_embedding.dim() == 1:
                z_g = target_embedding.unsqueeze(0)
            else:
                z_g = target_embedding
        else:
            goal_frame = goal_frame.to(self._device)
            if goal_frame.dim() == 3:
                goal_frame = goal_frame.unsqueeze(0)
            z_g = self._encoder(goal_frame).mean(dim=0, keepdim=True)

        def cost_fn(actions: torch.Tensor) -> torch.Tensor:
            pop = actions.shape[0]
            z_0_exp = z_0.expand(pop, -1)
            z_g_exp = z_g.expand(pop, -1)
            trajectory = self._world_model.rollout(z_0_exp, actions)
            z_H = trajectory[:, -1, :]
            costs = self._regularized_cost(actions, z_H, z_g_exp)
            captured["z_H"] = z_H.detach()
            captured["costs"] = costs.detach()
            return costs

        actions, cost_history = self._cem.optimize(
            cost_fn,
            horizon=self._config.horizon,
            action_dim=self._action_dim,
            population_size=self._config.population_size,
            elite_frac=self._config.elite_frac,
            n_iterations=self._config.n_iterations,
            action_low=self._config.action_low,
            action_high=self._config.action_high,
            init_mean=self._warm_start_mean,
            device=self._device,
        )

        if squeeze:
            actions = actions.squeeze(0) if actions.dim() == 3 else actions

        if self._on_replan is not None and captured["z_H"] is not None:
            self._on_replan(captured["z_H"], captured["costs"], z_g.detach())

        if return_cost:
            return actions, cost_history[-1]
        return actions

    def set_warm_start_mean(self, mean: torch.Tensor) -> None:
        self._warm_start_mean = mean

    def plan_to_latent(
        self,
        current_frame: torch.Tensor,
        goal_latent: torch.Tensor,
        *,
        return_cost: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, float]:
        if current_frame.dim() == 3:
            current_frame = current_frame.unsqueeze(0)
        current_frame = current_frame.to(self._device)
        if goal_latent.device != self._device:
            goal_latent = goal_latent.to(self._device)

        z_0 = self._encoder(current_frame).mean(dim=0, keepdim=True)
        z_g = goal_latent.unsqueeze(0) if goal_latent.dim() == 1 else goal_latent

        def cost_fn(actions: torch.Tensor) -> torch.Tensor:
            pop = actions.shape[0]
            z_0_exp = z_0.expand(pop, -1)
            z_g_exp = z_g.expand(pop, -1)
            trajectory = self._world_model.rollout(z_0_exp, actions)
            z_H = trajectory[:, -1, :]
            return self._regularized_cost(actions, z_H, z_g_exp)

        actions, cost_history = self._cem.optimize(
            cost_fn,
            horizon=self._config.horizon,
            action_dim=self._action_dim,
            population_size=self._config.population_size,
            elite_frac=self._config.elite_frac,
            n_iterations=self._config.n_iterations,
            action_low=self._config.action_low,
            action_high=self._config.action_high,
            init_mean=self._warm_start_mean,
            device=self._device,
        )

        if actions.dim() == 3 and actions.shape[0] == 1:
            actions = actions.squeeze(0)

        if return_cost:
            return actions, cost_history[-1]
        return actions

    def _normalize_frames(
        self,
        current_frame: torch.Tensor,
        goal_frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        if current_frame.dim() == 3:
            current_frame = current_frame.unsqueeze(0)
        if goal_frame.dim() == 3:
            goal_frame = goal_frame.unsqueeze(0)
        squeeze = current_frame.shape[0] == 1
        return current_frame, goal_frame, squeeze

    def _regularized_cost(
        self,
        actions: torch.Tensor,
        z_H: torch.Tensor,
        z_g: torch.Tensor,
    ) -> torch.Tensor:
        base_cost = self._cost_fn(z_H, z_g)
        if self._trm_head is not None:
            with torch.no_grad():
                trm_cost = self._trm_head(z_H, z_g.expand_as(z_H))
            base_cost = hybrid_cost(base_cost, trm_cost, lam=self._trm_lambda)
        penalty = self._inventory_stall_penalty(actions)
        penalty = penalty + self._diversity_penalty(actions)
        penalty = penalty + self._camera_still_penalty(actions)
        return base_cost + penalty

    def _inventory_stall_penalty(self, actions: torch.Tensor) -> torch.Tensor:
        if self._config.inventory_stall_penalty <= 0.0 or actions.shape[-1] <= 12:
            return torch.zeros(actions.shape[0], device=actions.device)
        inventory_usage = actions[..., 12]
        return self._config.inventory_stall_penalty * inventory_usage.pow(2).sum(
            dim=-1
        )

    def _diversity_penalty(self, actions: torch.Tensor) -> torch.Tensor:
        """Reward candidates that diverge from the population mean.

        Breaks the "all CEM elites converge to the same low-cost action"
        local minimum (e.g. button-spam, no-op). Each candidate's cost is
        reduced by how far it sits from the population mean, so the
        optimizer prefers diverse action sequences over a single sharp
        optimum that the world model may have predicted wrongly.
        """
        if self._config.diversity_penalty <= 0.0 or actions.shape[0] < 2:
            return torch.zeros(actions.shape[0], device=actions.device)
        pop_mean = actions.mean(dim=0, keepdim=True)
        deviation_sq = (actions - pop_mean).pow(2).sum(dim=(-2, -1))
        return -self._config.diversity_penalty * deviation_sq

    def _camera_still_penalty(self, actions: torch.Tensor) -> torch.Tensor:
        """Penalize plans that keep the camera still (dims 0 and 1).

        The "do nothing" basin often has zero camera motion but
        button-spam on every other dim; a positive penalty on
        ``1 - |camera|`` forces the planner to commit to some camera
        movement, which makes the agent visibly turn the view.
        """
        if self._config.camera_still_penalty <= 0.0 or actions.shape[-1] < 2:
            return torch.zeros(actions.shape[0], device=actions.device)
        camera = actions[..., :2].clamp(-1.0, 1.0)
        # 1 - |camera| is large when camera is still, near zero when moving.
        # clamp inside, no negative penalty.
        still = (1.0 - camera.abs()).sum(dim=(-2, -1))
        return self._config.camera_still_penalty * still
