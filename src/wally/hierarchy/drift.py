"""Drift detection and replanning logic for the hierarchy layers.

The drift is the L2 distance between the layer's predicted state
embedding and the actual state embedding it receives from the layer
below. When the drift exceeds the per-layer threshold (a function of
``epsilon * sqrt(D)``) the layer surfaces from its background loop and
decides between three actions:

- ``gentle_correct`` — small overshoot; nudge the current target
  embedding via gradient descent on the embedding-distance cost.
- ``replan`` — large overshoot; throw the current plan away and run
  the planner from scratch.
- ``escalate`` — no feasible target within the budget; signal the
  layer above.
"""

from __future__ import annotations

import math
from collections import deque
from enum import Enum
from typing import Iterable

import torch


class ReplanDecision(str, Enum):
    NONE = "none"
    GENTLE_CORRECT = "gentle_correct"
    REPLAN = "replan"
    ESCALATE = "escalate"


class DriftMonitor:
    """Tracks drift values and triggers a :class:`ReplanDecision` on exceed.

    Args:
        epsilon: Per-layer multiplier. The actual threshold is
            ``epsilon * sqrt(D)`` (matches the design document).
        D: Embedding dimension.
        gentle_overshoot: Multiple of the threshold above which to
            trigger ``gentle_correct`` rather than ``replan``.
        replan_overshoot: Multiple of the threshold above which to
            trigger ``replan`` rather than ``escalate``.
        history_size: How many recent drift values to keep for the
            p50/p90/p99 distribution summary.
    """

    def __init__(
        self,
        epsilon: float,
        D: int,
        *,
        gentle_overshoot: float = 1.0,
        replan_overshoot: float = 2.0,
        history_size: int = 1024,
    ) -> None:
        if epsilon < 0.0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}")
        if D < 1:
            raise ValueError(f"D must be >= 1, got {D}")
        if gentle_overshoot < 0.0:
            raise ValueError("gentle_overshoot must be >= 0")
        if replan_overshoot < gentle_overshoot:
            raise ValueError("replan_overshoot must be >= gentle_overshoot")
        self.epsilon = epsilon
        self.D = D
        self.gentle_overshoot = gentle_overshoot
        self.replan_overshoot = replan_overshoot
        self.threshold = epsilon * math.sqrt(D)
        self._history: deque[float] = deque(maxlen=history_size)
        self._total = 0
        self._triggered = 0

    @property
    def triggered(self) -> int:
        return self._triggered

    def update(self, drift: float) -> ReplanDecision:
        self._history.append(float(drift))
        self._total += 1
        if drift <= self.threshold:
            return ReplanDecision.NONE
        self._triggered += 1
        if drift <= self.threshold * self.replan_overshoot:
            return ReplanDecision.GENTLE_CORRECT
        return ReplanDecision.REPLAN

    def is_drifted(self, drift: float) -> bool:
        return drift > self.threshold

    def distribution(self) -> dict[str, float]:
        if not self._history:
            return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0, "count": 0}
        sorted_h = sorted(self._history)
        n = len(sorted_h)

        def pct(p: float) -> float:
            idx = min(n - 1, int(p * n))
            return sorted_h[idx]

        return {
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p99": pct(0.99),
            "mean": sum(sorted_h) / n,
            "count": self._total,
        }


class Replanner:
    """Decides which replan mode to use and runs the appropriate action.

    Args:
        world_model: The layer's :class:`JEPAWorldModel`. Used to compute
            gradient steps for ``gentle_correct``.
        drift_monitor: The :class:`DriftMonitor` for this layer.
        planner: Callable ``(actual_s, prev_target) -> new_target``. Used
            for full replans.
        gentle_step_size: Step size for gradient-based gentle correction.
        max_gentle_steps: Maximum number of gradient steps in a single
            gentle-correct call.
        replan_budget: Number of planner calls allowed in a single
            ``replan`` before falling back to ``escalate``.
    """

    def __init__(
        self,
        world_model: torch.nn.Module,
        drift_monitor: DriftMonitor,
        planner: "callable",
        *,
        gentle_step_size: float = 1e-2,
        max_gentle_steps: int = 5,
        replan_budget: int = 1,
    ) -> None:
        self._world_model = world_model
        self._drift = drift_monitor
        self._planner = planner
        self.gentle_step_size = gentle_step_size
        self.max_gentle_steps = max_gentle_steps
        self.replan_budget = replan_budget

    def gentle_correct(
        self,
        actual_s: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Nudge ``target`` toward ``actual_s`` via gradient descent on
        ``|| world_model.predict(actual_s, target) - actual_s ||^2``.
        """
        target = target.detach().clone().requires_grad_(True)
        last_target = target.detach().clone()
        for _ in range(self.max_gentle_steps):
            pred = self._world_model.predict(actual_s.unsqueeze(0), target.unsqueeze(0))
            loss = ((pred - actual_s.unsqueeze(0)) ** 2).sum()
            if target.grad is not None:
                target.grad.zero_()
            loss.backward()
            with torch.no_grad():
                target -= self.gentle_step_size * target.grad
            last_target = target.detach().clone()
        return last_target

    def replan(
        self,
        actual_s: torch.Tensor,
        prev_target: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Run the planner from scratch. Returns ``None`` if it cannot find
        a feasible target within the budget.
        """
        for _ in range(self.replan_budget):
            new_target = self._planner(actual_s, prev_target)
            if new_target is not None:
                return new_target
        return None

    def decide_and_act(
        self,
        drift: float,
        actual_s: torch.Tensor,
        prev_target: torch.Tensor | None,
    ) -> tuple[ReplanDecision, torch.Tensor | None]:
        """Single entry point used by :class:`LayerRuntime`."""
        decision = self._drift.update(drift)
        if decision is ReplanDecision.NONE:
            return decision, None
        if decision is ReplanDecision.GENTLE_CORRECT and prev_target is not None:
            return decision, self.gentle_correct(actual_s, prev_target)
        if decision is ReplanDecision.REPLAN:
            new = self.replan(actual_s, prev_target)
            if new is None:
                return ReplanDecision.ESCALATE, None
            return decision, new
        return decision, None


def summarise_distributions(
    monitors: Iterable[DriftMonitor],
) -> dict[str, dict[str, float]]:
    """Pretty-print the drift distribution for a collection of monitors."""
    return {f"layer_{i}": m.distribution() for i, m in enumerate(monitors)}
