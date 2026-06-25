"""Hierarchical embedding planner — multi-layer goal-conditioned planner.

The :class:`HierarchicalEmbeddingPlanner` owns a stack of
:class:`LayerRuntime` instances (L1, L2, L3) and a single L0
:class:`GoalConditionedPlanner`. At every call to :meth:`plan`, the
planner:

1. Sets the L3 layer's target embedding (either from the user-supplied
   ``target_embedding`` argument or by running L3's CEM in embedding
   space).
2. Steps each layer's runtime once with the L0 state embedding
   (extracted from the current frame via the L0 encoder) so the
   streaming protocol is exercised.
3. Reads each layer's latest target embedding, projects it to the
   next-lower layer's space, and chains down to L0.
4. Calls the L0 planner with the L1's projected target embedding as
   ``target_embedding`` and returns the action sequence.

The runtime has full control of the per-layer replanning logic; the
planner itself is a thin coordinator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch

from wally.hierarchy.bus import MessageBus
from wally.hierarchy.config import LayerSpec
from wally.hierarchy.drift import DriftMonitor
from wally.hierarchy.jepa import JEPAWorldModel
from wally.hierarchy.runtime import LayerRuntime, LayerRuntimeCallbacks
from wally.planner.cem import CEMOptimizer
from wally.planner.config import CEMConfig
from wally.planner.plan import GoalConditionedPlanner


@dataclass
class HierarchyPlanResult:
    """The planner's output."""

    actions: torch.Tensor
    subgoals: dict[str, torch.Tensor]
    success: bool = True
    cost: float = 0.0
    replan_count: int = 0
    low_confidence: bool = False


class HierarchicalEmbeddingPlanner:
    """Multi-layer embedding planner that owns the runtime stack.

    Args:
        l0_planner: The :class:`GoalConditionedPlanner` for L0.
        l0_state_fn: Callable ``(current_frame) -> Tensor[D0]`` that
            produces the L0 state embedding from the current frame.
        layers: List of ``(name, spec, world_model, layer_above)``
            tuples — one per L_n layer. ``layer_above`` is the name of
            the layer above this one (``None`` for the topmost).
        goal_embedding: Optional learned goal tensor (shape ``(D_top,)``)
            for the topmost layer.
        cem_config: :class:`CEMConfig` used for the per-layer embedding
            CEM.
        device: Torch device.
    """

    def __init__(
        self,
        l0_planner: GoalConditionedPlanner,
        l0_state_fn: Callable[[torch.Tensor], torch.Tensor],
        layers: list[tuple[str, LayerSpec, JEPAWorldModel, str | None]],
        goal_embedding: torch.Tensor | None = None,
        cem_config: CEMConfig | None = None,
        device: str | torch.device = "cpu",
        l0_dim: int = 192,
        lowest_encoder: torch.nn.Module | None = None,
    ) -> None:
        self._l0_planner = l0_planner
        self._l0_state_fn = l0_state_fn
        self._device = torch.device(device)
        self._log = logging.getLogger("wally.hierarchy.planner")

        self._bus = MessageBus()
        self._runtimes: dict[str, LayerRuntime] = {}
        self._cem_per_layer: dict[str, CEMOptimizer] = {}
        cem_config = cem_config or CEMConfig.default()

        layer_names = [name for name, *_ in layers]
        for name in layer_names:
            self._bus.register(name)

        for name, spec, world_model, above in layers:
            drift = DriftMonitor(epsilon=spec.drift_epsilon, D=spec.D)
            cem = CEMOptimizer()
            self._cem_per_layer[name] = cem

            def make_planner(layer_name: str, layer_spec: LayerSpec) -> Callable:
                def planner(
                    actual_s: torch.Tensor,
                    prev_target: torch.Tensor | None,
                ) -> torch.Tensor | None:
                    return self._plan_layer_embedding(
                        layer_name, layer_spec, actual_s, prev_target
                    )
                return planner

            rt = LayerRuntime(
                name=name,
                spec=spec,
                world_model=world_model,
                drift_monitor=drift,
                planner=make_planner(name, spec),
                bus=self._bus,
                above_layer=above,
                callbacks=LayerRuntimeCallbacks(),
                device=self._device,
            )
            rt.start()
            rt.set_target(torch.zeros(spec.D, device=self._device))
            self._runtimes[name] = rt

        self._lowest_layer_encoder = lowest_encoder
        if lowest_encoder is not None:
            method = getattr(lowest_encoder, "encode_sequence", None)
            if method is not None:
                self._lowest_encoder_encode_sequence = method

        self._layer_order = list(reversed(layer_names))
        self._specs = {n: s for n, s, *_ in layers}

        lowest_layer_name = self._layer_order[0]
        self._lowest_layer_name = lowest_layer_name
        self._projection_l1_to_l0 = torch.nn.Linear(
            self._specs[lowest_layer_name].D, l0_dim, bias=False
        ).to(self._device)
        torch.nn.init.normal_(self._projection_l1_to_l0.weight, std=1e-2)

        if goal_embedding is not None:
            self._goal_embedding = (
                goal_embedding.detach().clone().to(self._device).flatten()
            )
        else:
            top = self._specs[layer_names[0]]
            self._goal_embedding = torch.zeros(top.D, device=self._device)

    @property
    def runtimes(self) -> dict[str, LayerRuntime]:
        return self._runtimes

    @property
    def bus(self) -> MessageBus:
        return self._bus

    @property
    def l1_to_l0_projection(self) -> torch.nn.Linear:
        return self._projection_l1_to_l0

    def set_goal(self, target: torch.Tensor) -> None:
        """Set the topmost layer's target embedding."""
        target = target.detach().to(self._device).flatten()
        top = self._layer_order[-1]
        self._runtimes[top].set_target(target)
        self._goal_embedding = target.clone()

    def latest_l1_target(self) -> torch.Tensor | None:
        return self._runtimes[self._lowest_layer_name].latest_target

    def push_l0_state(self, state_embedding: torch.Tensor) -> None:
        """Push a state embedding (in the lowest layer's space) upward.

        Called by the agent loop every step. The state must already be
        in the lowest layer's embedding space (e.g. L1 space for an
        L1+L2+L3 stack). Use :meth:`encode_for_lowest_layer` to convert
        a raw frame to the right space.
        """
        self._runtimes[self._lowest_layer_name].tick(state_embedding.to(self._device))

    def encode_for_lowest_layer(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode a frame through the lowest layer's encoder.

        Args:
            frame: ``(3, H, W)`` or ``(1, 3, H, W)`` raw frame.
        Returns:
            ``(D_lowest,)`` state embedding in the lowest layer's space.
        """
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        if frame.dim() == 4:
            frame = frame.unsqueeze(1)
        method = getattr(self, "_lowest_encoder_encode_sequence", None)
        if method is None:
            raise RuntimeError("lowest layer encoder not configured")
        return method(frame).squeeze(0).squeeze(0)

    def plan(
        self,
        current_frame: torch.Tensor,
        target_embedding: torch.Tensor | None = None,
    ) -> HierarchyPlanResult:
        """Run one planning cycle.

        Args:
            current_frame: ``(3, H, W)`` or ``(1, 3, H, W)`` L0 frame.
            target_embedding: Optional goal embedding for the topmost
                layer. ``None`` uses the planner's stored
                ``self._goal_embedding``.
        """
        if target_embedding is not None:
            self.set_goal(target_embedding)

        with torch.no_grad():
            if hasattr(self, "_lowest_encoder_encode_sequence"):
                lowest_state = self.encode_for_lowest_layer(
                    current_frame.to(self._device)
                )
            else:
                lowest_state = self._l0_state_fn(
                    current_frame.to(self._device)
                ).flatten()
        self.push_l0_state(lowest_state)

        l1_target = self.latest_l1_target()
        if l1_target is None:
            l1_target = torch.zeros(
                self._specs[self._lowest_layer_name].D, device=self._device
            )

        with torch.no_grad():
            l0_target = self._projection_l1_to_l0(l1_target.unsqueeze(0)).squeeze(0)
        actions, cost = self._l0_planner.plan(
            current_frame, target_embedding=l0_target, return_cost=True
        )

        subgoals = {
            name: (
                rt.latest_target.detach().clone()
                if rt.latest_target is not None
                else torch.zeros(rt.spec.D)
            )
            for name, rt in self._runtimes.items()
        }
        return HierarchyPlanResult(
            actions=actions,
            subgoals=subgoals,
            cost=float(cost),
            replan_count=sum(rt.replan_count for rt in self._runtimes.values()),
        )

    def _plan_layer_embedding(
        self,
        layer_name: str,
        layer_spec: LayerSpec,
        actual_s: torch.Tensor,
        prev_target: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Run the layer's embedding-CEM to find a new target embedding.

        The cost is the L2 distance between the layer's predicted
        ``s_{t+K}`` (conditioned on the candidate target) and the
        layer's *goal* (``prev_target`` — what came from above). The
        candidate is the embedding the layer will send down to the
        layer below. Minimising this cost makes the world model
        realise the goal, which is the whole point of the hierarchy.
        Falls back to ``actual_s`` as the goal when no target has been
        received yet (e.g. on the very first replan).
        """
        cem = self._cem_per_layer[layer_name]
        wm = self._runtimes[layer_name]._world_model
        goal = prev_target if prev_target is not None else actual_s

        def cost_fn(candidates: torch.Tensor) -> torch.Tensor:
            B = candidates.shape[0]
            actual_exp = actual_s.unsqueeze(0).expand(B, -1).to(candidates.device)
            goal_exp = goal.unsqueeze(0).expand(B, -1).to(candidates.device)
            with torch.no_grad():
                predicted = wm.predict(actual_exp, candidates)
            return ((predicted - goal_exp) ** 2).sum(dim=-1)

        new_target, _ = cem.optimize(
            cost_fn,
            horizon=1,
            action_dim=layer_spec.D,
            population_size=16,
            elite_frac=0.25,
            n_iterations=2,
            action_low=-1.0,
            action_high=1.0,
            init_mean=prev_target.unsqueeze(0) if prev_target is not None else None,
            device=self._device,
            search_space="embedding",
        )
        return new_target

    def drift_distributions(self) -> dict[str, dict[str, float]]:
        return {name: rt._drift.distribution() for name, rt in self._runtimes.items()}
