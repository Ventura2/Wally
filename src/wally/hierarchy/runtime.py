"""Layer runtime — the in-process loop for one hierarchy layer.

A :class:`LayerRuntime` ties together a layer's :class:`JEPAWorldModel`,
its drift monitor, and a planner. It exposes a single ``tick`` method
that the agent loop calls once per L0 step. The runtime handles the
streaming-embedding protocol with the layer above/below: read the
incoming state embedding, push the actual embedding up, and surface
from the background loop when drift exceeds the per-layer threshold.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable

import torch

from wally.hierarchy.bus import MessageBus
from wally.hierarchy.config import LayerSpec
from wally.hierarchy.drift import DriftMonitor, ReplanDecision
from wally.hierarchy.jepa import JEPAWorldModel
from wally.hierarchy.types import LayerMessage, LayerState


@dataclass
class LayerRuntimeCallbacks:
    """Optional callbacks a layer runtime fires on replan events.

    All callbacks default to no-ops; the agent loop wires in
    logging/metrics by replacing them.
    """

    on_gentle_correct: Callable[[torch.Tensor], None] = lambda g: None
    on_replan: Callable[[torch.Tensor], None] = lambda g: None
    on_escalate: Callable[[torch.Tensor], None] = lambda s: None


class LayerRuntime:
    """In-process per-layer loop.

    Args:
        name: Layer name (``"l1"``, ``"l2"``, ``"l3"``).
        spec: Layer hyperparameters.
        world_model: The layer's :class:`JEPAWorldModel`.
        drift_monitor: A :class:`DriftMonitor` configured with the per-layer
            threshold.
        planner: A callable that, given the current state embedding and
            the latest target, returns a new target embedding. It runs
            synchronously (the background loop blocks briefly when
            replanning).
        bus: The shared :class:`MessageBus`.
        above_layer: The name of the layer above this one (for
            escalations). ``None`` for the topmost layer (L3 in V1).
    """

    def __init__(
        self,
        name: str,
        spec: LayerSpec,
        world_model: JEPAWorldModel,
        drift_monitor: DriftMonitor,
        planner: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
        bus: MessageBus,
        above_layer: str | None = None,
        callbacks: LayerRuntimeCallbacks | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.name = name
        self.spec = spec
        self._world_model = world_model.to(device)
        self._world_model.eval()
        self._drift = drift_monitor
        self._planner = planner
        self._bus = bus
        self._above_layer = above_layer
        self._callbacks = callbacks or LayerRuntimeCallbacks()
        self._device = torch.device(device)

        self._state = LayerState()
        self._lock = threading.Lock()
        self._running = False
        self._log = logging.getLogger(f"wally.hierarchy.runtime.{name}")
        self._replan_count = 0
        self._escalation_count = 0
        self._gentle_correction_count = 0
        self._tick_count = 0

    @property
    def latest_target(self) -> torch.Tensor | None:
        with self._lock:
            return (
                self._state.target_embedding.clone()
                if self._state.target_embedding is not None
                else None
            )

    @property
    def latest_actual(self) -> torch.Tensor | None:
        with self._lock:
            return (
                self._state.actual_s.clone()
                if self._state.actual_s is not None
                else None
            )

    @property
    def latest_predicted(self) -> torch.Tensor | None:
        with self._lock:
            return (
                self._state.predicted_s.clone()
                if self._state.predicted_s is not None
                else None
            )

    @property
    def last_drift(self) -> float | None:
        with self._lock:
            return self._state.drift

    @property
    def replan_count(self) -> int:
        return self._replan_count

    @property
    def escalation_count(self) -> int:
        return self._escalation_count

    @property
    def gentle_correction_count(self) -> int:
        return self._gentle_correction_count

    def start(self) -> None:
        with self._lock:
            self._running = True

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def set_target(self, target: torch.Tensor) -> None:
        """Send a new target embedding from above (top-down)."""
        target = target.detach().to(self._device).flatten()
        with self._lock:
            self._state.target_embedding = target.clone()

    def tick(self, actual_s: torch.Tensor) -> None:
        """One step of the layer's background loop.

        Args:
            actual_s: The actual state embedding produced by the layer
                below. For L1 this is the L0 state embedding; for L2 the
                L1 state embedding; for L3 the L2 state embedding.
        """
        if not self._running:
            return
        actual_s = actual_s.detach().to(self._device).flatten()
        with torch.no_grad():
            with self._lock:
                prev_pred = self._state.predicted_s
                self._state.actual_s = actual_s.clone()
                if self._state.target_embedding is None:
                    return
                predicted = self._world_model.predict(
                    actual_s.unsqueeze(0),
                    self._state.target_embedding.unsqueeze(0),
                ).squeeze(0)
                self._state.predicted_s = predicted.clone()
                drift_val = (
                    (actual_s - predicted).norm().item()
                    if prev_pred is not None
                    else None
                )
                if drift_val is not None:
                    self._state.drift = drift_val

            self._bus.push_up(
                self.name,
                LayerMessage.from_state(actual_s, drift=drift_val),
            )

            if drift_val is None:
                self._log.debug(
                    "%s.tick: warmup (no prev_pred yet)", self.name
                )
                return
            self._tick_count += 1
            decision = self._drift.update(drift_val)
            if self._tick_count % 50 == 0 or decision is not None and decision is not ReplanDecision.NONE:
                self._log.info(
                    "%s tick=%d drift=%.4f replan=%d gentle=%d escalate=%d decision=%s",
                    self.name,
                    self._tick_count,
                    drift_val,
                    self._replan_count,
                    self._gentle_correction_count,
                    self._escalation_count,
                    decision.name if decision is not None else "NONE",
                )
            self._handle_decision(decision, actual_s)

    def _handle_decision(
        self, decision: ReplanDecision, actual_s: torch.Tensor
    ) -> None:
        if decision is None or decision is ReplanDecision.NONE:
            return
        if decision is ReplanDecision.GENTLE_CORRECT:
            self._do_gentle_correct(actual_s)
        elif decision is ReplanDecision.REPLAN:
            new_target = self._do_replan(actual_s)
            if new_target is None:
                self._do_escalate(actual_s)
        elif decision is ReplanDecision.ESCALATE:
            self._do_escalate(actual_s)

    def _do_gentle_correct(self, actual_s: torch.Tensor) -> None:
        with self._lock:
            current_target = (
                self._state.target_embedding.clone()
                if self._state.target_embedding is not None
                else None
            )
        if current_target is None:
            return
        new_target = self._planner(actual_s, current_target)
        new_target = new_target.detach().to(self._device).flatten()
        with self._lock:
            self._state.target_embedding = new_target.clone()
        self._bus.push_down(self.name, LayerMessage.from_target(new_target))
        self._gentle_correction_count += 1
        self._callbacks.on_gentle_correct(new_target)

    def _do_replan(self, actual_s: torch.Tensor) -> torch.Tensor | None:
        with self._lock:
            prev_target = (
                self._state.target_embedding.clone()
                if self._state.target_embedding is not None
                else None
            )
        new_target = self._planner(actual_s, prev_target)
        if new_target is None:
            return None
        new_target = new_target.detach().to(self._device).flatten()
        with self._lock:
            self._state.target_embedding = new_target.clone()
        self._bus.push_down(self.name, LayerMessage.from_target(new_target))
        self._replan_count += 1
        self._callbacks.on_replan(new_target)
        return new_target

    def _do_escalate(self, actual_s: torch.Tensor) -> None:
        self._escalation_count += 1
        self._callbacks.on_escalate(actual_s)
        if self._above_layer is not None:
            self._bus.push_up(
                self._above_layer,
                LayerMessage.from_state(actual_s, drift=float("inf")),
            )

    def state(self) -> LayerState:
        with self._lock:
            actual = (
                self._state.actual_s.clone()
                if self._state.actual_s is not None
                else None
            )
            predicted = (
                self._state.predicted_s.clone()
                if self._state.predicted_s is not None
                else None
            )
            target = (
                self._state.target_embedding.clone()
                if self._state.target_embedding is not None
                else None
            )
            return LayerState(
                actual_s=actual,
                predicted_s=predicted,
                target_embedding=target,
                drift=self._state.drift,
            )
