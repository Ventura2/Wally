"""Tests for the hierarchy layer runtime, message bus, and drift logic."""

from __future__ import annotations

import torch

from wally.hierarchy.bus import MessageBus
from wally.hierarchy.config import LayerSpec
from wally.hierarchy.drift import DriftMonitor, ReplanDecision
from wally.hierarchy.jepa import JEPAWorldModel
from wally.hierarchy.runtime import LayerRuntime
from wally.hierarchy.types import LayerMessage


def _make_rt(
    name: str,
    D: int = 8,
    above: str | None = None,
    drift_epsilon: float = 0.1,
) -> tuple[LayerRuntime, MessageBus, dict[str, int]]:
    bus = MessageBus()
    bus.register(name)
    if above is not None:
        bus.register(above)
    spec = LayerSpec(name, K=2, D=D, depth=1, heads=1, drift_epsilon=drift_epsilon)
    wm = JEPAWorldModel(state_dim=D, target_dim=D, hidden_dim=D, depth=1, num_heads=1)
    dm = DriftMonitor(epsilon=drift_epsilon, D=D)
    counts = {"n": 0}

    def planner(actual_s, prev_target):
        counts["n"] += 1
        return torch.zeros(D)

    rt = LayerRuntime(
        name=name,
        spec=spec,
        world_model=wm,
        drift_monitor=dm,
        planner=planner,
        bus=bus,
        above_layer=above,
    )
    rt.start()
    rt.set_target(torch.zeros(D))
    return rt, bus, counts


class TestMessageBus:
    def test_register_and_push_pop(self):
        bus = MessageBus()
        bus.register("l1")
        bus.push_down("l1", LayerMessage.from_target(torch.zeros(3)))
        msg = bus.pop_down("l1")
        assert msg is not None
        assert msg.target_embedding is not None
        assert bus.pop_down("l1") is None

    def test_bounded_queue_drops_oldest(self):
        bus = MessageBus(maxlen=2)
        bus.register("l1")
        bus.push_down("l1", LayerMessage.from_target(torch.tensor([1.0])))
        bus.push_down("l1", LayerMessage.from_target(torch.tensor([2.0])))
        bus.push_down("l1", LayerMessage.from_target(torch.tensor([3.0])))
        msgs = bus.drain_down("l1")
        assert len(msgs) == 2
        assert torch.allclose(msgs[-1].target_embedding, torch.tensor([3.0]))

    def test_unregistered_layer_raises(self):
        bus = MessageBus()
        with __import__("pytest").raises(ValueError):
            bus.push_down("l1", LayerMessage.from_target(torch.zeros(1)))


class TestDriftMonitor:
    def test_below_threshold_returns_none(self):
        dm = DriftMonitor(epsilon=0.1, D=16)
        for _ in range(5):
            assert dm.update(0.1) is ReplanDecision.NONE
        assert dm.triggered == 0

    def test_gentle_correct_zone(self):
        dm = DriftMonitor(epsilon=0.1, D=16)
        for _ in range(5):
            assert dm.update(0.5) is ReplanDecision.GENTLE_CORRECT
        assert dm.triggered == 5

    def test_replan_zone(self):
        dm = DriftMonitor(epsilon=0.1, D=16)
        for _ in range(5):
            assert dm.update(1.5) is ReplanDecision.REPLAN
        assert dm.triggered == 5

    def test_distribution_percentiles(self):
        dm = DriftMonitor(epsilon=0.1, D=16)
        for v in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            dm.update(v)
        d = dm.distribution()
        assert d["count"] == 6
        assert d["p50"] >= 0.5
        assert d["p90"] >= 2.0


class TestLayerRuntime:
    def test_first_tick_records_state_but_skips_drift(self):
        rt, bus, counts = _make_rt("l1")
        s = torch.randn(8)
        rt.tick(s)
        assert rt.last_drift is None
        assert rt.replan_count == 0
        assert bus.latest_up("l1") is not None

    def test_subsequent_ticks_trigger_replan(self):
        rt, bus, counts = _make_rt("l1")
        for i in range(5):
            s = torch.randn(8) * 5.0
            rt.tick(s)
        assert rt.replan_count >= 1
        assert counts["n"] >= 1

    def test_set_target_updates_latest_target(self):
        rt, bus, _ = _make_rt("l1")
        rt.set_target(torch.ones(8))
        assert rt.latest_target is not None
        assert torch.allclose(rt.latest_target, torch.ones(8))

    def test_escalation_signal_on_replan_failure(self):
        rt, bus, _ = _make_rt("l1", above="l2")

        def failing_planner(actual_s, prev_target):
            return None

        rt._planner = failing_planner
        for _ in range(3):
            rt.tick(torch.randn(8) * 5.0)
        assert rt.escalation_count >= 1
        assert bus.latest_up("l2") is not None

    def test_drift_distribution_history(self):
        rt, bus, _ = _make_rt("l1")
        for _ in range(10):
            rt.tick(torch.randn(8) * 5.0)
        d = rt._drift.distribution()
        assert d["count"] >= 9
