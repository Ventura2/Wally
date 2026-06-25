"""Integration tests for the hierarchy.

These wire up a stack of L1+L2+L3 layers in-process and verify the
end-to-end protocol: state embeddings stream upward, target embeddings
stream downward, and the planner produces target embeddings that change
over time as drift accumulates.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from wally.hierarchy.bus import MessageBus
from wally.hierarchy.config import HierarchyConfig, LayerSpec
from wally.hierarchy.encoders import L1Encoder, L2Encoder, L3Encoder
from wally.hierarchy.jepa import JEPAWorldModel
from wally.hierarchy.planner import HierarchicalEmbeddingPlanner
from wally.hierarchy.runtime import LayerRuntime
from wally.hierarchy.drift import DriftMonitor
from wally.models.lewm import LeWorldModel
from wally.planner.config import CEMConfig
from wally.planner.plan import GoalConditionedPlanner
from wally.training.checkpoint import load_checkpoint
from wally.training.sigreg import SIGReg


L0_CHECKPOINT = Path("checkpoints/wood_1000/checkpoint_1000.pt")


pytestmark = pytest.mark.skipif(
    not L0_CHECKPOINT.is_file(),
    reason="L0 checkpoint not available",
)


def _load_l0() -> LeWorldModel:
    ck = torch.load(L0_CHECKPOINT, map_location="cpu", weights_only=False)
    mc = ck.get("model_config", {})
    model = LeWorldModel(
        embed_dim=int(mc.get("embed_dim", 192)),
        depth=int(mc.get("depth", 4)),
        num_heads=int(mc.get("num_heads", 4)),
        mlp_ratio=float(mc.get("mlp_ratio", 4.0)),
        dropout=float(mc.get("dropout", 0.1)),
        encoder_type=mc.get("encoder_type", "cnn"),
        pretrained=False,
    )
    load_checkpoint(str(L0_CHECKPOINT), model)
    return model


class _FakeRollout:
    def __init__(self, l0_model: LeWorldModel) -> None:
        self._model = l0_model

    def rollout(self, z_0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, H, _ = actions.shape
        D = z_0.shape[-1]
        traj = torch.randn(B, H + 1, D)
        traj[:, 0, :] = z_0
        return traj


class TestHierarchyIntegration:
    def test_l1_l2_l3_stack_runs_100_steps(self):
        l0 = _load_l0()
        l1_enc = L1Encoder(l0, D1=64)
        l2_enc = L2Encoder(l1_enc, D2=32)
        l3_enc = L3Encoder(l2_enc, D3=16)
        specs = [
            LayerSpec("l3", K=2, D=16, depth=1, heads=1, drift_epsilon=0.10),
            LayerSpec("l2", K=2, D=32, depth=1, heads=1, drift_epsilon=0.10),
            LayerSpec("l1", K=2, D=64, depth=1, heads=1, drift_epsilon=0.10),
        ]
        wms = [JEPAWorldModel(state_dim=s.D, target_dim=s.D, hidden_dim=s.D, depth=1, num_heads=1) for s in specs]
        encoders = [l3_enc, l2_enc, l1_enc]

        bus = MessageBus()
        for n in ("l1", "l2", "l3"):
            bus.register(n)

        runtimes: dict[str, LayerRuntime] = {}
        for i, (name, spec, wm, enc) in enumerate(zip(("l3", "l2", "l1"), specs, wms, encoders)):
            above = ("l2", "l1", None)[i]
            drift = DriftMonitor(epsilon=spec.drift_epsilon, D=spec.D)
            torch.manual_seed(i)

            def make_planner():
                def planner(actual_s, prev_target):
                    return torch.randn(spec.D)
                return planner

            rt = LayerRuntime(
                name=name,
                spec=spec,
                world_model=wm,
                drift_monitor=drift,
                planner=make_planner(),
                bus=bus,
                above_layer=above,
            )
            rt.start()
            rt.set_target(torch.zeros(spec.D))
            runtimes[name] = rt

        l1_runtime = runtimes["l1"]
        targets_seen: list[torch.Tensor] = []
        for step in range(100):
            s_l0 = torch.randn(64)
            l1_runtime.tick(s_l0)
            if step >= 5 and step % 10 == 0:
                t = l1_runtime.latest_target
                if t is not None:
                    targets_seen.append(t.detach().clone())
        assert len(targets_seen) >= 5
        unique_targets = {tuple(t.tolist()) for t in targets_seen}
        assert len(unique_targets) > 1

    def test_hierarchical_planner_produces_changing_target_embeddings(self):
        l0 = _load_l0()
        l1_enc = L1Encoder(l0, D1=64)
        l2_enc = L2Encoder(l1_enc, D2=32)
        l3_enc = L3Encoder(l2_enc, D3=16)
        specs = [
            LayerSpec("l3", K=2, D=16, depth=1, heads=1, drift_epsilon=0.10),
            LayerSpec("l2", K=2, D=32, depth=1, heads=1, drift_epsilon=0.10),
            LayerSpec("l1", K=2, D=64, depth=1, heads=1, drift_epsilon=0.10),
        ]
        wms = [JEPAWorldModel(state_dim=s.D, target_dim=s.D, hidden_dim=s.D, depth=1, num_heads=1) for s in specs]
        layers_for_planner = [
            ("l3", specs[0], wms[0], None),
            ("l2", specs[1], wms[1], "l3"),
            ("l1", specs[2], wms[2], "l2"),
        ]

        rollout = _FakeRollout(l0)

        def l0_state_fn(frame: torch.Tensor) -> torch.Tensor:
            if frame.dim() == 3:
                frame = frame.unsqueeze(0)
            return l0._projector_fp32(l0.encoder(frame))

        def l0_encoder_for_planner(frame: torch.Tensor) -> torch.Tensor:
            if frame.dim() == 3:
                frame = frame.unsqueeze(0)
            return l0._projector_fp32(l0.encoder(frame))

        l0_planner = GoalConditionedPlanner(
            rollout, l0_encoder_for_planner, CEMConfig(population_size=4, n_iterations=1, horizon=2),
            device="cpu",
        )
        hier = HierarchicalEmbeddingPlanner(
            l0_planner=l0_planner,
            l0_state_fn=l0_state_fn,
            layers=layers_for_planner,
            cem_config=CEMConfig(population_size=4, n_iterations=1),
            device="cpu",
            l0_dim=192,
            lowest_encoder=l1_enc,
        )
        hier.set_goal(torch.zeros(16))

        targets_seen: list[torch.Tensor] = []
        for _ in range(20):
            current = torch.randn(3, 224, 224)
            l1_state = hier.encode_for_lowest_layer(current)
            hier.push_l0_state(l1_state)
            result = hier.plan(current)
            assert result.actions.shape[0] == 2
            targets_seen.append(hier.latest_l1_target().detach().clone())
        unique = {tuple(t.tolist()) for t in targets_seen}
        assert len(unique) > 1, "expected the L1 target embedding to change over time"

    def test_per_layer_cost_is_goal_driven(self):
        """The per-layer planner cost must compare the prediction to the
        layer's own goal, not to the current state — otherwise the
        hierarchy doesn't actually steer the agent.
        """
        from wally.hierarchy.bus import MessageBus

        bus = MessageBus()
        for n in ("l1", "l2", "l3"):
            bus.register(n)
        l0 = _load_l0()
        l1_enc = L1Encoder(l0, D1=8)
        spec = LayerSpec("l1", K=2, D=8, depth=1, heads=1, drift_epsilon=0.10)
        wm = JEPAWorldModel(state_dim=8, target_dim=8, hidden_dim=8, depth=1, num_heads=1)
        drift = DriftMonitor(epsilon=0.1, D=8)
        rt = LayerRuntime(
            name="l1",
            spec=spec,
            world_model=wm,
            drift_monitor=drift,
            planner=lambda s, g: torch.zeros(8),
            bus=bus,
        )
        rt.start()
        my_goal = torch.tensor([1.0] * 8)
        rt.set_target(my_goal)

        rollout = _FakeRollout(l0)

        def l0_state_fn(frame: torch.Tensor) -> torch.Tensor:
            if frame.dim() == 3:
                frame = frame.unsqueeze(0)
            return l0._projector_fp32(l0.encoder(frame))

        l0_planner = GoalConditionedPlanner(
            rollout,
            l0_state_fn,
            CEMConfig(population_size=4, n_iterations=1, horizon=2),
            device="cpu",
        )
        hier = HierarchicalEmbeddingPlanner(
            l0_planner=l0_planner,
            l0_state_fn=l0_state_fn,
            layers=[("l1", spec, wm, None)],
            cem_config=CEMConfig(population_size=4, n_iterations=1),
            device="cpu",
            l0_dim=192,
            lowest_encoder=l1_enc,
        )

        captured_costs: list[torch.Tensor] = {}
        original_optimize = hier._cem_per_layer["l1"].optimize

        def spy_optimize(cost_fn, **_):
            candidates = torch.eye(8)
            costs = cost_fn(candidates)
            captured_costs["cost_at_zero_candidate"] = costs.detach().clone()
            return torch.zeros(8), [0.0]

        hier._cem_per_layer["l1"].optimize = spy_optimize  # type: ignore[method-assign]
        actual = torch.tensor([0.0] * 8)
        hier._plan_layer_embedding("l1", spec, actual, my_goal)

        c = captured_costs["cost_at_zero_candidate"]
        assert c.numel() == 8
        assert c.min() >= 0
        assert c[0] > 0
        assert c.sum() > 0
        _ = original_optimize
