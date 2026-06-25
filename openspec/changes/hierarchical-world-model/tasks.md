## 1. Package scaffolding

- [x] 1.1 Create `src/wally/hierarchy/` package with `__init__.py`, `config.py`, `types.py`
- [x] 1.2 Define `LayerSpec` dataclass in `config.py` with fields: `name`, `K` (time horizon in frames), `D` (embedding dim), `depth` (transformer depth), `heads` (attention heads), `drift_epsilon` (per-layer drift threshold multiplier)
- [x] 1.3 Define `HierarchyConfig` dataclass with `layers: list[LayerSpec]`, `l0_checkpoint: str`, `lr: float`, `weight_decay: float`, `warmup_steps: int`, `max_steps: int`, `batch_size: int`, `alpha: float`
- [x] 1.4 Define `LayerState` and `LayerMessage` dataclasses in `types.py` for inter-layer communication (only `Tensor[D]` fields, no strings)

## 2. L1 world model

- [x] 2.1 Implement `JEPAWorldModel(nn.Module)` in `src/wally/hierarchy/jepa.py` with `predict(s_t, g) -> s_{t+K1}` signature, AdaLN-style transformer predictor, no pixel reconstruction
- [x] 2.2 Implement `L1Encoder(nn.Module)` that wraps the frozen L0 encoder weights + a learned linear projection to D1=64
- [x] 2.3 Implement the temporal-coherence training loss in `src/wally/hierarchy/loss.py`: L2 distance between predicted and actual L1 embeddings of states K1 frames apart, conditioned on the target embedding = actual future embedding
- [x] 2.4 Add `SIGReg` regularization term to the L1 loss (same formulation as L0, applied to the projected L1 embedding)
- [x] 2.5 Implement `wally-train-hierarchy --layer l1` that loads the L0 checkpoint, freezes it, trains L1 on `data/shards/treechop_full/`, saves `checkpoints/hierarchy_l1.pt`
- [x] 2.6 Add config `configs/hierarchy_l1.yaml` with K1=64, D1=64, depth=2, heads=4, lr=1e-4, max_steps=2000

## 3. L2 world model

- [x] 3.1 Implement `L2Encoder(nn.Module)` that uses the frozen L1 encoder + a learned linear projection to D2=32
- [x] 3.2 Reuse `JEPAWorldModel` with K2=512, D2=32, depth=2
- [x] 3.3 Extend `wally-train-hierarchy` to accept `--layer l2` (loads the L1 checkpoint, freezes it, trains L2 on top)
- [x] 3.4 Add config `configs/hierarchy_l2.yaml` with K2=512, D2=32, depth=2, heads=2, lr=5e-5, max_steps=2000

## 4. L3 world model

- [x] 4.1 Implement `L3Encoder(nn.Module)` that uses the frozen L2 encoder + a learned linear projection to D3=16
- [x] 4.2 Reuse `JEPAWorldModel` with K3=4096, D3=16, depth=2
- [x] 4.3 Extend `wally-train-hierarchy` to accept `--layer l3` (loads the L2 checkpoint, freezes it, trains L3 on top)
- [x] 4.4 Add config `configs/hierarchy_l3.yaml` with K3=4096, D3=16, depth=2, heads=2, lr=1e-5, max_steps=2000
- [x] 4.5 Implement learned goal embedding `g3` as a `nn.Parameter` of shape `(D3,)` per task, optimized end-to-end on the same loss as L3

## 5. Communication protocol

- [x] 5.1 Implement `MessageBus` in `src/wally/hierarchy/bus.py` with thread-safe async queues for up/down messages per layer
- [x] 5.2 Implement `LayerRuntime` in `src/wally/hierarchy/runtime.py` with `start()`, `stop()`, `tick(actual_s: Tensor) -> None`, and `latest_target: Tensor` properties
- [x] 5.3 Each `LayerRuntime` SHALL read from its input queue non-blockingly, update its `predicted_s`, and push the new `actual_s` up to the layer above
- [x] 5.4 Each `LayerRuntime` SHALL receive a new `target_embedding` from above, store it, and use it in the next predictor call
- [x] 5.5 Add a smoke test that wires up L0, L1, L2, L3 in a single process, feeds 100 fake state embeddings through, and verifies each layer's `latest_target` updates when its planner fires

## 6. Drift detection and replanning

- [x] 6.1 Implement `DriftMonitor` in `src/wally/hierarchy/drift.py` with `update(actual_s) -> drift_scalar` and `is_drifted() -> bool` based on the per-layer threshold
- [x] 6.2 Implement `Replanner` with three modes: `gentle_correct` (small drift, gradient-step the target embedding), `replan` (large drift, run CEM from scratch), `escalate` (no feasible target, signal the layer above)
- [x] 6.3 Wire `DriftMonitor` and `Replanner` into each `LayerRuntime`
- [x] 6.4 Log the per-layer drift distribution (p50, p90, p99) on a held-out trajectory at the end of `wally-train-hierarchy` and print to stdout
- [x] 6.5 Add a unit test that simulates a sequence of actual states with a known drift and verifies `gentle_correct` / `replan` / `escalate` fire at the right thresholds

## 7. Planner extensions

- [x] 7.1 Extend `CEMOptimizer.optimize` in `src/wally/planner/cem.py` with the `search_space='embedding'` mode (horizon=1, returns `Tensor[D]`)
- [x] 7.2 Add validation that `search_space='embedding'` requires `horizon=1` (raises `ValueError` otherwise)
- [x] 7.3 Extend `GoalConditionedPlanner.plan` in `src/wally/planner/plan.py` with the `target_embedding: torch.Tensor | None = None` argument
- [x] 7.4 Add validation that exactly one of `goal_frame` or `target_embedding` is provided (raises `ValueError` otherwise)
- [x] 7.5 When `target_embedding` is provided, the planner SHALL use it directly as `z_g` and skip the goal-frame encoding step
- [x] 7.6 Add a unit test for the new `plan` signature covering: goal_frame only, target_embedding only, both (raises), neither (raises)
- [x] 7.7 Add a unit test for `CEMOptimizer.optimize(search_space='embedding')` returning a single vector of the correct shape

## 8. Agent loop integration

- [x] 8.1 Implement `HierarchicalEmbeddingPlannerAdapter` in `src/wally/agent/planner_factory.py` that wraps a `HierarchicalEmbeddingPlanner` (which owns a stack of `LayerRuntime`s and a single `CEMOptimizer` for each layer)
- [x] 8.2 Extend `AgentLoop.run_episode` in `src/wally/agent/loop.py` to accept an optional `target_embedding: torch.Tensor | None = None` argument
- [x] 8.3 When a `HierarchicalEmbeddingPlannerAdapter` is in use, the agent loop SHALL push the new L0 state embedding to the L1 layer's input queue on every step
- [x] 8.4 When a `HierarchicalEmbeddingPlannerAdapter` is in use, the agent loop SHALL pull the latest L1 target embedding and pass it as `target_embedding` to the L0 planner on every replan
- [x] 8.5 Add a unit test that wires a fake hierarchy adapter into the agent loop and verifies the state-embedding push and target-embedding pull happen at the right times

## 9. CLI and configs

- [x] 9.1 Implement `wally-train-hierarchy` CLI in `src/wally/cli/train_hierarchy.py` with arguments: `--layer` (l1, l2, or l3), `--config` (path to hierarchy YAML), `--l0-checkpoint`, `--log-file`
- [x] 9.2 The CLI SHALL exit with a clear error if `torch.cuda.is_available()` is False
- [x] 9.3 The CLI SHALL log per-step `loss`, `lr`, `gpu_time`, `total_time` to `--log-file`
- [x] 9.4 The CLI SHALL save a checkpoint to `checkpoints/hierarchy_<layer>_<step>.pt` at the configured `checkpoint_interval`
- [x] 9.5 Extend `wally-play` CLI in `src/wally/agent/play.py` with `--target-embedding`, `--planner hierarchical-embedding`, `--layer-depth`, `--hierarchy-checkpoint` arguments
- [x] 9.6 Add configs `configs/hierarchy_l1.yaml`, `configs/hierarchy_l2.yaml`, `configs/hierarchy_l3.yaml` per the design document

## 10. Validation

- [x] 10.1 Add a unit test in `tests/test_hierarchy_smoke.py` that loads a frozen L0 checkpoint, trains L1 for 100 steps on a tiny synthetic dataset, and verifies the L1 embedding distance between two states from the same trajectory is smaller than between two random states
- [x] 10.2 Add a unit test for the drift detection logic with synthetic drift sequences
- [x] 10.3 Add a unit test for the embedding-mode CEM optimizer
- [x] 10.4 Add an integration test in `tests/test_hierarchy_integration.py` that wires L0+L1+L2 in-process, runs a 100-step fake episode, and verifies the hierarchy produces target embeddings that change over time
- [x] 10.5 Run the existing test suite (`pytest -m smoke -x`) and verify no regressions in the flat CEM planner path
- [x] 10.6 Document the new CLI in `docs/hierarchical-world-model.md` with: how to train each layer, how to run a hierarchical agent, the variable-depth semantics, the drift threshold tuning
