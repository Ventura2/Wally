# Hierarchical world model

Wally's L0 LeWorldModel predicts frames 8 steps ahead and plans at the
motor level. The hierarchy stacks three more JEPA world models on top
(L1 → L2 → L3) that think at longer time horizons, in their own learned
embedding spaces. The stack is purely additive: the existing
``wally-train``, ``wally-play --planner cem``, and ``wally-deploy``
commands keep working unchanged.

## What lives in the stack

| Layer | Time horizon | Embedding dim | Conditioning |
|-------|--------------|---------------|--------------|
| L0    | 8 frames     | 192           | action sequence (LeWM AdaLN) |
| L1    | 64 frames    | 64            | target embedding `g1`        |
| L2    | 512 frames   | 32            | target embedding `g2`        |
| L3    | 4096 frames  | 16            | target embedding `g3`        |
| L4    | session      | 16            | objective text → `g4` (V2)   |

Each upper layer is a `JEPAWorldModel` (`src/wally/hierarchy/jepa.py`)
— an AdaLN-conditioned Transformer that predicts the L_n-embedding of a
state K_n frames in the future. L1+ do **not** reconstruct pixels; the
loss is pure L2 on the predicted embedding plus a SIGReg term on the
projected embedding (matches the L0 design).

## How layers communicate

All inter-layer traffic is continuous: only `Tensor[D]` fields, no
strings, no symbolic task names, no discrete skill IDs
(`src/wally/hierarchy/types.py`).

- **Bottom-up**: each layer emits a `state_embedding` (its most recent
  actual embedding) to the layer above, plus a `drift` scalar.
- **Top-down**: each layer receives a `target_embedding` from above and
  uses it as the conditioning for the next predictor call.

A `MessageBus` (`src/wally/hierarchy/bus.py`) holds bounded queues
between layers, and `LayerRuntime` (`src/wally/hierarchy/runtime.py`)
runs the per-layer background loop that:
1. Computes `predicted_s = world_model.predict(actual_s, g)`.
2. Computes `drift = || actual_s − predicted_s ||`.
3. If `drift > threshold`, surfaces and either gently corrects the
   target, runs a full replan, or escalates to the layer above.

## Variable depth at runtime

Tasks pick the depth at runtime, not at architecture build time:

| Task             | Default depth | Activated layers |
|------------------|---------------|------------------|
| `move_forward`   | 0             | L0 only          |
| `get_wood`       | 2             | L0 + L1 + L2     |
| `survive_night`  | 3             | L0 + L1 + L2 + L3 |

The `wally-play` CLI's `--layer-depth N` argument sets this; layers
above `N` are simply no-ops.

## CLI

### Train a layer

```powershell
# 1. L1 — needs the L0 checkpoint
& .venv-windows\Scripts\python.exe -m wally.cli.train_hierarchy `
    --layer l1 `
    --config configs/hierarchy_l1.yaml `
    --l0-checkpoint checkpoints/wood_1000/checkpoint_1000.pt `
    --log-file logs/hierarchy_l1.log

# 2. L2 — needs the L1 checkpoint
& .venv-windows\Scripts\python.exe -m wally.cli.train_hierarchy `
    --layer l2 `
    --config configs/hierarchy_l2.yaml `
    --l0-checkpoint checkpoints/wood_1000/checkpoint_1000.pt `
    --lower-checkpoint checkpoints/hierarchy_l1/checkpoint_2000.pt `
    --log-file logs/hierarchy_l2.log

# 3. L3 — needs the L2 checkpoint
& .venv-windows\Scripts\python.exe -m wally.cli.train_hierarchy `
    --layer l3 `
    --config configs/hierarchy_l3.yaml `
    --l0-checkpoint checkpoints/wood_1000/checkpoint_1000.pt `
    --lower-checkpoint checkpoints/hierarchy_l2/checkpoint_2000.pt `
    --log-file logs/hierarchy_l3.log
```

Each `wally-train-hierarchy` call writes a single checkpoint
`checkpoints/hierarchy_<layer>_<step>.pt` per `checkpoint_interval`,
containing:

```python
{
    "model_state_dict": <JEPA world model>,
    "encoder_state_dict": <L_n encoder (L0..L_(n-1) frozen + L_n projection)>,
    "global_step": int,
    "config": <HierarchyConfig.to_dict()>,
}
```

`CUDA` is required; the CLI exits with a clear error if
`torch.cuda.is_available()` is False, matching `wally-train`.

### Run a hierarchical agent

```powershell
# Train a goal embedding for L3 (one per task) and save it
& .venv-windows\Scripts\python.exe -c "
import torch
from wally.hierarchy.goal import LearnedGoalEmbedding
g3 = LearnedGoalEmbedding(D=16, task_id='get_wood')
torch.save({'g': g3.g.detach()}, 'checkpoints/g3_get_wood.pt')
"

# Run the agent
& .venv-windows\Scripts\python.exe -m wally.agent.play `
    --checkpoint checkpoints/wood_1000/checkpoint_1000.pt `
    --target-embedding checkpoints/g3_get_wood.pt `
    --planner hierarchical-embedding `
    --layer-depth 3 `
    --hierarchy-checkpoint checkpoints/hierarchy_l3/checkpoint_2000.pt `
    --config configs/ag_test_wood.yaml `
    --record --output-dir ag-tests/run_hierarchy_3k
```

## Drift threshold tuning

Each layer's threshold is `epsilon * sqrt(D)`. The defaults in
`configs/hierarchy_l*.yaml` are:

| Layer | `epsilon` | `D` | `threshold` |
|-------|-----------|-----|-------------|
| L1    | 0.10      | 64  | 0.80        |
| L2    | 0.20      | 32  | 1.13        |
| L3    | 0.30      | 16  | 1.20        |

Tune by:
1. Train the layer (`wally-train-hierarchy --layer lN`).
2. Run a held-out trajectory through the layer's `DriftMonitor`.
3. Pick the threshold at the 90th percentile of the drift distribution
   (logged by `wally-train-hierarchy` at the end of training).
4. Override the per-layer `drift_epsilon` in the config YAML.

Higher layers tolerate more drift because they think in coarser units.
If a layer triggers gentle-correct / replan too often, raise the
threshold; if it never triggers, the upper layer is useless — lower
the threshold.

## Architecture

```
src/wally/hierarchy/
├── __init__.py
├── bus.py            # MessageBus — per-layer FIFO queues
├── config.py         # LayerSpec, HierarchyConfig (dataclasses)
├── drift.py          # DriftMonitor, Replanner, ReplanDecision enum
├── encoders.py       # L1Encoder, L2Encoder, L3Encoder
├── goal.py           # LearnedGoalEmbedding (g3 nn.Parameter)
├── jepa.py           # JEPAWorldModel (AdaLN-conditioned predictor)
├── loss.py           # temporal_coherence_loss, combined_hierarchy_loss
├── planner.py        # HierarchicalEmbeddingPlanner (multi-layer coordinator)
├── runtime.py        # LayerRuntime (per-layer background loop)
├── trainer.py        # HierarchyTrainer (training loop)
└── types.py          # LayerState, LayerMessage

src/wally/cli/
└── train_hierarchy.py # wally-train-hierarchy CLI

src/wally/agent/
├── planner_factory.py # build_planner() now also handles "hierarchical-embedding"
├── protocol.py        # HierarchicalEmbeddingPlannerAdapter
└── loop.py            # AgentLoop.run_episode(goal_frame, target_embedding=...)

configs/
├── hierarchy_l1.yaml
├── hierarchy_l2.yaml
└── hierarchy_l3.yaml
```

## Smoke tests

`pytest -m smoke -k hierarchy` runs the integration suite
(`tests/test_hierarchy_*.py`): a full L1-training step on a real L0
checkpoint, the runtime wiring, the drift detection logic, and the
end-to-end multi-layer planner on a fake rollout.
