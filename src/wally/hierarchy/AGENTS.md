# src/wally/hierarchy — LeWorldModel hierarchy stack

L1, L2, L3 JEPA world models on top of the L0 LeWorldModel. The L0
is treated as a frozen black box (only its encoder is used). L1+ are
pure JEPA predictors in their own learned embedding spaces; they do
not reconstruct pixels, only embeddings.

## Public API

| Symbol                         | Where                                       |
|--------------------------------|---------------------------------------------|
| `JEPAWorldModel`               | `jepa.py`                                   |
| `L1Encoder`, `L2Encoder`, `L3Encoder` | `encoders.py`                         |
| `LayerSpec`, `HierarchyConfig` | `config.py`                                 |
| `LayerState`, `LayerMessage`   | `types.py` (no strings, only `Tensor[D]`)   |
| `MessageBus`                   | `bus.py`                                    |
| `LayerRuntime`                 | `runtime.py`                                |
| `DriftMonitor`, `ReplanDecision` | `drift.py`                                |
| `LearnedGoalEmbedding`         | `goal.py`                                   |
| `temporal_coherence_loss`, `combined_hierarchy_loss` | `loss.py`           |
| `HierarchyTrainer`             | `trainer.py`                                |
| `HierarchicalEmbeddingPlanner` | `planner.py`                                |
| `HierarchicalEmbeddingPlannerAdapter` | `wally/agent/protocol.py`             |

## Frozen / trainable invariant

- The L0 LeWorldModel is **frozen** at every layer's encoder. Its
  parameters have `requires_grad = False` on construction.
- Each layer's encoder has exactly one **trainable** linear
  projection (D_lower → D_n); all other parameters are frozen.
- Each layer's `JEPAWorldModel` is fully **trainable**.

This invariant is checked by the smoke tests
(`tests/test_hierarchy_smoke.py`).

## Training each layer

Run the CLI in this exact order, since each upper layer depends on the
checkpoint of the layer below:

```bash
# L1: needs the L0 checkpoint only
wally-train-hierarchy --layer l1 --config configs/hierarchy_l1.yaml \
    --l0-checkpoint checkpoints/wood_1000/checkpoint_1000.pt

# L2: needs the L1 checkpoint
wally-train-hierarchy --layer l2 --config configs/hierarchy_l2.yaml \
    --l0-checkpoint checkpoints/wood_1000/checkpoint_1000.pt \
    --lower-checkpoint checkpoints/hierarchy_l1/checkpoint_2000.pt

# L3: needs the L2 checkpoint
wally-train-hierarchy --layer l3 --config configs/hierarchy_l3.yaml \
    --l0-checkpoint checkpoints/wood_1000/checkpoint_1000.pt \
    --lower-checkpoint checkpoints/hierarchy_l2/checkpoint_2000.pt
```

CUDA is required (matches `wally-train`). The CLIs share the same
device-error path; the hierarchy CLIs also exit with code 2 if
`torch.cuda.is_available()` is False.

### Constraint: `K` must be less than the shard chunk size

The trainer enforces `seq_length > K` strictly
(`src/wally/hierarchy/trainer.py:156-159`). The converted shards in
`data/shards/<task>/` are stored as **64-frame chunks** (per
`src/wally/AGENTS.md`), and the WebDataset dataloader uses
`skip_short=True`, which drops every sample shorter than `seq_length`.

Putting these together: the maximum usable `K` per layer is **63** on
the current 64-frame-chunk shards. The shipped `configs/hierarchy_l1.yaml`
(`K=64, seq_length=128`) is therefore incompatible with the current
data — it will hang silently in the dataloader, never log a step, and
never write a checkpoint. For a smoke run, use `K=2..8, seq_length=16..32`
as in `configs/hierarchy_l1_smoke.yaml` and
`configs/hierarchy_l1_5k.yaml`. To get to the design-point `K=64` (and
the upstream `K=128` for L2, `K=1024` for L3) you need a
**chunk-concatenation dataloader** that glues adjacent
`__chunkNNN` entries from the same `episode_id` into long sequences;
that dataloader does not exist yet (see "L2 path is not viable yet"
below).

### Dataloader settings

Both the L0 (`wally-train`) and L1 (`wally-train-hierarchy`) trainers
now honour `num_workers`, `persistent_workers`, and `prefetch_factor`
from the config. Recommended values for the GPU on this host:

```yaml
num_workers: 8
persistent_workers: true
prefetch_factor: 4
```

This takes the L0 from 0.45 s/step to 0.18 s/step on the 64-frame
shards (≈ 2.5× speedup) and saturates the GPU. WebDataset workers are
fragile on Windows — if the training hangs without progress, the
workers have died; check stderr for
`DataLoader worker (pid(s) ...) exited unexpectedly` and fall back to
`num_workers: 4` or a different sharding scheme.

### Early stopping (L1/L2/L3)

The hierarchy trainer mirrors the L0 trainer's EMA-based early stop
(`src/wally/AGENTS.md##-early-stopping`). When the EMA of
`total_loss` stops improving for `patience` steps, training stops
and `checkpoint_best.pt` holds the lowest-EMA-loss weights.

```yaml
early_stop: true               # default: false
early_stop_patience: 1500      # stop if EMA total_loss doesn't improve for N steps
early_stop_min_step: 2000      # don't consider stopping before this step
early_stop_ema_alpha: 0.1      # EMA smoothing factor (lower = smoother)
early_stop_min_delta: 0.0      # minimum improvement to count as "better"
```

The defaults in `configs/hierarchy_l1_5k.yaml` and
`configs/hierarchy_l1_smoke.yaml` leave early stop disabled so the
short smoke runs always complete. `configs/hierarchy_l1.yaml` and
`configs/hierarchy_l2.yaml` are designed for full runs — enable
`early_stop: true` to save compute once the EMA plateaus.

### wandb logging

Set `wandb_enabled: true` (default) in the config to log to the
`wally` wandb project. The run name is
`<wandb_project>-<K>-<D>-step-<global_step>` so resumed runs are
distinguishable in the dashboard. Set `wandb_project` to change the
project, or `wandb_enabled: false` to skip wandb entirely (smoke
tests, CI).

## L2 path: now viable

The four pieces listed below are all in place as of this change;
L1 + L2 can now be trained end-to-end with the optimized
``ConcatenatedShardDataset`` dataloader.

1. **Chunk concatenation.** ``ConcatenatedShardDataset`` in
   :mod:`wally.data.concat_dataset` pre-indexes every
   ``__chunkNNN`` member, groups by ``episode_id``, and yields
   random ``seq_length``-frame windows that may span multiple
   chunks. It handles the "last chunk in episode is shorter than
   ``chunk_size``" case (the converter packs the remainder;
   observed tail sizes 2..64 frames). 8-worker DataLoader with
   ``prefetch_factor=4`` hits ≈ 70 samples/s on the wood shards
   (CPU-bound; GPU step is ≈ 0.05 s).

2. **Real K values.** ``configs/hierarchy_l1.yaml`` is now
   ``K=32, seq_length=64`` (10 000 steps, ≈ 1 h) and
   ``configs/hierarchy_l2.yaml`` is ``K=128, seq_length=256``
   (5 000 steps, ≈ 2-3 h). Both default to
   ``use_concat_dataloader: true``.

3. **``g2`` goal data.** ``logs/make_g2.py`` encodes a centroid
   of the late-episode "have wood" frames (the last 200 frames
   of each episode, scored by wood-trunk ratio + recent
   ``attack=1`` rate) through the trained L1 + L2 encoders and
   saves the 32-dim tensor to ``checkpoints/g2_have_wood.pt``.

4. **Re-trained checkpoints.** Run, in order::

       wally-train-hierarchy --layer l1 --config configs/hierarchy_l1.yaml
           --l0-checkpoint checkpoints/wood_1000/checkpoint_1000.pt
       wally-train-hierarchy --layer l2 --config configs/hierarchy_l2.yaml
           --l0-checkpoint checkpoints/wood_1000/checkpoint_1000.pt
           --lower-checkpoint checkpoints/hierarchy_l1/checkpoint_10000.pt
       & .venv-windows\Scripts\python.exe logs\make_g2.py
           --l0-checkpoint checkpoints\wood_1000\checkpoint_1000.pt
           --l1-checkpoint checkpoints\hierarchy_l1\checkpoint_10000.pt
           --l2-checkpoint checkpoints\hierarchy_l2\checkpoint_5000.pt
           --output checkpoints\g2_have_wood.pt

   See :mod:`wally.data.concat_dataset` for the dataset API and
   ``logs/make_g2.py`` for the goal recipe.

## Runtime streaming protocol

The agent loop calls `planner.tick_with_frame(current_frame)` every
step. The planner:

1. Encodes the frame through the lowest layer's encoder
   (e.g. L1Encoder for an L1+L2+L3 stack).
2. Pushes that L1 state embedding upward to L1's runtime.
3. L1's runtime computes `predicted_s` (using its latest
   `target_embedding` from above), measures drift, and either
   gentle-corrects, replans, or escalates.
4. The planner then reads the latest L1 `target_embedding`,
   projects it to L0 space, and runs the L0 CEM as usual.

All inter-layer messages are `Tensor[D]` (see `LayerMessage` in
`types.py`); no strings flow between layers at runtime.

### Per-layer runtime log

`LayerRuntime.tick` writes an INFO line to
`wally.hierarchy.runtime.<layer>` every 50 ticks, plus every time
`DriftMonitor.update` returns a non-`NONE` decision. The line carries
the tick counter, the drift value, the cumulative replan / gentle /
escalate counts, and the decision. To see L1 activity live:

```bash
podman exec wally-dev tail -f /tmp/wally-play.log | grep wally.hierarchy
```

If a layer's drift stays above `epsilon * sqrt(D)` for the whole
episode, the layer is undertrained for the data — either train more
steps, raise `epsilon`, or fix the goal embedding (`g1`, `g2`, `g3`).
A drift value of ~3-7 with the 64-dim L1 + 200-step training
(smoke) is expected and the L1 will replan every tick; that is the
point of the protocol under test, not a bug.

## Variable depth

`HierarchicalEmbeddingPlanner.set_goal(target)` sets the topmost
layer's target. The depth of the activated stack is whatever layers
were passed in to the constructor; for runtime variable depth, build
multiple planners and pick at agent start.
