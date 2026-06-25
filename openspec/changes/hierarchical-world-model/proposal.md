## Why

The current LeWM + CEM agent is a single-level (L0) world model: it predicts the next 64×64 frame and plans 8 motor steps ahead. It can get the agent near a tree but cannot break a log — because "breaking a log" is a concept that lives at a higher level of abstraction than motor commands. After 12k training steps the cost is still 1206 and `mine_block` events are 0. The bottleneck is not the world model; it is the **abstraction level** at which it plans.

The agent also has no recovery from drift: if L0's predicted state diverges from the actual state, there is no higher layer to notice and correct. In human terms, the agent has motor control but no conscious monitoring.

This change adds a stack of **learned world models, one per abstraction level**, that plan in continuous embedding space and communicate by streaming state + drifting detection. Each layer is the same JEPA-style world model architecture with different time-scale hyperparameters. Layers coordinate via a continuous protocol — no discrete skills, no strings, no human-interpretable vocabulary.

## What Changes

- **Add a hierarchical world-model stack** (L1, L2, L3) on top of the existing L0 LeWM. Each layer is a learned JEPA world model that predicts the next layer's state embedding, conditioned on a target embedding sent from above.
- **Add a learned goal-embedding space**: "I want 16 oak logs" is a single vector `g3`, not a string. The hierarchy steers the agent toward states whose embeddings are close to `g3`.
- **Add drift-detection replanning**: every layer continuously compares its predicted state embedding to the actual state embedding streamed from below. When drift exceeds a threshold, the layer surfaces from its background loop and re-plans (or signals the layer above).
- **Add variable depth at runtime**: simple tasks (move forward) use L0 only; medium tasks (get wood) use L0–L2; long-horizon tasks use L0–L3. The depth is a runtime choice, not a fixed architecture decision.
- **Add a unified training pipeline**: each upper layer is trained with a temporal-coherence self-supervised objective on the same trajectories that L0 already uses. No new data collection needed for V1.
- **Extend the agent loop** to support the streaming state protocol between layers, with the L0 LeWM unchanged at the bottom.

## Capabilities

### New Capabilities

- `hierarchical-world-model`: The core stack of L1/L2/L3 world models, each a JEPA-style predictor operating at a different time horizon.
- `layer-communication-protocol`: Continuous-embedding message format between layers, drift detection, and replanning triggers.
- `learned-goal-embedding`: The goal vector `gN` per layer, learned from objective descriptions; replaces the current single goal-frame for the CEM planner.
- `drift-detection-replanning`: Per-layer prediction-error monitoring and the planning-loop logic that surfaces from background mode when drift exceeds threshold.

### Modified Capabilities

- `agent-loop`: The episode loop is extended from a single planner call per step to a streaming multi-layer loop. L0 still runs every tick; L1+ run in background and tick on drift events.
- `goal-conditioned-planning`: The single-goal-frame cost is replaced by a multi-layer embedding-distance cost. L0 still uses pixel-level cost (backwards compatible) but L1+ use learned embedding distance.
- `high-level-planner`: The current hand-coded hierarchical planner (`src/wally/planner/hierarchical_planner.py`) is extended to support the new continuous-embedding protocol. Its public API stays the same so existing callers are unaffected.
- `mpc-cem-planner`: The CEM planner's cost function gains an optional `target_embedding` argument (in addition to the existing `goal_frame`). When omitted, behavior is identical to today.

## Impact

- **New code under `src/wally/hierarchy/`**: layer implementations, communication protocol, training loops, replanning logic.
- **Extended code under `src/wally/agent/`**: `loop.py` and `planner_factory.py` get a new `hierarchical-embedding` planner kind.
- **Extended code under `src/wally/planner/`**: `cem.py` and `plan.py` accept an optional target-embedding argument.
- **New CLI**: `wally-train-hierarchy` for training the upper layers on existing data.
- **New configs under `configs/`**: `hierarchy_l1.yaml`, `hierarchy_l2.yaml`, `hierarchy_l3.yaml`.
- **No new data collection required for V1** — the upper layers are trained on the existing `data/shards/treechop_full/` shards.
- **No breaking changes to existing commands** — `wally-train`, `wally-play --planner cem` (the default), and the existing `lewm_*` configs continue to work as before.

## Apply order recommendation

When implementing this change, **start with task group 7 (planner extensions)** — adding the `target_embedding: torch.Tensor | None = None` argument to `GoalConditionedPlanner.plan` and the `search_space='embedding'` mode to `CEMOptimizer.optimize`. This is the smallest additive change that unblocks everything else: once L0's planner accepts a `target_embedding` instead of a `goal_frame`, the hierarchy can be wired in incrementally on top. Bootstrap L1 from `checkpoints/wood_12000/checkpoint_12000.pt` (the 12k-step L0 checkpoint from the previous session).
