## Context

The wally project currently trains a LeWorldModel (L0) on 64×64 frame + 25-dim action trajectories, then plans with a CEM-based MPC that minimizes pixel distance to a single 64×64 goal frame. After 12k training steps the agent can get close to trees but never breaks a log (`mine_block` events: 0, `pickup` events: 0). Diagnostic evidence shows the world model is improving — total loss 0.10 at 12k vs 0.65 at 1k — but the policy has not crossed the threshold needed for the wood task. The bottleneck is **abstraction level**, not model capacity or training time.

The agent also has no recovery from drift: the CEM planner re-plans every 4 frames against a static goal frame, but if the agent's path diverges from the planner's expectation, the planner simply re-optimizes against the new state without ever recognizing that the *plan itself* may be wrong. There is no equivalent of the human experience of "drifting into the wrong shop and then realizing."

The LeCun JEPA line of work (I-JEPA, V-JEPA, V-JEPA 2) and LeCun's "Path Towards Autonomous Machine Intelligence" architecture propose exactly this: a stack of joint-embedding world models, each operating at a different time scale, communicating in embedding space, with a higher-level monitor that detects drift. This change brings that pattern into wally, but built from first principles on the existing LeWM substrate rather than imported from those papers.

## Goals / Non-Goals

**Goals:**

- Stack 4 learned world models (L1, L2, L3) on top of the existing L0 LeWM, each operating at an 8× time horizon of the layer below.
- Replace the current single goal-frame cost with a multi-layer embedding-distance cost. Higher layers optimize in their own learned embedding space; L0 remains pixel-cost for backward compatibility.
- Implement a streaming communication protocol: every layer emits a state embedding every tick; every layer receives a target embedding from above and a state embedding from below.
- Implement drift detection per layer, with a background loop that re-plans when predicted-vs-actual embedding distance exceeds a per-layer threshold.
- Make layer depth a **runtime choice** per task, not a fixed architecture property. Simple tasks (move forward) use L0; medium tasks (get wood) use L0–L2; long-horizon tasks (survive a night) use L0–L3.
- Train the upper layers from the existing `data/shards/treechop_full/` shards using temporal-coherence self-supervision. No new data collection required for V1.

**Non-Goals:**

- Online RL training of the upper layers in this change. That is V3 of the design (the proposal lists it) and belongs in a follow-up change.
- Unsupervised skill discovery with diversity objectives (DIAYN, DADS). That is V2 of the design and a follow-up.
- Replacing the L0 LeWM. L0 is treated as a fixed black box at the bottom of the stack.
- Adding language conditioning (e.g. "get 16 oak logs" → tokens). The V1 objective is just a learned `g3` vector per task, not natural language.
- More than L3 in the stack. 4 new layers (L1, L2, L3) on top of L0 is enough to plan at the time horizon of a Minecraft task session.

## Decisions

### Decision 1: 5-layer stack with 8× time-horizon multiplier

```
Layer  Horizon (frames)  Time       Embedding dim  Conditioning on
L0     8                 0.4 s      192            action sequence
L1     64                3.2 s       64            target embedding g1
L2     512               25.6 s      32            target embedding g2
L3     4096              ~3.4 min    16            target embedding g3
L4     session           varies      16            objective text → g4
```

L0 already exists. L4 is just a goal vector (no world model); the four new world models are L1, L2, L3.

**Rationale**: 8× is a reasonable log scale that makes a "chain of thinking" of N steps correspond to 8^N frames. Three layers above L0 covers ~3.4 minutes of L3-level planning, which is enough for the "get wood" / "find food" / "survive the night" tasks. Going deeper would be needed for "build a castle" but is out of scope.

**Alternatives considered**:
- *Fixed 3-layer stack* (L0, L1, L2). Too shallow for long-horizon tasks; replanning at L2 would still need to think at 25-second granularity, which is too coarse for "chop 16 logs one at a time."
- *Continuous-depth stack* (decide depth per task at runtime). This is what the user asked for. Implementation: each task declares which layers it activates; unactivated layers are simply no-ops. The architecture is fixed at 5 layers, but the depth used is variable.

### Decision 2: Each layer is the same JEPA architecture with different hyperparameters

L1, L2, L3 are all instances of the same `JEPAWorldModel` class. They differ only in:
- time horizon (K frames)
- embedding dimension (D)
- attention depth / width
- training data sampling rate (L2 samples every 8 frames, L3 every 64)

L0 stays as the existing `LEWMAdaLNPredictor` — it has its own action-conditioning and is not a generic JEPA, just a LeWorldModel with a JEPA-style loss. L1+ are pure JEPA: input is `(state_embedding, target_embedding)`, output is predicted next state embedding, loss is L2 on the embedding (no pixel reconstruction).

**Rationale**: One architecture, many instantiations. The same training code, inference code, and planning code work for L1, L2, L3. New layers can be added by adding a row to the config.

**Alternatives considered**:
- *Different architecture per layer* (e.g. Transformer for L1, Mamba for L2, etc.). More flexibility but more code surface. We can refactor later if a layer's nature really demands a different architecture.
- *Use the L0 architecture (LeWM) for every layer*. Doesn't work — LeWM is action-conditioned; L1+ take a target embedding instead, and a 25-dim action vector is the wrong shape at the higher levels.

### Decision 3: Continuous embeddings for all layer-to-layer communication. No strings, no discrete skills.

Every message between layers is a `np.ndarray` (or `torch.Tensor`):
- top-down: `target_embedding: Tensor[D]` — what the layer above wants the layer below to steer toward
- bottom-up: `state_embedding: Tensor[D]` — what the layer below actually produced

No vocabulary, no skill IDs, no symbolic task names anywhere in the runtime path. The "what is the skill" question is answered implicitly by the geometry of the embedding space.

**Rationale**: The user's framing: when you think "go to the airport" you don't have a string `call_taxi` in your head — you have a fuzzy goal-state representation. Strings in the runtime path would force us to pre-define a skill vocabulary, which would be brittle and not learned. Embeddings let the geometry emerge from training.

**Alternatives considered**:
- *Hybrid (discrete + continuous)*, as in my first design pass. Rejected by the user — they pointed out that even the "what is a concrete plan" is something only the higher layer knows, so it should be a learned embedding.
- *Pre-trained language embeddings (e.g. CLIP text) for the goal*. Out of scope for V1; a possible V3 if we want natural-language goals.

### Decision 4: Drift detection via continuous embedding distance, with a per-layer threshold

Every layer runs a background loop that:
1. Computes `predicted_s = world_model.predict(s_prev, g)` for the next tick
2. Receives `actual_s` from the layer below
3. Computes `drift = ||actual_s − predicted_s||`
4. If `drift > threshold(layer)`, surface from background mode: either re-plan the current target embedding, or signal the layer above that the plan needs adjustment

The threshold scales with the layer's embedding dimension: `threshold(L_n) = ε_n * sqrt(D_n)`. Concrete defaults: ε0=0.05, ε1=0.10, ε2=0.20, ε3=0.30 (higher layers tolerate more drift because they think in coarser units).

**Rationale**: Matches the user's airport-shop example. The agent doesn't *fail* — the low level happily reports "I walked forward 6 blocks" — but the high level notices the predicted-vs-actual mismatch and redirects. This is continuous, not event-driven, so it catches the subtle drift case (walking into the wrong building) that a hard-failure trigger would miss.

**Alternatives considered**:
- *Hard failure only* (my V0 design). The user pointed out this misses the drift case.
- *Learned drift detector* (a small classifier that predicts "is this drift a problem?"). V2 of this design.

### Decision 5: Background loop per upper layer, not request/response

L1+ do not block on a request from below. They maintain a continuously-updated belief about the world state (`predicted_s`), and they run a slow background tick that re-checks the belief every ~5 seconds (L1) or ~30 seconds (L2) or ~3 minutes (L3). When the belief error is too high, they either:
- **Adjust `g`** (gentle correction — the same plan, slightly retargeted), or
- **Re-issue the task** (full replan — "this plan won't work, try a different one"), or
- **Signal the layer above** (escalation — "I cannot achieve this, please re-plan me")

L0 runs in a tight per-tick loop as today. L1+ run in background and only block briefly when surfacing from background to re-plan.

**Rationale**: The user's framing: "you are walking, you are thinking in something else, then you stop thinking and you realized that you enter in a shop instead of restaurant." The high level is not constantly micromanaging; it surfaces periodically, checks, and either continues or corrects. This is also how STRIPS / classical planners work: the planner is a discrete process invoked on demand, not a continuous control loop.

**Alternatives considered**:
- *Event-driven only* (re-plan on hard failure or subtask boundary). Misses the drift case.
- *Synchronous tick per layer* (every layer runs every tick). Wasteful — L3 doesn't need to think at 0.4s granularity.

### Decision 6: Training by temporal-coherence self-supervision on existing shards

Each upper layer L_n is trained on the same `data/shards/treechop_full/` shards that L0 uses. For each trajectory:
1. Sample a state at time `t` and a state at time `t + K` (where K is L_n's time horizon)
2. Encode both states into L_n's embedding space using a small CNN encoder (initialized from the L0 encoder and fine-tuned, or trained from scratch)
3. Train L_n's predictor: `world_model(s_t, g) → s_{t+K}` where `g` is set to `s_{t+K}` itself (the temporal-coherence objective — predict the future, conditioned on the future as the goal)

This is exactly the JEPA training objective (Bardes et al, 2024), applied to the semantic level. The encoder is shared across layers with output projection per layer.

**Rationale**: No new data collection. The upper layers learn the *dynamics* of the world from existing trajectories. They will not learn *what makes a good plan* — that comes from V2 (diversity) and V3 (RL). But the world model is the foundation, and this is the cheapest way to get one.

**Alternatives considered**:
- *Collect new goal-conditioned trajectories*. Too expensive for V1.
- *Bootstrap from L0's existing representations*. Partially done — we reuse the L0 encoder weights as initialization for the L1 encoder.

### Decision 7: L0 is treated as a fixed black box. No changes to LeWM training or inference.

L0 (`lewm-adaln-predictor`) is the world model the user has already spent time getting right. It has its own SIGReg loss, AdaLN-Zero predictor, action conditioning, and CEM planner. We do not modify it.

L1+ are implemented in a new `src/wally/hierarchy/` package and interact with L0 only through its public API: `encoder.encode(frames)`, `predictor.predict(state, actions)`, `cem.plan(state, target_embedding)`.

**Rationale**: Don't break what works. The single biggest risk in this change is destabilizing L0, which is the foundation everything else sits on. Keep L0 frozen, build the hierarchy on top.

**Alternatives considered**:
- *Joint end-to-end training of all layers*. More powerful in theory, much more unstable in practice. Layer-wise training (L0 first, then L1 on top of frozen L0, etc.) is the standard recipe and the right starting point.

## Risks / Trade-offs

- **[Risk] L1's embedding space doesn't usefully cluster "near a tree" or "facing a log" in a way the planner can use.** The temporal-coherence objective learns a *predictive* embedding, not a *useful* one. → **Mitigation**: V2 adds a diversity-based loss that pulls "approach tree" L0-states together in L1's embedding space. V1 may not solve the wood task; the goal is to validate the architecture. The real test is "does the L1 world model + planner steer the agent toward more wood-relevant states than the flat L0 agent?"
- **[Risk] Drift threshold is hard to tune.** If too low, the agent re-plans every tick and L1+ become no-ops. If too high, the agent never re-plans and the hierarchy is useless. → **Mitigation**: defaults are `0.05, 0.10, 0.20, 0.30 × sqrt(D)`. We expose a `--drift-threshold` CLI flag and a per-layer override. Validation: log the drift distribution on a held-out trajectory and pick thresholds at the 90th percentile.
- **[Risk] Variable depth at runtime complicates the agent loop.** Each task needs to declare which layers to activate. → **Mitigation**: the agent loop accepts a `layer_depth: int` argument; layers above that depth are no-ops. The task description includes a default depth (`move_forward: 0`, `get_wood: 2`, `survive: 3`).
- **[Risk] Existing tests for the agent loop break because we changed `loop.py`.** → **Mitigation**: the change to `loop.py` is additive. The existing `CEMPlannerAdapter` path is unchanged; a new `HierarchicalPlannerAdapter` is added alongside it. Existing tests still pass.
- **[Risk] New layers add inference latency.** Each L_n tick requires running L_n's predictor and an L_(n-1) embedding lookup. The hierarchy could add 5-20 ms per tick at the upper layers. → **Mitigation**: L1+ run in background, not in the per-tick loop. L0 latency is unchanged. We accept ~50ms of L1 latency amortized over multiple L0 ticks.
- **[Risk] Embedding dimensions (64, 32, 16) are too small to capture enough structure.** → **Mitigation**: validate on the wood task first. If the L1 embedding can't distinguish "near a tree" from "in a forest but not near a tree," the next experiment doubles D1. The architecture supports this without code changes.

## Migration Plan

This change is purely additive. No existing commands, configs, or tests change behavior:

1. **Land the new code under `src/wally/hierarchy/`.** L0, L1, L2, L3 are not affected.
2. **Add the new `hierarchical-embedding` planner kind to `planner_factory.py`.** The default (`cem`) and existing (`gradient`, `hierarchical`) kinds continue to work as before.
3. **Add `target_embedding` as an optional argument to `CEMPlanner.plan()`.** When omitted, behavior is identical to today. When provided, the cost function adds an embedding-distance term.
4. **Add the new CLI `wally-train-hierarchy`.** This is a new command; it does not modify `wally-train`.
5. **Add the new configs `hierarchy_l1.yaml`, `hierarchy_l2.yaml`, `hierarchy_l3.yaml`.** The existing `lewm_*.yaml` configs are unchanged.
6. **Add a smoke test that loads a frozen L0 checkpoint, trains L1 for 100 steps, and verifies the L1 embedding distance between two states from the same trajectory is smaller than between two random states.** This validates the training pipeline end-to-end without requiring a real episode.

**Rollback**: delete `src/wally/hierarchy/`, revert the changes to `planner_factory.py` and `cem.py`, remove the new CLI and configs. No existing functionality is affected.

## Open Questions

- **Where does the encoder live?** The L1 encoder takes frames and produces the L1 state embedding. Should it be the L0 encoder (frozen, with a learned linear projection to L1's D1=64), or a new CNN trained from scratch? Decision: start with the L0 encoder frozen + linear projection; revisit if the L1 embedding space is too impoverished.
- **Should L1 reuse L0's CEM planner or have its own?** L1 needs to search over target embeddings, not over action sequences. A CEM over a 64-dim continuous space is fine; a gradient-based optimizer would also work. Decision: start with a small gradient-based optimizer (CEM is overkill for 64 dims), and only use CEM if we find gradient descent gets stuck in local minima.
- **How is the L3 goal `g3` set?** The proposal says "learned from objective descriptions." For V1, `g3` is a learned parameter that's optimized end-to-end on the task loss (e.g. "16 oak logs collected in N steps"). For V2, it could be derived from a language embedding. For V3, it could be user-provided as a vector. Decision: V1 is the optimized-end-to-end version. The user provides the *task* (a description), and `g3` is learned.
- **How do we validate that the hierarchy works?** A simple test: does the L1 world model + planner steer the agent toward more wood-relevant states than the flat L0 agent? Concrete metric: in a 1500-step episode, how many frames does the agent spend with `dist_to_nearest_wood < 2`? This is a better proxy for "is the hierarchy planning well" than the final "did we get wood" binary.
