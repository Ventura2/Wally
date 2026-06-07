## Context

Wally currently has a working LeWorldModel (ViT encoder + causal Transformer predictor) and a flat CEM-based MPC planner (`GoalConditionedPlanner`). The planner encodes current/goal frames to latents, then uses CEM to find an action sequence that minimizes latent-space distance to the goal over a fixed horizon (default 8 steps). The `LatentRollout` class handles autoregressive latent prediction with gradient detachment.

This architecture works for short-horizon tasks but cannot handle multi-phase goals like "craft diamond sword" which require 50+ steps across distinct phases (mining, crafting, smelting). The existing planner has no notion of subgoals, no memory beyond the current frame, no gradient-based refinement, and no safety or uncertainty handling.

Constraints:
- AMD RX 6700 XT with ROCm — all modules must be ROCm-compatible (no CUDA-only ops)
- Single-GPU training — hierarchical models must fit in 12GB VRAM
- Existing `WorldModelProtocol` and `LatentRollout` interfaces should be preserved/extended, not replaced

## Goals / Non-Goals

**Goals:**
- Enable planning over 50+ step horizons via hierarchical decomposition
- Detect subgoals automatically from trajectory data using context-change detection
- Train a high-level world model on abstract transitions for subgoal planning
- Add gradient-based MPC for fine-grained low-level action optimization
- Add recurrent memory (LSTM) to handle partial observability
- Implement curriculum training to progressively scale planning horizon
- Add curiosity-driven exploration for diverse training data collection
- Add safety constraints and uncertainty estimation to avoid catastrophic plans

**Non-Goals:**
- Text-conditioned goal specification (future work)
- Multi-agent or collaborative planning
- Real-time interactive planning (< 100ms latency)
- Transfer learning across different Minecraft worlds/seeds
- Full Dreamer-style actor-critic training (only warm-starting from value/policy head)

## Decisions

### 1. Subgoal Detection: THICK-style change-point detection over C-RSSM

**Decision**: Use a THICK-inspired approach — detect context-change points by measuring prediction error spikes in a trained world model, then segment trajectories at these boundaries.

**Rationale**: THICK is simpler to implement than C-RSSM since it leverages the existing LeWorldModel's prediction error directly. C-RSSM requires training a separate recurrent state-space model with context-change gating, adding significant complexity. THICK's approach of "where does the world model get surprised?" aligns naturally with our existing architecture.

**Alternatives considered**:
- C-RSSM: More principled but requires a separate RSSM training pipeline
- Manual subgoal specification: Simpler but doesn't scale and requires domain knowledge
- Clustering on latent embeddings: Less temporally aware, misses sequential structure

### 2. High-Level Model: Separate LeWorldModel on abstract transitions

**Decision**: Train a second LeWorldModel instance on abstract transitions (subgoal-to-subgoal) rather than extending the existing model with a hierarchical head.

**Rationale**: A separate model cleanly separates concerns — the low-level model remains unchanged and can be used independently. The high-level model operates on the same latent space (same encoder) but predicts transitions between context-change points. This avoids architectural coupling and allows independent training schedules.

**Alternatives considered**:
- Shared encoder with hierarchical prediction head: Tighter coupling, harder to debug
- Single model with variable-length prediction: Complex training, unstable gradients
- Option-critic framework: Requires policy training, not compatible with our planning-as-inference approach

### 3. Low-Level Planning: Gradient-based MPC with CEM warm-start

**Decision**: Keep CEM as the primary optimizer but add gradient-based refinement as a second stage. Use CEM output as initialization for gradient descent on the action sequence.

**Rationale**: Pure gradient-based planning is brittle (local minima, requires differentiable rollouts). CEM is robust but sample-inefficient. Combining them — CEM for global search, then gradient descent for local refinement — gets the best of both. The existing `LatentRollout` already supports `gradient_policy="straight_through"` which enables this.

**Alternatives considered**:
- Pure gradient descent: Too sensitive to initialization
- Pure CEM with larger population: Computationally expensive, diminishing returns
- MPPI (Model Predictive Path Integral): Good alternative but less compatible with existing CEM infrastructure

### 4. Memory: LSTM over Transformer-XL

**Decision**: Add a single-layer LSTM after the ViT encoder to maintain observation history. The LSTM processes the sequence of patch-token-means from the encoder and outputs a context-augmented latent.

**Rationale**: LSTM is simpler, faster to train, and has lower memory overhead than Transformer-XL. For our use case (10-50 frame history), LSTM's vanishing gradient problem is manageable. Transformer-XL's segment-level recurrence is more valuable for very long sequences (>100 steps) which is beyond our current scope.

**Alternatives considered**:
- Transformer-XL (GTrXL): Better for long-range dependencies but heavier and more complex
- Temporal convolution: Fixed window, less flexible
- Attention over recent frames: No persistent state, loses long-range context

### 5. Curriculum: Fixed schedule with performance thresholds

**Decision**: Use a fixed curriculum schedule (8 → 16 → 32 → full horizon) with automatic progression when validation loss drops below a threshold for N consecutive epochs.

**Rationale**: Simple, predictable, and easy to tune. More sophisticated approaches (adaptive curriculum, self-paced learning) add complexity without clear benefit when the horizon stages are well-defined.

**Alternatives considered**:
- Adaptive curriculum based on learning progress: More complex, harder to reproduce
- Self-play curriculum: Requires environment interaction during training
- Automatic curriculum learning (ACL): Needs reward signals we don't have during offline training

### 6. Exploration: ICM-style prediction error

**Decision**: Use Intrinsic Curiosity Module (ICM) approach — train a forward model to predict next-state latent from current latent + action, use prediction error as intrinsic reward for data collection.

**Rationale**: ICM is well-established, simple to implement (reuses our existing predictor architecture), and directly measures "novelty" in the learned latent space. This aligns with our world model approach — states the model can't predict well are exactly where we need more training data.

**Alternatives considered**:
- RND (Random Network Distillation): Simpler but less aligned with world model
- Count-based exploration: Discretization issues in continuous state space
- Information gain: Computationally expensive

### 7. Safety: Ensemble-based uncertainty

**Decision**: Train an ensemble of 3-5 world models (shared encoder, separate predictors) and use variance across ensemble predictions as uncertainty. Discard candidate plans where cumulative uncertainty exceeds a threshold.

**Rationale**: Ensembles provide calibrated uncertainty estimates with minimal architectural changes. Dropout-based uncertainty (MC dropout) is cheaper but less reliable. The shared encoder keeps memory overhead manageable.

**Alternatives considered**:
- MC Dropout: Cheaper but less reliable uncertainty
- Evidential networks: More principled but significant architectural changes
- Conformal prediction: Distribution-free but requires calibration dataset

### 8. Module organization

**Decision**: New modules under existing package structure:
- `src/wally/models/recurrent_encoder.py` — LSTM wrapper around ViT encoder
- `src/wally/planner/subgoal_detector.py` — THICK-style change-point detection
- `src/wally/planner/high_level_planner.py` — CEM over abstract transitions
- `src/wally/planner/gradient_mpc.py` — Gradient refinement stage
- `src/wally/planner/hierarchical_planner.py` — Orchestrates high+low level
- `src/wally/training/curriculum.py` — Curriculum training loop
- `src/wally/training/curiosity.py` — ICM intrinsic reward
- `src/wally/training/ensemble.py` — Ensemble training and uncertainty

**Rationale**: Follows existing package conventions. Planner modules go in `planner/`, training utilities in `training/`, model components in `models/`.

## Risks / Trade-offs

- **[VRAM pressure]** Two LeWorldModels + LSTM + ensemble predictors may exceed 12GB. → Mitigation: Use smaller high-level model (fewer layers), gradient checkpointing, and freeze low-level model during high-level training.
- **[Subgoal quality]** THICK-style detection may produce noisy subgoals on diverse Minecraft data. → Mitigation: Add minimum segment length constraint, smooth prediction error with moving average, validate on hand-labeled trajectories.
- **[Training time]** Multi-phase training (5 phases) with curriculum will be very slow on single GPU. → Mitigation: Pre-train components independently, use smaller datasets for early phases, provide resume capability.
- **[Gradient instability]** Gradient-based MPC through autoregressive rollouts can diverge. → Mitigation: Gradient clipping, short refinement horizons (4-8 steps), warm-start from CEM to stay in good region.
- **[Integration complexity]** Seven new modules with cross-dependencies increase system complexity. → Mitigation: Protocol-based interfaces, comprehensive unit tests per module, integration tests for the full pipeline.
- **[Curriculum overfitting]** Model may overfit to short-horizon data in early curriculum stages. → Mitigation: Mix in longer sequences at each stage, use data augmentation, monitor validation metrics across all horizons.
