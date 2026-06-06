## Context

The Wally project has a planned pipeline of three changes:

1. `minecraft-trajectory-collection` — capture (observation, action) trajectories from a running MineStudio Minecraft instance and export them as WebDataset shards.
2. `lewm-training` — train a LeWorldModel-style latent dynamics model (ViT-Tiny encoder + Transformer predictor + SIGReg) on those shards.
3. `minecraft-latent-planner` (this change) — use the trained world model as a latent simulator and run Cross-Entropy Method (CEM) Model Predictive Control to convert a `(current_frame, goal_frame)` pair into an action sequence.

This change consumes the trained checkpoint produced by change 2 and does not retrain anything. The world model is treated as a frozen black box during planning. No GUI, telemetry, or service infra is in scope — this is a Python library + CLI addition to the `wally` package.

The reference algorithm is the standard latent-CEM recipe used in LeWorldModel and Dreamer-style work: sample candidate action sequences, roll the world model forward in latent space, score by distance from a goal latent, refit the sampling distribution to the elite candidates, and iterate.

## Goals / Non-Goals

**Goals:**
- Provide a reusable `mpc-cem-planner` (`wally.planner.cem`) that is independent of any specific world model — it consumes a cost callable.
- Provide a reusable `LatentRollout` (`wally.planner.rollout`) that wraps the trained `LeWorldModel` for `H`-step latent prediction.
- Provide a high-level `GoalConditionedPlanner` (`wally.planner.plan`) that exposes a single `plan(current_frame, goal_frame) -> action_sequence` interface, composes the frozen encoder + rollout + CEM, and is configured from YAML.
- Provide a `wally-plan` CLI supporting both offline `(current, goal)` frame pairs and live MineStudio rollouts.
- Add a self-contained smoke test using a toy linear-Gaussian dynamics model so the planner is testable without a trained Minecraft checkpoint.
- Stay within the existing `wally` package layout established by `lewm-training` (subpackage `wally/planner/`, sibling of `wally/models/`, `wally/data/`, `wally/training/`).

**Non-Goals:**
- Retraining, fine-tuning, or otherwise modifying the world model.
- Differentiable planning (gradient-based trajectory optimization). CEM is the only optimizer in scope. A `straight_through` rollout option is exposed for future work but no gradient-based planner is implemented.
- Discrete action search (e.g., Gumbel-softmax CEM). Continuous-bounded CEM is the only search method. A discrete adapter is provided as a post-processing step.
- A learned policy network or value function. This is purely model-based planning.
- Distributed/multi-GPU CEM. Single-process, single-device (with batched population) is sufficient.
- A web UI, dashboard, or visualization tool. CLI only.

## Decisions

### Decision: CEM over a learned policy
- **Choice:** Cross-Entropy Method over bounded continuous action sequences.
- **Rationale:** CEM is the de-facto baseline for latent MPC with a learned dynamics model. It is sample-efficient for short horizons, requires no value function, and the world model is already differentiable-enough to evaluate candidate trajectories cheaply. Adding a learned policy is a much larger commitment and is not needed to validate the world model.
- **Alternatives considered:**
  - Random shooting — weaker baseline; included only as a sanity check in the smoke test.
  - MPPI (Model Predictive Path Integral) — theoretically nicer, but adds temperature/sample-weight hyperparameters with no clear win over CEM at our horizon/dimension.
  - Learned policy / value function — defers the real evaluation; we want to evaluate the world model first.

### Decision: Two-tier architecture (reusable CEM, dedicated planner)
- **Choice:** Split the implementation into a generic `CEMOptimizer` (cost-agnostic) and a `GoalConditionedPlanner` that wires it to the world model.
- **Rationale:** A cost-agnostic CEM is testable in isolation (linear-Gaussian toy, smoke test) and can be reused for other latent-cost formulations later (e.g. goal-progress shaping, value-function-based costs). The high-level planner is a thin composition layer.
- **Alternatives considered:**
  - Single monolithic planner class — simpler but un-testable without a trained checkpoint, and harder to extend.

### Decision: Latent rollout as a separate capability
- **Choice:** `LatentRollout` is its own module, not a private method on the planner.
- **Rationale:** The rollout is the only piece of the planner that depends on the world model internals. Keeping it separate lets the planner be tested with a stand-in dynamics model and lets future capabilities (imagination-based RL, evaluation rollouts) reuse it without going through CEM.
- **Alternatives considered:**
  - Rollout as a private method on `GoalConditionedPlanner` — couples planning to world-model internals and prevents reuse.

### Decision: World model parameters are frozen at load time
- **Choice:** The planner loads the world model checkpoint, sets `requires_grad=False` on every parameter, and never mutates it.
- **Rationale:** Planning is a read-only consumer of the world model. Allowing parameter mutation would (a) be a footgun, (b) cause silent GPU memory growth from optimizer state, and (c) make planning non-deterministic w.r.t. checkpoint hashes.
- **Alternatives considered:**
  - Trust the caller to freeze the model — fragile, easy to get wrong.

### Decision: Default cost is squared L2 in latent space
- **Choice:** `cost(z_H, z_g) = ||z_H - z_g||^2` is the default; pluggable via a callable.
- **Rationale:** Squared L2 is the simplest reasonable proxy for "reach the goal latent." It is convex in the rollout, makes CEM elites well-defined, and matches the LeWorldModel/Dreamer convention.
- **Alternatives considered:**
  - Cosine distance — would require normalization guarantees we don't have.
  - Learned reward model — out of scope; we have no reward data.

### Decision: Population on GPU, single process
- **Choice:** The CEM population is materialized as a single `(population_size, H, A)` tensor on the planning device, and the rollout cost is computed in one batched `LatentRollout` call.
- **Rationale:** Batched rollout is the cheapest way to score candidates — one forward pass through the Transformer predictor for all `population_size` trajectories — and avoids the per-candidate Python overhead that would dominate at `population_size=64, H=8`.
- **Alternatives considered:**
  - Per-candidate Python loop — would be 64x slower; not viable.
  - Multi-GPU sharding — premature; the population is small enough to fit on a single GPU.

### Decision: Action adapter is post-hoc quantization, not part of search
- **Choice:** CEM searches in bounded continuous space. Conversion to MineStudio's discrete action vocabulary is a deterministic post-processing step.
- **Rationale:** Searching directly in the discrete space would require either Gumbel-softmax relaxation (extra hyperparameters) or combinatorial sampling (exponential in `H*D`). Continuous search with post-hoc quantization is the standard simplification and is sufficient for first-pass results. A future "discrete-relaxed CEM" can be added without changing the planner API.
- **Alternatives considered:**
  - Gumbel-softmax CEM — added complexity with no clear win at our horizon.
  - Combinatorial enumeration — infeasible at `H=8, D=12`.

### Decision: Configuration via Pydantic + YAML
- **Choice:** `CEMConfig` is a Pydantic `BaseModel` (or `dataclass(frozen=True)` if Pydantic is not yet a dependency) with `from_yaml` classmethod.
- **Rationale:** Pydantic gives us free validation, clear error messages, and IDE completion. The `lewm-training` change already uses PyYAML, so the YAML format itself is reused. A pure dataclass is the fallback if Pydantic is not yet in `pyproject.toml`.
- **Alternatives considered:**
  - Plain dict — no validation, hard to refactor.
  - OmegaConf / Hydra — heavier dependency than needed for one config class.

### Decision: CLI uses argparse, not Click or Typer
- **Choice:** `argparse` for the `wally-plan` CLI.
- **Rationale:** Zero new dependencies, matches the standard-library baseline used by the rest of the `wally` package, and the CLI surface is small (4-5 flags).
- **Alternatives considered:**
  - Click — adds a dependency for marginal ergonomics.
  - Typer — same.

### Decision: Smoke test uses a synthetic linear-Gaussian world model
- **Choice:** The smoke test instantiates a 2-layer linear "world model" (a single `nn.Linear`) and verifies the planner on that stand-in.
- **Rationale:** Lets the planner be CI-tested without a 24h training run. The test asserts shape, bounds, and cost-decrease properties that are world-model-agnostic. End-to-end testing with a real checkpoint is deferred to the `evaluation` change.
- **Alternatives considered:**
  - Skip smoke test entirely — would leave planner correctness uncovered until `evaluation` lands.
  - Mock the world model — would couple the test to implementation details.

## Risks / Trade-offs

- **[Risk] CEM is brittle to action-space scaling.** If the action bounds are too wide, the search distribution dilutes; too narrow and the planner is blind to good actions. → **Mitigation:** The action bounds are part of the YAML config and we ship conservative defaults derived from MineStudio's action ranges. The smoke test verifies bounds enforcement. The `evaluation` change will sweep these.
- **[Risk] Latent cost may not correlate with task success.** A small latent distance does not guarantee the agent reaches the visible goal (e.g. a chest that looks similar from above vs. the side). → **Mitigation:** The cost function is pluggable; future work can swap in a learned reward or a task-specific distance. The `evaluation` change will measure this gap.
- **[Risk] CEM is compute-heavy at plan time.** A single `plan()` call costs `population_size * horizon` forward passes through the Transformer predictor. At default `population_size=64, H=8` this is 512 forward passes per plan call. → **Mitigation:** Defaults are conservative. The CLI logs iteration cost. A future change can add action-sequence reuse across plan calls (receding horizon / warm-starting the CEM mean from the previous plan).
- **[Risk] MineStudio action vocabulary is not yet stable.** Discrete↔continuous adapter code is coupled to the vocabulary exposed by MineStudio at the time of writing. → **Mitigation:** The adapter takes the vocabulary as a config argument; vocabulary changes only require updating the config, not the planner code.
- **[Risk] Frozen encoder is a single point of failure.** If the encoder produces low-quality latents for out-of-distribution goal frames (e.g. novel biomes), the planner will target the wrong latent. → **Mitigation:** Out of scope for this change; the world model itself is evaluated by the `evaluation` change.

## Migration Plan

Not applicable — this is a net-new subpackage. There is no existing planner to migrate from. On rollout, the planner is opt-in: nothing in `lewm-training` or `minecraft-trajectory-collection` depends on it, and the CLI is a new entry point that does not affect existing scripts.

## Open Questions

- Should the planner expose a `warm_start` hook so that subsequent `plan()` calls can be initialized from the previous plan's action sequence (receding-horizon control)? Deferred — would change the API surface, not needed for first-pass results.
- Should the cost function support a discount factor across the rollout trajectory, or only score the final latent? Current default scores only the final latent; a `cost_trajectory(z_trajectory, z_g)` variant is a possible future extension.
- Should the CLI support batched planning (plan for multiple `(current, goal)` pairs at once)? Deferred — single-pair planning is the common case; batched planning can be added as a thin wrapper without changing the planner class.
