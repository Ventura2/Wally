## Context

The project has a working CEM-based MPC planner (`src/wally/planner/`) that takes a current frame and goal frame, then produces an action sequence via latent-space optimization. The collector (`src/collector/`) already wraps MineStudio's `MinecraftSim` for trajectory collection. What's missing is the closed-loop agent: an execution harness that repeatedly plans, acts, observes, and replans in a live Minecraft environment.

Key existing interfaces:
- `GoalConditionedPlanner.plan(current_frame, goal_frame) -> actions` — returns a (H, 25) tensor of continuous actions
- `MineStudioEnv.step(action_dict) -> (frame, reward, done, info)` — gym-like step interface
- `MineStudioActionVocab` + `continuous_to_discrete()` — converts planner output to MineStudio action dicts
- `CEMConfig` — planner hyperparameters (horizon, population, iterations)

## Goals / Non-Goals

**Goals:**
- Build a `MineStudioAgentEnv` wrapper that provides a clean interface for agent-environment interaction with proper episode lifecycle
- Implement the plan-execute-observe loop with configurable replanning interval
- Add safety bounds: action clipping, episode timeout, and graceful shutdown on interrupt
- Provide a `wally-play` CLI to run the agent with a trained checkpoint against a live Minecraft instance
- Record executed trajectories for later analysis and replay

**Non-Goals:**
- Multi-agent or multiplayer support
- Learning or fine-tuning during execution (this is inference-only)
- Text-based or latent-based goal specification (frame goals only for now)
- Real-time visualization or GUI overlay

## Decisions

### 1. Package location: `src/agent/`

**Decision**: New top-level package `src/agent/` alongside `collector/`, `exporter/`, `validator/`, `wally/`.

**Rationale**: The agent is a distinct capability (deploy + execute) separate from training (`wally/`) and collection (`collector/`). Keeping it top-level follows the existing project structure where each package maps to a capability.

**Alternatives considered**:
- Nesting under `src/wally/agent/` — rejected because the agent is a deployment concern, not a training concern
- Extending `src/collector/` — rejected because collection and agent execution have different lifecycles and configs

### 2. Replanning strategy: fixed-interval with warm-start

**Decision**: Replan every `replan_interval` steps (default: `horizon // 2`). When replanning, shift the previous plan's remaining actions as the initial mean for CEM.

**Rationale**: Fixed-interval is simple and predictable. Warm-starting CEM from the previous plan's tail accelerates convergence since the world model's predictions should be close to reality for near-future steps.

**Alternatives considered**:
- Replan every step — too slow (CEM with 64 population × 5 iterations = 320 rollouts per step)
- Divergence-triggered replanning — requires a reliable divergence metric, which is hard to define in latent space; can be added later as an enhancement

### 3. Env wrapper: thin adapter over existing `MineStudioEnv`

**Decision**: `MineStudioAgentEnv` wraps `MineStudioEnv` (from collector) and adds: frame preprocessing (resize, normalize to tensor), action postprocessing (continuous → discrete), and episode timeout tracking.

**Rationale**: Reusing the existing `MineStudioEnv` avoids duplicating the MineStudio integration. The agent env adds only what's needed for the planner loop.

### 4. Goal specification: frame-based only

**Decision**: Goals are specified as a single RGB frame (numpy array or tensor). The agent encodes it once at episode start and reuses the latent.

**Rationale**: The planner already operates on frame pairs. Text and latent goals require additional infrastructure (text encoder, latent sampler) that doesn't exist yet. Frame goals are sufficient for the initial integration and can be extended later.

### 5. Trajectory recording: in-memory with optional shard export

**Decision**: Buffer frames and actions in memory during an episode. On episode end, optionally write to a `.tar` shard using the existing `RawShardWriter` pattern.

**Rationale**: In-memory is simple for single episodes. Shard export enables collecting agent performance data without a separate recording pass.

### 6. Config: Pydantic model with YAML loading

**Decision**: `AgentConfig` as a Pydantic `BaseModel` with `from_yaml()` classmethod, following the pattern of `CEMConfig` and `CollectorConfig`.

**Rationale**: Consistent with existing config patterns. Pydantic provides validation and defaults.

## Risks / Trade-offs

- **[CEM latency]** Each plan call runs 320 rollouts. At ~50ms per rollout on GPU, a single plan takes ~16s. With replan_interval=4, the agent plans every 4 env steps. → Mitigation: reduce population_size or n_iterations for faster (lower-quality) plans; document latency expectations.
- **[World model drift]** The LeWorldModel's predictions diverge from reality over long horizons. Replanning mitigates this but doesn't eliminate it. → Mitigation: keep replan_interval short; log prediction error metrics for monitoring.
- **[MineStudio stability]** MineStudio may crash or hang during long episodes. → Mitigation: episode timeout with graceful shutdown; catch and log MineStudio exceptions; `close()` always called in finally block.
- **[Action discretization loss]** Converting continuous planner output to discrete MineStudio actions loses precision, especially for camera control. → Mitigation: use finer bin counts for camera dimensions (11 bins vs 2 for buttons); this is already handled by `MineStudioActionVocab`.
