## Context

The wally project has a trained LeWorldModel, CEM and gradient-based planners, and a hierarchical planner. The `src/agent/` package has scaffolding (`MineStudioAgentEnv`, `AgentConfig`, `TrajectoryBuffer`) but no orchestration layer to close the plan-execute-observe loop. The existing planner interfaces are heterogeneous: flat planners return `Tensor | (Tensor, float)`, while the hierarchical planner returns `HierarchicalPlanResult`.

## Goals / Non-Goals

**Goals:**
- Unified `PlannerProtocol` so `AgentLoop` works with any planner type
- `AgentLoop` that runs a single episode with fixed-interval replanning
- `wally-play` CLI for one-shot episode execution from a checkpoint
- Full test coverage using mock env and mock planner

**Non-Goals:**
- Multi-episode evaluation or benchmarking (covered by evaluation spec)
- Text or latent goal specification (deferred to future work)
- Real-time visualization or streaming of agent behavior
- Modification of existing planner internals

## Decisions

### 1. PlannerProtocol via adapter wrappers, not modifying existing planners

**Decision**: Create `PlannerProtocol` as a `runtime_checkable` Protocol with a single `plan(current_frame, goal_frame) -> PlanResult` method. Wrap existing planners in thin adapter classes (`FlatPlannerAdapter`, `HierarchicalPlannerAdapter`) that normalize outputs.

**Rationale**: The existing `GoalConditionedPlanner.plan()` and `GradientMPC.plan()` have a `return_cost` kwarg and return union types. Modifying their signatures would break existing CLI entry points (`wally-plan`, `wally-plan-hierarchical`). Adapters keep the agent loop clean without touching working code.

**Alternative considered**: Add a `plan_unified()` method to each planner. Rejected because it couples planner code to agent concerns and requires changes across three planner classes.

### 2. PlanResult as a frozen dataclass

**Decision**: `PlanResult` is a `@dataclass(frozen=True)` with fields: `actions: Tensor`, `subgoals: Tensor | None`, `success: bool`, `cost: float`, `replan_count: int`, `low_confidence: bool`.

**Rationale**: Frozen dataclass prevents accidental mutation during the loop. Mirrors the existing `HierarchicalPlanResult` shape so the hierarchical adapter is a trivial copy. Flat adapters fill defaults (`subgoals=None`, `replan_count=0`).

### 3. AgentLoop as a stateful class with `run_episode()` method

**Decision**: `AgentLoop` takes env, planner, config, and optional buffer in `__init__`. Single `run_episode(goal_frame) -> EpisodeResult` method drives the full loop. `EpisodeResult` contains steps, final cost, duration, and optional trajectory dict.

**Rationale**: Stateful class allows future extension (multi-episode, pause/resume) without restructuring. Keeping `run_episode` as a single method makes the loop easy to test and reason about.

### 4. Replanning via step counter, not plan-tail tracking

**Decision**: Replan every `replan_interval` steps. On replan, pass the current observation frame and the goal frame. For flat planners with CEM, warm-start from the tail of the previous plan (shift actions left by `replan_interval`).

**Rationale**: Step-counter is simple and predictable. Plan-tail tracking (replan when remaining actions < threshold) adds complexity without clear benefit at this stage. Warm-start gives CEM faster convergence on replan.

### 5. CLI in `src/agent/play.py`, registered in pyproject.toml

**Decision**: `wally-play` entry point in `src/agent/play.py` with `argparse`. Arguments: `--checkpoint`, `--goal-frame`, `--config`, `--record`, `--output-dir`, `--planner` (choice of `cem`, `gradient`, `hierarchical`).

**Rationale**: Follows the existing CLI pattern (`wally-plan`, `wally-collect`). The `--planner` flag lets users select which planner to construct without code changes.

## Risks / Trade-offs

- **[Risk] MineStudio not available in test/CI** → Mitigation: All tests use mock env and mock planner. `MineStudioAgentEnv` already handles `ImportError` gracefully.
- **[Risk] Warm-start CEM with shifted actions may diverge if environment state changes significantly** → Mitigation: Warm-start is optional; `AgentConfig` could add a `warm_start` flag later. For now, always warm-start as the spec requires it.
- **[Risk] Hierarchical planner produces variable-length action sequences** → Mitigation: `AgentLoop` consumes actions sequentially and replans based on step counter regardless of plan length. If the plan is shorter than `replan_interval`, replan immediately after exhausting actions.
- **[Trade-off] Frozen PlanResult means copying tensors for warm-start** → Acceptable overhead given the small tensor sizes (H x 25).
