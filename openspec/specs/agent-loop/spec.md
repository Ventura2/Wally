# agent-loop Specification

## Purpose
TBD - created by archiving change minestudio-agent-loop. Update Purpose after archive.
## Requirements
### Requirement: PlannerProtocol abstraction
The system SHALL provide a `PlannerProtocol` as a `runtime_checkable` Protocol with a `plan(current_frame: Tensor, goal_frame: Tensor) -> PlanResult` method. Any planner satisfying this protocol SHALL be usable by `AgentLoop` without type-specific branching.

#### Scenario: Flat planner satisfies protocol
- **WHEN** a `GoalConditionedPlanner` is wrapped in `FlatPlannerAdapter`
- **THEN** calling `plan(current_frame, goal_frame)` SHALL return a `PlanResult` with `actions` of shape `(H, 25)`, `subgoals=None`, `success=True`, `replan_count=0`, and `low_confidence=False`

#### Scenario: GradientMPC satisfies protocol
- **WHEN** a `GradientMPC` is wrapped in `FlatPlannerAdapter`
- **THEN** calling `plan(current_frame, goal_frame)` SHALL return a `PlanResult` with `actions` of shape `(H, 25)`, `subgoals=None`, `success=True`, `replan_count=0`, and `low_confidence=False`

#### Scenario: Hierarchical planner satisfies protocol
- **WHEN** a `HierarchicalPlanner` is wrapped in `HierarchicalPlannerAdapter`
- **THEN** calling `plan(current_frame, goal_frame)` SHALL return a `PlanResult` with fields mapped from `HierarchicalPlanResult`: `actions`, `subgoals`, `success`, `cost`, `replan_count`, `low_confidence`

### Requirement: PlanResult dataclass
The system SHALL provide a `PlanResult` frozen dataclass with fields: `actions: Tensor`, `subgoals: Tensor | None`, `success: bool`, `cost: float`, `replan_count: int`, `low_confidence: bool`.

#### Scenario: PlanResult is immutable
- **WHEN** a `PlanResult` instance is created
- **THEN** attempts to modify any field SHALL raise `FrozenInstanceError`

#### Scenario: PlanResult default fields
- **WHEN** a `PlanResult` is constructed with only `actions` and `cost` provided
- **THEN** `subgoals` SHALL default to `None`, `success` to `True`, `replan_count` to `0`, and `low_confidence` to `False`

### Requirement: AgentLoop plan-execute-observe cycle
The system SHALL provide an `AgentLoop` class that runs a single episode via `run_episode(goal_frame) -> EpisodeResult`. The loop SHALL: (1) encode the goal frame to latent once, (2) invoke the planner to get an action sequence, (3) execute actions step-by-step via the environment, (4) replan at fixed intervals.

#### Scenario: Episode runs to timeout
- **WHEN** the environment never signals `done=True` and `episode_timeout` is 100
- **THEN** `run_episode` SHALL return an `EpisodeResult` with `steps=100` after executing exactly 100 actions

#### Scenario: Episode ends on environment done signal
- **WHEN** the environment returns `done=True` at step 42
- **THEN** `run_episode` SHALL return an `EpisodeResult` with `steps=42` and SHALL NOT execute further actions

#### Scenario: Planner invoked at episode start
- **WHEN** `run_episode` is called
- **THEN** the planner SHALL be invoked before any actions are executed, using the initial observation frame and the goal frame

#### Scenario: Replanning at fixed interval
- **WHEN** `replan_interval` is 4 and the episode runs for 12 steps
- **THEN** the planner SHALL be invoked at steps 0, 4, and 8 (3 total invocations)

#### Scenario: Warm-start CEM on replan
- **WHEN** a flat planner with CEM is replanned at step N
- **THEN** the adapter SHALL pass the tail of the previous plan (actions from step N onward) as warm-start mean to the underlying planner

#### Scenario: Actions exhausted before replan interval
- **WHEN** the planner returns 3 actions but `replan_interval` is 8
- **THEN** after executing all 3 actions, the planner SHALL be invoked again immediately

### Requirement: Graceful shutdown on KeyboardInterrupt
The system SHALL catch `KeyboardInterrupt` during episode execution, close the environment, and return a partial `EpisodeResult` with the steps completed so far.

#### Scenario: Interrupt mid-episode
- **WHEN** `KeyboardInterrupt` is raised during `env.step()` at step 15
- **THEN** `run_episode` SHALL close the environment and return an `EpisodeResult` with `steps=15` and `interrupted=True`

### Requirement: Trajectory recording integration
The system SHALL buffer frames and actions in a `TrajectoryBuffer` when `AgentConfig.record_trajectory` is `True`. The buffer SHALL be included in the `EpisodeResult`.

#### Scenario: Recording enabled
- **WHEN** `record_trajectory=True` and an episode runs for 10 steps
- **THEN** `EpisodeResult.trajectory` SHALL contain a dict with `frames` of shape `(10, H, W, 3)` and `actions` of shape `(10, 25)`

#### Scenario: Recording disabled
- **WHEN** `record_trajectory=False`
- **THEN** `EpisodeResult.trajectory` SHALL be `None` and no buffering SHALL occur

### Requirement: EpisodeResult dataclass
The system SHALL provide an `EpisodeResult` dataclass with fields: `steps: int`, `final_cost: float`, `duration_seconds: float`, `trajectory: dict | None`, `interrupted: bool`.

#### Scenario: Normal episode result
- **WHEN** an episode completes normally
- **THEN** `interrupted` SHALL be `False` and `duration_seconds` SHALL reflect wall-clock time from start to end

#### Scenario: Interrupted episode result
- **WHEN** an episode is interrupted by `KeyboardInterrupt`
- **THEN** `interrupted` SHALL be `True` and `steps` SHALL reflect the number of actions executed before interruption

### Requirement: wally-play CLI entry point
The system SHALL provide a `wally-play` CLI with arguments: `--checkpoint` (path to LeWorldModel checkpoint, required), `--goal-frame` (path to goal image, required), `--config` (path to AgentConfig YAML, optional), `--record` (flag to enable trajectory recording), `--output-dir` (directory for trajectory export, default `.`), `--planner` (choice of `cem`, `gradient`, `hierarchical`, default `cem`).

#### Scenario: Run with CEM planner
- **WHEN** `wally-play --checkpoint model.pt --goal-frame goal.png --planner cem` is executed
- **THEN** the CLI SHALL load the checkpoint, construct a `GoalConditionedPlanner` via `FlatPlannerAdapter`, create `MineStudioAgentEnv` and `AgentLoop`, run one episode, and print episode statistics (steps, final cost, duration)

#### Scenario: Run with trajectory recording
- **WHEN** `wally-play --checkpoint model.pt --goal-frame goal.png --record --output-dir ./out` is executed
- **THEN** the CLI SHALL export the trajectory as a numpy dict to `./out/episode_0.npz`

#### Scenario: Invalid checkpoint path
- **WHEN** `wally-play --checkpoint nonexistent.pt --goal-frame goal.png` is executed
- **THEN** the CLI SHALL exit with a non-zero code and an error message

#### Scenario: Config from YAML
- **WHEN** `wally-play --checkpoint model.pt --goal-frame goal.png --config agent.yaml` is executed
- **THEN** the CLI SHALL load `AgentConfig` from the YAML file and use its values for the episode

