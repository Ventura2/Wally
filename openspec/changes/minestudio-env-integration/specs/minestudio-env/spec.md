## ADDED Requirements

### Requirement: Planner protocol abstraction
The system SHALL define a `PlannerProtocol` that abstracts over both flat and hierarchical planners, allowing the `AgentLoop` to work with either interchangeably.

#### Scenario: Flat planner satisfies protocol
- **WHEN** a `GoalConditionedPlanner` or `GradientMPC` is passed to the `AgentLoop`
- **THEN** the `AgentLoop` SHALL call `plan(current_frame, goal_frame)` and normalize the returned `torch.Tensor` into a `PlanResult` with `actions`, `success=True`, and `cost`

#### Scenario: Hierarchical planner satisfies protocol
- **WHEN** a `HierarchicalPlanner` is passed to the `AgentLoop`
- **THEN** the `AgentLoop` SHALL call `plan(current_frame, goal_frame)` and use the returned `HierarchicalPlanResult` directly, mapping its fields to the unified `PlanResult`

### Requirement: Unified plan result
The system SHALL define a `PlanResult` dataclass with fields: `actions` (torch.Tensor), `subgoals` (torch.Tensor | None), `success` (bool), `cost` (float), `replan_count` (int), and `low_confidence` (bool). The `AgentLoop` SHALL normalize all planner outputs into this type.

#### Scenario: Flat planner result normalization
- **WHEN** a flat planner returns a `torch.Tensor` of shape `(H, 25)`
- **THEN** the system SHALL wrap it as `PlanResult(actions=tensor, subgoals=None, success=True, cost=0.0, replan_count=0, low_confidence=False)`

#### Scenario: Hierarchical planner result pass-through
- **WHEN** a hierarchical planner returns a `HierarchicalPlanResult`
- **THEN** the system SHALL map its fields directly to `PlanResult` without transformation

### Requirement: MineStudio agent environment wrapper
The system SHALL provide a `MineStudioAgentEnv` class that wraps the MineStudio simulator and exposes `reset()`, `step()`, and `close()` methods suitable for agent-driven interaction.

#### Scenario: Reset returns preprocessed frame tensor
- **WHEN** `reset()` is called on a `MineStudioAgentEnv` instance
- **THEN** it SHALL return a `torch.Tensor` of shape `(3, H, W)` with pixel values normalized to `[0, 1]`

#### Scenario: Step accepts continuous action tensor
- **WHEN** `step(action)` is called with a 1-D `torch.Tensor` of shape `(25,)`
- **THEN** it SHALL convert the continuous action to a discrete MineStudio action dict using `MineStudioActionVocab`
- **THEN** it SHALL return a tuple of `(frame_tensor, reward, done, info)` where `frame_tensor` has shape `(3, H, W)`

#### Scenario: Close releases MineStudio resources
- **WHEN** `close()` is called
- **THEN** it SHALL call the underlying MineStudio simulator's `close()` method
- **THEN** subsequent calls to `step()` SHALL raise a `RuntimeError`

### Requirement: Agent configuration via Pydantic model
The system SHALL provide an `AgentConfig` Pydantic model with fields for: `replan_interval` (int, default 4), `episode_timeout` (int, default 1000 steps), `resize` (tuple, default (64, 64)), `action_vocab_path` (optional path), and `record_trajectory` (bool, default False).

#### Scenario: Load config from YAML
- **WHEN** `AgentConfig.from_yaml(path)` is called with a valid YAML file
- **THEN** it SHALL return an `AgentConfig` instance with values from the file, using defaults for missing fields

#### Scenario: Validate replan_interval
- **WHEN** `AgentConfig` is constructed with `replan_interval < 1`
- **THEN** it SHALL raise a `ValidationError`

#### Scenario: Validate episode_timeout
- **WHEN** `AgentConfig` is constructed with `episode_timeout < 1`
- **THEN** it SHALL raise a `ValidationError`

### Requirement: Plan-execute-observe loop
The system SHALL provide an `AgentLoop` class that orchestrates the planner and environment in a closed loop.

#### Scenario: Run episode with fixed-interval replanning
- **WHEN** `AgentLoop.run_episode(goal_frame)` is called
- **THEN** it SHALL call `planner.plan(current_frame, goal_frame)` to obtain an action sequence
- **THEN** it SHALL execute `replan_interval` actions from the plan via `env.step()`
- **THEN** it SHALL replan using the latest observed frame
- **THEN** it SHALL repeat until `done` is True or `episode_timeout` is reached

#### Scenario: Warm-start CEM from previous plan
- **WHEN** replanning occurs and a previous plan exists
- **THEN** the remaining actions from the previous plan SHALL be passed as the initial mean to the CEM optimizer

#### Scenario: Episode timeout
- **WHEN** the number of executed steps reaches `episode_timeout`
- **THEN** the episode SHALL terminate and the result SHALL indicate timeout

### Requirement: Safety bounds
The system SHALL enforce safety bounds during agent execution.

#### Scenario: Action clipping
- **WHEN** the planner produces continuous actions outside the `MineStudioActionVocab` bounds
- **THEN** the actions SHALL be clipped to `[low, high]` for each dimension before discretization

#### Scenario: Graceful shutdown on interrupt
- **WHEN** a `KeyboardInterrupt` or `SIGINT` is received during `run_episode()`
- **THEN** the agent SHALL call `env.close()` and return the partial trajectory collected so far

### Requirement: Trajectory recording
The system SHALL optionally record executed trajectories when `record_trajectory` is enabled.

#### Scenario: Record frames and actions during episode
- **WHEN** `record_trajectory` is True and an episode runs
- **THEN** the system SHALL buffer all observed frames and executed actions in memory
- **THEN** on episode end, the result SHALL include the recorded trajectory as a dict with `frames` (T, H, W, 3) and `actions` (T, 25) arrays

#### Scenario: No recording when disabled
- **WHEN** `record_trajectory` is False
- **THEN** the system SHALL NOT buffer frames or actions

### Requirement: wally-play CLI entry point
The system SHALL provide a `wally-play` CLI command registered in `pyproject.toml`.

#### Scenario: Run agent from CLI
- **WHEN** `wally-play --checkpoint <path> --goal-frame <path> --config <path>` is invoked
- **THEN** it SHALL load the LeWorldModel from the checkpoint
- **THEN** it SHALL load the goal frame from the specified image path
- **THEN** it SHALL create a `GoalConditionedPlanner` and `AgentLoop` and run one episode
- **THEN** it SHALL print episode statistics (steps, final cost, duration)

#### Scenario: CLI with trajectory recording
- **WHEN** `wally-play` is invoked with `--record --output-dir <path>`
- **THEN** it SHALL save the executed trajectory as a `.tar` shard to the output directory
