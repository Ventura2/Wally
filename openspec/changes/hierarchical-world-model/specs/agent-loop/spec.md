## MODIFIED Requirements

### Requirement: AgentLoop plan-execute-observe cycle
The system SHALL provide an `AgentLoop` class that runs a single episode via `run_episode(goal_frame, target_embedding=None) -> EpisodeResult`. The loop SHALL: (1) encode the goal frame to latent once (when `target_embedding` is `None`), (2) invoke the planner to get an action sequence, (3) execute actions step-by-step via the environment, (4) replan at fixed intervals, (5) when a hierarchical planner is in use, stream the current state embedding upward and receive a target embedding downward.

#### Scenario: Episode runs to timeout
- **WHEN** the environment never signals `done=True` and `episode_timeout` is 100
- **THEN** `run_episode` SHALL return an `EpisodeResult` with `steps=100` after executing exactly 100 actions

#### Scenario: Episode ends on environment done signal
- **WHEN** the environment returns `done=True` at step 42
- **THEN** `run_episode` SHALL return an `EpisodeResult` with `steps=42` and SHALL NOT execute further actions

#### Scenario: Planner invoked at episode start
- **WHEN** `run_episode` is called
- **THEN** the planner SHALL be invoked before any actions are executed, using the initial observation frame and the goal frame (or target embedding)

#### Scenario: Replanning at fixed interval
- **WHEN** `replan_interval` is 4 and the episode runs for 12 steps
- **THEN** the planner SHALL be invoked at steps 0, 4, and 8 (3 total invocations)

#### Scenario: Warm-start CEM on replan
- **WHEN** a flat planner with CEM is replanned at step N
- **THEN** the adapter SHALL pass the tail of the previous plan (actions from step N onward) as warm-start mean to the underlying planner

#### Scenario: Actions exhausted before replan interval
- **WHEN** the planner returns 3 actions but `replan_interval` is 8
- **THEN** after executing all 3 actions, the planner SHALL be invoked again immediately

#### Scenario: Hierarchical planner streams state upward
- **WHEN** a `HierarchicalPlannerAdapter` is in use and the L0 planner completes one step
- **THEN** the agent loop SHALL push the new L0 state embedding to the L1 layer's input queue

#### Scenario: Hierarchical planner receives target downward
- **WHEN** a `HierarchicalPlannerAdapter` is in use and L1 has produced a new target embedding
- **THEN** the L0 CEM planner SHALL receive the new target embedding as its `target_embedding` argument on the next replan

### Requirement: wally-play CLI entry point
The system SHALL provide a `wally-play` CLI with arguments: `--checkpoint` (path to LeWorldModel checkpoint, required), `--goal-frame` (path to goal image, optional when `--target-embedding` is provided), `--target-embedding` (path to a `.pt` file containing a goal embedding tensor, optional when `--goal-frame` is provided), `--config` (path to AgentConfig YAML, optional), `--record` (flag to enable trajectory recording), `--output-dir` (directory for trajectory export, default `.`), `--planner` (choice of `cem`, `gradient`, `hierarchical`, `hierarchical-embedding`, default `cem`), `--layer-depth` (int, default 0 — number of additional hierarchy layers above L0; 0 disables the hierarchy).

#### Scenario: Run with CEM planner (unchanged)
- **WHEN** `wally-play --checkpoint model.pt --goal-frame goal.png --planner cem` is executed
- **THEN** the CLI SHALL load the checkpoint, construct a `GoalConditionedPlanner` via `FlatPlannerAdapter`, create `MineStudioAgentEnv` and `AgentLoop`, run one episode, and print episode statistics

#### Scenario: Run with hierarchical-embedding planner
- **WHEN** `wally-play --checkpoint model.pt --target-embedding g3.pt --planner hierarchical-embedding --layer-depth 3 --hierarchy-checkpoint hierarchy.pt` is executed
- **THEN** the CLI SHALL load the L0 checkpoint and the hierarchy checkpoint, construct a `HierarchicalEmbeddingPlannerAdapter` with `layer_depth=3`, and run one episode in which L1/L2/L3 are active

#### Scenario: Missing both goal frame and target embedding
- **WHEN** `wally-play --checkpoint model.pt` is executed without `--goal-frame` or `--target-embedding`
- **THEN** the CLI SHALL exit with a non-zero code and an error message identifying the missing argument
