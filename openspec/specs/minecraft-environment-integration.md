## MineStudio Environment Integration

Purpose:
- Bridge between planner and MineStudio Minecraft environment
- Enable real-time agent execution in a closed plan-execute-observe loop
- Support both flat (CEM/GradientMPC) and hierarchical planning modes

Components:
- `MineStudioAgentEnv`: Gym-like wrapper over `MineStudioEnv` with frame preprocessing (resize, normalize to tensor), action postprocessing (continuous -> discrete via `MineStudioActionVocab`), and lifecycle tracking
- `AgentLoop`: Plan-execute-observe loop that orchestrates planner and environment
- `PlannerProtocol`: Abstraction over flat and hierarchical planners, normalizing outputs into a unified `PlanResult`
- `TrajectoryBuffer`: In-memory frame/action accumulator with optional `.tar` shard export
- `AgentConfig`: Pydantic config with `replan_interval`, `episode_timeout`, `resize`, `action_vocab_path`, `record_trajectory`
- `wally-play` CLI entry point for running the agent against a live MineStudio instance

### Planner Abstraction

The `AgentLoop` accepts any planner satisfying `PlannerProtocol`. Both flat and hierarchical planners implement this protocol, and their outputs are normalized into a unified `PlanResult`:

**Flat planners** (`GoalConditionedPlanner`, `GradientMPC`):
- Return `Tensor` of shape `(H, 25)` from `plan()`
- Normalized to: `PlanResult(actions=tensor, subgoals=None, success=True, cost=float, replan_count=0, low_confidence=False)`

**Hierarchical planner** (`HierarchicalPlanner`):
- Returns `HierarchicalPlanResult` from `plan()`
- Mapped directly to `PlanResult` with subgoals, replan_count, success, low_confidence fields

The `AgentLoop` doesn't need to know which planner type it's using — it just calls `plan()` and works with the normalized result.

### Replanning Strategy
- Fixed-interval replanning: re-invoke planner every `replan_interval` steps (default: `horizon // 2`)
- At episode start: planner invoked immediately before any actions execute
- Warm-start CEM from previous plan tail for faster convergence
- Hierarchical mode: subgoal-driven replanning with automatic retry on failure

### Action Execution
- Planner produces continuous `(25,)` action vectors
- Actions clipped to `MineStudioActionVocab` bounds `[low, high]` per dimension before discretization
- Converted to discrete action dicts via `continuous_to_discrete()`
- Executed sequentially via `env.step()`, observing resulting frame after each step

### Episode Termination
- Environment signals `done=True` (terminated or truncated)
- Step count reaches `episode_timeout` (default: 1000)
- `KeyboardInterrupt` received (graceful shutdown: close env, return partial trajectory)

### Goal Specification
- Frame-based goals only (RGB numpy array or tensor of shape `(H, W, 3)` or `(3, H, W)`)
- Goal frame encoded to latent once at episode start, reused for all planning calls
- Text and latent goals deferred to future work

### Trajectory Recording
- When `record_trajectory=True`: buffer all observed frames and executed actions in memory
- On episode end: export as dict with `frames` `(T, H, W, 3)` and `actions` `(T, 25)` numpy arrays
- Optional `.tar` shard export via `RawShardWriter` pattern
- When `record_trajectory=False`: no buffering overhead

### CLI: `wally-play`
- Arguments: `--checkpoint`, `--goal-frame`, `--config`, `--record`, `--output-dir`
- Loads LeWorldModel from checkpoint, extracts encoder and latent rollout for planner construction
- Creates `MineStudioAgentEnv`, `AgentLoop`, runs one episode
- Prints episode statistics: steps taken, final latent cost, wall-clock duration
- Optional trajectory export when `--record` is set

Input:
- Trained LeWorldModel checkpoint
- Goal frame (RGB image path)
- AgentConfig YAML (optional, uses defaults)

Output:
- Executed trajectory (frames + actions) when recording enabled
- Episode statistics (steps, cost, duration)
- Optional `.tar` shard file

### Live POV Frame Exposure

The raw full-resolution frame (640×360) returned by the underlying
`MinecraftSim` is plumbed through the env stack as `info["pov"]` so callers
(such as the live viewer in `live-agent-viewer`) can render it without
forcing every consumer of the agent's preprocessed tensor to also pay for
the larger image.

### Requirement: Live POV frame is exposed by the environment
The `MineStudioEnv.step()` return tuple MUST include a `pov` key in the
`info` dict containing the full-resolution first-person frame (shape
`(360, 640, 3)`, dtype `uint8`, BGR or RGB consistent with MineStudio's
native obs dict) returned by the underlying `MinecraftSim` observation.
The existing 224×224 `image` frame in the env's return tuple MUST
continue to be returned unchanged for the agent loop's preprocessing.

#### Scenario: POV frame is plumbed through env step
- **WHEN** `MineStudioEnv.step(action)` is called and the underlying
  `MinecraftSim` returns `obs_dict` containing both `pov` and `image`
- **THEN** the returned `info` dict contains a `pov` key with the
  full-resolution frame and the function's other return values
  (`image`, `reward`, `done`) match the prior behavior

#### Scenario: Agent env surfaces POV in info
- **WHEN** `MineStudioAgentEnv.step(action)` is called and the
  underlying `MineStudioEnv` returns `info` containing `pov`
- **THEN** the agent env's returned `info` dict contains the same
  `pov` key and value, untouched by frame preprocessing

### AgentLoop Viewer Integration

`AgentLoop` is viewer-agnostic: it accepts an optional `viewer` argument
and calls `viewer.show(...)` and `viewer.should_quit()` after each
environment step when one is provided. The viewer is a passive observer
and never influences planner behavior or step selection.

### Requirement: AgentLoop supports an optional viewer
`AgentLoop.run_episode()` MUST accept an optional `viewer` argument.
When a viewer is provided, the loop MUST call `viewer.show(pov, info)`
after each `env.step()` and MUST call `viewer.should_quit()` to detect
user-initiated shutdown. The viewer's return values MUST NOT alter the
agent's per-step action selection or planner behavior — the viewer is
purely a passive observer.

#### Scenario: Viewer is invoked per step
- **WHEN** the loop executes an environment step and a viewer is set
- **THEN** `viewer.show(...)` is called with the current observation
  and info dict, and the loop continues to the next step

#### Scenario: Viewer can trigger clean episode shutdown
- **WHEN** `viewer.should_quit()` returns `True` after a step
- **THEN** the loop closes the environment, returns an
  `EpisodeResult` with `interrupted=True` populated, and exits without
  raising

#### Scenario: Loop works without a viewer
- **WHEN** `AgentLoop.run_episode()` is called with `viewer=None`
  (the default)
- **THEN** the loop runs exactly as before — no viewer is invoked, no
  quit check is performed, and the return type is unchanged

### Requirement: wally-play CLI exposes viewer selection
`wally-play` MUST accept a `--viewer` flag with values `cv2` (default)
and `none`. When `--viewer cv2` is set, `wally-play` MUST instantiate
a `FrameViewer` and pass it to `AgentLoop.run_episode()`. When
`--viewer none` is set, `wally-play` MUST run with `viewer=None` and
MUST NOT import `cv2`. A convenience `--no-viewer` alias MUST be
accepted as equivalent to `--viewer none`.

#### Scenario: Default viewer is cv2
- **WHEN** the user runs `wally-play --checkpoint ... --goal-frame ...`
  with no `--viewer` flag
- **THEN** a `FrameViewer` is constructed and passed to the agent loop
  and a POV window appears during the episode

#### Scenario: --viewer none disables the window
- **WHEN** the user runs `wally-play ... --viewer none` (or `--no-viewer`)
- **THEN** no `FrameViewer` is constructed, `cv2` is not imported, and
  the loop runs headlessly

#### Scenario: --viewer cv2 is explicit
- **WHEN** the user runs `wally-play ... --viewer cv2`
- **THEN** the behavior is identical to the default (`--viewer cv2`)
