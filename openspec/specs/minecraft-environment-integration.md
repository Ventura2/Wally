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
