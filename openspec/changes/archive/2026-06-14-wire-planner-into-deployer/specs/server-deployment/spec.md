## MODIFIED Requirements

### Requirement: Action execution translating vectors to packets
The `ActionExecutor` SHALL translate continuous `(25,)` action vectors into Minecraft protocol packets and SHALL write the corresponding pyCraft packets on the live `Connection` whenever one is configured. Movement actions (forward, backward, turn, strafe) SHALL be translated to position/rotation packets (`PlayerPositionAndLookPacket`). Block interactions (break, place) SHALL be translated to block action packets (`PlayerDiggingPacket`, `PlayerBlockPlacementPacket`). Inventory actions (craft, equip, use) SHALL be translated to inventory management packets (held-item-change and similar). The system SHALL validate action vector format and bounds before translation. The `execute(action)` method SHALL continue to return the list of translated packet dicts so callers (tests, `info` payload) can inspect what was sent; the write is a side effect of `execute` and SHALL occur only after `validate()` passes.

#### Scenario: Movement action translation and write
- **WHEN** an action vector with non-zero movement dimensions (forward, strafe, turn) is passed to `ActionExecutor.execute()` and a live `Connection` is configured
- **THEN** the `ActionExecutor` SHALL write a `PlayerPositionAndLookPacket` on the connection AND SHALL return the corresponding packet dict in the returned list

#### Scenario: Movement action translation without live connection
- **WHEN** an action vector with non-zero movement dimensions is passed to `ActionExecutor.execute()` and no `Connection` is configured (mock mode, unit tests)
- **THEN** the `ActionExecutor` SHALL return the corresponding packet dict in the returned list and SHALL NOT attempt to write on `None`

#### Scenario: Block break action translation
- **WHEN** an action vector indicates a block break action targeting a specific block position and a live `Connection` is configured
- **THEN** the `ActionExecutor` SHALL write a `PlayerDiggingPacket` (start dig, then finished dig) for the target block coordinates on the connection AND SHALL return the packet dicts in the returned list

#### Scenario: Inventory action translation
- **WHEN** an action vector indicates a hotbar slot selection or craft action and a live `Connection` is configured
- **THEN** the `ActionExecutor` SHALL write a held-item-change (or equivalent) packet on the connection AND SHALL return the corresponding packet dict

#### Scenario: Invalid action vector rejected
- **WHEN** an action vector has incorrect shape or values outside valid bounds
- **THEN** the `ActionExecutor` SHALL reject the action, log a warning, return an empty list, and SHALL NOT write any packets on the connection

### Requirement: Action throttling at server tick rate
The system SHALL execute actions at 50ms intervals (20 TPS) using an async queue-based `ActionThrottler` when the deployer runs in live mode. The throttler SHALL decouple planner output timing from server execution. When the queue depth exceeds a configurable threshold, the system SHALL emit a backpressure warning. The system SHALL monitor server TPS and adapt timing when the server lags below 20 TPS. The queue SHALL be flushed on shutdown. The throttler SHALL expose a `submit_sync(item)` method that callers from a synchronous `step()` path can use to enqueue an item, and a `start()` / `stop()` lifecycle that runs `_process_loop` as a background asyncio task bound to a loop owned by the throttler. When the throttler is disabled (`interval is None` or `0`), `submit_sync` SHALL write the item to its handler synchronously, bypassing the queue.

#### Scenario: Actions executed at 20 TPS in live mode
- **WHEN** the planner produces actions faster than 20 TPS and `ServerEnv` is in live mode with `ActionThrottler(interval=0.05)` started
- **THEN** the `ActionThrottler` SHALL queue items submitted via `submit_sync` and execute the handler at 50ms intervals

#### Scenario: Throttler bypassed in mock mode
- **WHEN** `ServerEnv` is constructed in mock mode with `ActionThrottler(interval=None)` and `submit_sync` is called
- **THEN** the item SHALL be passed to the handler synchronously and the queue SHALL be unused

#### Scenario: Backpressure warning
- **WHEN** the action queue depth exceeds the configured `max_queue_depth` (default: 10)
- **THEN** the system SHALL log a warning with the current queue depth

#### Scenario: Queue flush on shutdown
- **WHEN** a graceful shutdown is initiated
- **THEN** `ActionThrottler.stop()` SHALL be called, the background task SHALL be cancelled, and any pending items in the queue SHALL be discarded

### Requirement: ServerEnv adapter for AgentLoop compatibility
The `ServerEnv` SHALL implement the same interface as `MineStudioAgentEnv` (`reset()`, `step(action) -> (frame, reward, done, info)`, `close()`) so that `AgentLoop` works unchanged. `reset()` SHALL return a preprocessed `(C,H,W)` float tensor reconstructed from server chunk data. `step()` SHALL accept a continuous `(25,)` action tensor, clamp it to `[-1, 1]`, translate it via `ActionExecutor` (which writes on the live connection when one is configured), submit the resulting packet list through the `ActionThrottler` in live mode, and return the next preprocessed frame, reward, done flag, and info dict. The info dict SHALL include a `packets` key with the translated packet list.

#### Scenario: Reset returns preprocessed frame
- **WHEN** `ServerEnv.reset()` is called after connecting
- **THEN** the system SHALL reconstruct a first-person view image from chunk data and return it as a normalized `(C,H,W)` float tensor

#### Scenario: Step executes action and writes packets on the connection
- **WHEN** `ServerEnv.step(action)` is called with a valid `(25,)` action tensor in live mode
- **THEN** the system SHALL translate the action to packets, the `ActionExecutor` SHALL write them on the live `Connection`, the `ActionThrottler` SHALL rate-limit the write, and the method SHALL return `(frame, reward, done, info)` with `info["packets"]` listing the translated packet dicts

#### Scenario: Step executes action without writing when no connection is configured
- **WHEN** `ServerEnv.step(action)` is called in mock mode (no live `Connection`)
- **THEN** the system SHALL translate the action, return the packet list in `info["packets"]`, and SHALL NOT attempt to write on `None`

#### Scenario: Close disconnects cleanly
- **WHEN** `ServerEnv.close()` is called
- **THEN** the system SHALL stop the `ActionThrottler` (if started), disconnect from the server, and release all resources

### Requirement: DeployConfig and wally-deploy CLI
The system SHALL provide a `DeployConfig` Pydantic model with fields for server address, authentication mode, checkpoint path, goal frame path, safety filter toggles, reconnect policy, and logging options. The `wally-deploy` CLI SHALL accept arguments `--server`, `--checkpoint`, `--goal-frame`, `--config`, `--record`, `--output-dir`, `--planner` (choice of `cem`, `gradient`, `hierarchical`, default `cem`), and `--mock` (boolean flag, default `false`). The CLI SHALL load the LeWorldModel checkpoint, construct a planner (using the same factory the `wally-play` CLI uses), build a `ServerEnv` (live mode) or `MockServerEnv` (mock mode), construct an `AgentLoop` with the env and planner, and run one episode via `AgentLoop.run_episode(goal_frame)`. The CLI SHALL print an `EpisodeResult` summary line (steps, final cost, duration). When `--record` is set, the CLI SHALL export the trajectory as a numpy dict to `--output-dir/episode_0.npz`.

#### Scenario: CLI with minimal arguments in live mode
- **WHEN** `wally-deploy --server host:port --checkpoint model.pt --goal-frame goal.png` is invoked
- **THEN** the system SHALL load the checkpoint, build a `GoalConditionedPlanner` via the shared planner factory, construct `ServerEnv`, build `AgentLoop`, run one episode, and print the `EpisodeResult` summary

#### Scenario: CLI in mock mode without a real server
- **WHEN** `wally-deploy --mock --checkpoint model.pt --goal-frame goal.png` is invoked
- **THEN** the system SHALL build a `MockServerEnv` (no server connection), run one `AgentLoop` episode against the planner, and print the summary — no packets SHALL be written to any connection

#### Scenario: CLI with planner selection
- **WHEN** `wally-deploy --planner gradient --checkpoint model.pt --goal-frame goal.png` is invoked
- **THEN** the system SHALL build a `GradientMPC` planner (instead of `GoalConditionedPlanner`) via the same factory

#### Scenario: CLI with YAML config file
- **WHEN** `wally-deploy --config path/to/config.yaml` is invoked
- **THEN** the system SHALL load `DeployConfig` from the YAML file and override with any explicitly provided CLI arguments

#### Scenario: CLI with trajectory recording
- **WHEN** `wally-deploy --record --output-dir data/recordings` is invoked
- **THEN** the system SHALL buffer frames and actions in `AgentLoop`'s `TrajectoryBuffer` and export them to `data/recordings/episode_0.npz` at the end of the episode

#### Scenario: Missing checkpoint or goal frame
- **WHEN** `wally-deploy --checkpoint nonexistent.pt --goal-frame goal.png` is invoked
- **THEN** the system SHALL exit with a non-zero code and a clear error message identifying the missing path

### Requirement: MockServerEnv for offline planner integration tests
The system SHALL provide a `MockServerEnv` class implementing the same `reset() / step(action) / close()` interface as `ServerEnv` and intended for offline smoke testing of the planner integration (no Minecraft server required). `MockServerEnv.reset()` SHALL return a preprocessed `(C, H, W)` frame drawn from a seeded noise generator. `MockServerEnv.step(action)` SHALL run the `ActionExecutor` with `_connection=None` (no write attempts), update an internal mutable position based on the translated packet dicts, return a synthetic next frame, and report `info["packets"]` as a list. The class SHALL be importable from `deployer.env` and SHALL NOT require pyCraft.

#### Scenario: MockServerEnv runs an episode without a server
- **WHEN** `AgentLoop` is constructed with a `MockServerEnv` and a planner, and `run_episode(goal_frame)` is called
- **THEN** the loop SHALL invoke the planner, translate the planned actions, return translated packet dicts in `info["packets"]`, and complete the episode with a finite `EpisodeResult` — no pyCraft connection SHALL be opened

#### Scenario: MockServerEnv step does not write to a connection
- **WHEN** `MockServerEnv.step(action)` is called
- **THEN** the `ActionExecutor` SHALL NOT attempt to write a packet on any connection (no exception is raised on `None`)
