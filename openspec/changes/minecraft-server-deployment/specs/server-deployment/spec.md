## ADDED Requirements

### Requirement: Server connection via pyCraft
The system SHALL establish a connection to a Minecraft Java Edition server using the pyCraft library. The `ServerConnector` SHALL manage connection state transitions (disconnected → connecting → connected) and emit events on connect, disconnect, and error. The system SHALL support vanilla, Paper, Spigot, and Fabric server types.

#### Scenario: Successful connection to offline-mode server
- **WHEN** `ServerConnector.connect()` is called with a valid host:port and offline-mode username
- **THEN** the connection state transitions to `connected` and the `on_connect` event fires

#### Scenario: Connection failure with retry
- **WHEN** `ServerConnector.connect()` is called and the server is unreachable
- **THEN** the system SHALL raise a `ConnectionError` after exhausting retries and the `on_error` event fires with the failure reason

#### Scenario: Unexpected disconnection
- **WHEN** the server drops the connection while state is `connected`
- **THEN** the state transitions to `disconnected` and the `on_disconnect` event fires with the disconnect reason

### Requirement: Authentication support
The system SHALL support both Microsoft OAuth authentication for online-mode servers and username-only authentication for offline-mode servers. The authentication method SHALL be auto-selected based on server mode. Microsoft OAuth tokens SHALL be cached for session reuse.

#### Scenario: Offline-mode authentication
- **WHEN** `DeployConfig.auth_mode` is `"offline"` and a username is provided
- **THEN** the system SHALL authenticate with the username only, without requiring Microsoft credentials

#### Scenario: Online-mode authentication with cached token
- **WHEN** `DeployConfig.auth_mode` is `"online"` and a valid cached token exists
- **THEN** the system SHALL reuse the cached token without prompting for browser-based login

#### Scenario: Online-mode authentication requiring new token
- **WHEN** `DeployConfig.auth_mode` is `"online"` and no valid cached token exists
- **THEN** the system SHALL initiate the Microsoft OAuth flow and cache the resulting token

### Requirement: Session persistence with reconnection
The system SHALL automatically reconnect on unexpected disconnection using exponential backoff (1s, 2s, 4s, 8s, up to 60s maximum). The system SHALL attempt a maximum of 10 reconnections before performing a graceful exit. On reconnection, the system SHALL restore the last known position, inventory state, and goal progress.

#### Scenario: Automatic reconnection after disconnect
- **WHEN** the server connection drops unexpectedly
- **THEN** the `SessionManager` SHALL wait 1 second and attempt to reconnect, doubling the wait time on each subsequent failure up to 60 seconds

#### Scenario: Maximum reconnection attempts exhausted
- **WHEN** 10 reconnection attempts have failed
- **THEN** the system SHALL perform a graceful shutdown, saving agent state, and exit with code 1

#### Scenario: Successful reconnection with state restoration
- **WHEN** reconnection succeeds after a disconnect
- **THEN** the agent SHALL resume from its last known position and inventory state, not from the server spawn point

### Requirement: Action throttling at server tick rate
The system SHALL execute actions at 50ms intervals (20 TPS) using an async queue-based `ActionThrottler`. The throttler SHALL decouple planner output timing from server execution. When the queue depth exceeds a configurable threshold, the system SHALL emit a backpressure warning. The system SHALL monitor server TPS and adapt timing when the server lags below 20 TPS. The queue SHALL be flushed on shutdown.

#### Scenario: Actions executed at 20 TPS
- **WHEN** the planner produces actions faster than 20 TPS
- **THEN** the `ActionThrottler` SHALL queue actions and execute them at 50ms intervals

#### Scenario: Backpressure warning
- **WHEN** the action queue depth exceeds the configured `max_queue_depth` (default: 10)
- **THEN** the system SHALL log a warning with the current queue depth

#### Scenario: Adaptive timing for lagging server
- **WHEN** the server TPS drops below 20 (detected via time update packets)
- **THEN** the throttler SHALL increase the action interval proportionally to match the actual server TPS

#### Scenario: Queue flush on shutdown
- **WHEN** a graceful shutdown is initiated
- **THEN** all pending actions in the queue SHALL be discarded and the queue SHALL be emptied

### Requirement: Action execution translating vectors to packets
The `ActionExecutor` SHALL translate continuous `(25,)` action vectors into Minecraft protocol packets. Movement actions (forward, backward, turn, strafe) SHALL be translated to position/rotation packets. Block interactions (break, place) SHALL be translated to block action packets. Inventory actions (craft, equip, use) SHALL be translated to inventory management packets. The system SHALL validate action vector format and bounds before translation.

#### Scenario: Movement action translation
- **WHEN** an action vector with non-zero movement dimensions (forward, strafe, turn) is received
- **THEN** the `ActionExecutor` SHALL send the corresponding position and rotation packets to the server

#### Scenario: Block break action translation
- **WHEN** an action vector indicates a block break action targeting a specific block position
- **THEN** the `ActionExecutor` SHALL send a block dig packet for the target block coordinates

#### Scenario: Invalid action vector rejected
- **WHEN** an action vector has incorrect shape or values outside valid bounds
- **THEN** the `ActionExecutor` SHALL reject the action, log a warning, and not send any packets

### Requirement: ServerEnv adapter for AgentLoop compatibility
The `ServerEnv` SHALL implement the same interface as `MineStudioAgentEnv` (`reset()`, `step()`, `close()`) so that `AgentLoop` works unchanged. `reset()` SHALL return a preprocessed `(C,H,W)` float tensor reconstructed from server chunk data. `step()` SHALL accept a continuous `(25,)` action tensor, translate it via `ActionExecutor`, and return the next preprocessed frame, reward, done flag, and info dict.

#### Scenario: Reset returns preprocessed frame
- **WHEN** `ServerEnv.reset()` is called after connecting
- **THEN** the system SHALL reconstruct a first-person view image from chunk data and return it as a normalized `(C,H,W)` float tensor

#### Scenario: Step executes action and returns observation
- **WHEN** `ServerEnv.step(action)` is called with a valid `(25,)` action tensor
- **THEN** the system SHALL translate the action to packets, send them to the server, wait for the resulting world update, and return `(frame, reward, done, info)`

#### Scenario: Close disconnects cleanly
- **WHEN** `ServerEnv.close()` is called
- **THEN** the system SHALL disconnect from the server and release all resources

### Requirement: Safety filters
The `SafetyFilter` SHALL provide configurable guards that prevent destructive or dangerous agent actions. The system SHALL include: bedrock breaking prevention (block ID check), lava interaction prevention (block type check), void fall prevention (Y-coordinate check), and configurable action cooldowns. Safety violations SHALL be logged. Each filter SHALL be independently enable/disable-able via `DeployConfig`.

#### Scenario: Bedrock breaking prevented
- **WHEN** the agent attempts to break a bedrock block and the bedrock filter is enabled
- **THEN** the action SHALL be blocked, a warning SHALL be logged, and no packet SHALL be sent

#### Scenario: Lava interaction prevented
- **WHEN** the agent attempts to place a block adjacent to lava and the lava filter is enabled
- **THEN** the action SHALL be blocked and a warning SHALL be logged

#### Scenario: Void fall prevented
- **WHEN** the agent's Y-coordinate drops below the configured void threshold and the void filter is enabled
- **THEN** the system SHALL trigger an emergency movement action (jump/move up) and log a warning

#### Scenario: Safety filter disabled
- **WHEN** a specific safety filter is disabled in `DeployConfig`
- **THEN** actions that would trigger that filter SHALL be allowed through without blocking

### Requirement: Structured logging and monitoring
The system SHALL log all executed actions with timestamps in structured format. The system SHALL log the agent's position `(x, y, z)` every 5 seconds. The system SHALL log server events: chat messages, player joins, and deaths. Log output SHALL be written to a file with rotation. Optional stdout logging SHALL be available for debugging.

#### Scenario: Action logging
- **WHEN** an action is executed on the server
- **THEN** the system SHALL log a structured entry with timestamp, action vector summary, and resulting position

#### Scenario: Position tracking
- **WHEN** 5 seconds have elapsed since the last position log
- **THEN** the system SHALL log the current `(x, y, z)` position

#### Scenario: Server event logging
- **WHEN** a chat message, player join, or death event is received from the server
- **THEN** the system SHALL log the event with timestamp and event details

### Requirement: Graceful shutdown
The system SHALL register signal handlers for SIGINT and SIGTERM. On shutdown, the system SHALL save agent state (position, inventory, goal progress), send a disconnect packet to the server, and exit with code 0.

#### Scenario: SIGINT graceful shutdown
- **WHEN** SIGINT (Ctrl+C) is received
- **THEN** the system SHALL save agent state to a checkpoint file, send a disconnect packet, and exit with code 0

#### Scenario: SIGTERM graceful shutdown
- **WHEN** SIGTERM is received
- **THEN** the system SHALL save agent state to a checkpoint file, send a disconnect packet, and exit with code 0

### Requirement: DeployConfig and wally-deploy CLI
The system SHALL provide a `DeployConfig` Pydantic model with fields for server address, authentication mode, checkpoint path, goal frame path, safety filter toggles, reconnect policy, and logging options. The `wally-deploy` CLI SHALL accept arguments `--server`, `--checkpoint`, `--goal-frame`, `--config`, `--record`, `--output-dir`. The CLI SHALL load the LeWorldModel checkpoint, connect to the server, and begin autonomous gameplay.

#### Scenario: CLI with minimal arguments
- **WHEN** `wally-deploy` is invoked with `--server`, `--checkpoint`, and `--goal-frame`
- **THEN** the system SHALL use default values for all other config fields, load the model, connect, and start the agent loop

#### Scenario: CLI with YAML config file
- **WHEN** `wally-deploy` is invoked with `--config path/to/config.yaml`
- **THEN** the system SHALL load `DeployConfig` from the YAML file and override with any explicitly provided CLI arguments

#### Scenario: CLI with trajectory recording
- **WHEN** `wally-deploy` is invoked with `--record --output-dir data/recordings`
- **THEN** the system SHALL record all observations and actions to a `.tar` shard in the specified output directory
