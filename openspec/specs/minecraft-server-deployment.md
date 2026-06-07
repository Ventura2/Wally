## Minecraft Server Agent Deployment

Purpose:
- Deploy the trained planner agent as a player on a live Minecraft server
- Enable persistent, autonomous gameplay in a multi-player environment
- Build on top of `minecraft-environment-integration` (AgentLoop, planner protocol, trajectory recording)

Components:
- `ServerConnector`: pyCraft wrapper handling Minecraft protocol, authentication, connection state
- `SessionManager`: High-level session lifecycle (join, reconnect, heartbeat, shutdown)
- `ActionThrottler`: Async queue-based rate limiter matching server tick rate (20 TPS = 50ms per action)
- `ActionExecutor`: Translates planner action vectors to Minecraft protocol packets (movement, block interaction, inventory)
- `SafetyFilter`: Configurable action filters (bedrock breaking, lava interaction, void fall prevention, cooldowns)
- `DeployConfig`: Server address, credentials, model checkpoint path, goal spec, safety bounds
- `wally-deploy` CLI entry point

### Server Connection
- Protocol bridge via `pyCraft` (Python-native, supports Minecraft 1.8-1.20+)
- Supports Java Edition servers: vanilla, Paper, Spigot, Fabric
- Connection state tracking: disconnected -> connecting -> connected
- Connection event listeners: on_connect, on_disconnect, on_error

### Authentication
- Microsoft account OAuth flow for online-mode servers (via pyCraft)
- Offline mode for local/private servers (username-only)
- Authentication token caching for session reuse
- Method auto-selection based on server mode

### Session Persistence
- Automatic reconnection on disconnect with exponential backoff (1s, 2s, 4s, 8s, max 60s)
- Maximum 10 reconnection attempts before graceful exit
- State persistence: last known position, inventory, goal progress
- State restoration on reconnection (resume from similar state, not spawn)

### Action Throttling
- Async queue decouples planner output from server execution
- Actions executed at 50ms intervals (20 TPS) matching server tick rate
- Backpressure warning when queue depth exceeds configurable threshold
- Server TPS monitoring with adaptive timing (adjusts when server lags below 20 TPS)
- Queue flush on shutdown

### Action Execution
- Translates continuous `(25,)` action vectors to Minecraft protocol packets
- Movement actions (forward, backward, turn, strafe) -> position/rotation packets
- Block interactions (break, place) -> block action packets
- Inventory actions (craft, equip, use) -> inventory management packets
- Action validation: check vector format and bounds before translation

### Integration with AgentLoop
- Server deployment wraps `MineStudioAgentEnv` with a `ServerEnv` adapter that reads observations from the server connection instead of MineStudio
- Uses the same `AgentLoop` from `minecraft-environment-integration` for plan-execute-observe cycle
- Supports both flat and hierarchical planner modes via `PlannerProtocol`
- Action throttler sits between `AgentLoop` action output and server packet sending

### Safety Bounds
- Bedrock breaking prevention (block ID check)
- Lava interaction prevention (block type check)
- Void fall prevention (Y-coordinate check)
- Configurable action cooldowns (prevent action spam)
- Safety violation logging
- Filters configurable per-deployment (enable/disable specific filters)

### Logging and Monitoring
- Structured logging: all executed actions with timestamps
- Position tracking: log position (x, y, z) every 5 seconds
- Server event logging: chat messages, player joins, deaths
- Log file output with rotation
- Optional stdout logging for debugging

### Graceful Shutdown
- Signal handlers for SIGINT and SIGTERM
- Save agent state (position, inventory, goal progress) before disconnect
- Send disconnect packet to server
- Exit with code 0 on clean shutdown

### CLI: `wally-deploy`
- Arguments: `--server` (host:port), `--checkpoint`, `--goal-frame`, `--config`, `--record`, `--output-dir`
- Loads LeWorldModel, connects to server, begins autonomous gameplay
- YAML config file support for advanced options
- Signal handlers for graceful shutdown

Input:
- Trained LeWorldModel checkpoint
- Server address and credentials (or offline mode username)
- Goal frame (RGB image path)
- DeployConfig YAML (optional, uses defaults)

Output:
- Agent joins server and acts autonomously
- Logs of actions, positions, and events
- Optional trajectory recording (`.tar` shard)
- State checkpoint for resumption

### Dependencies
- `minecraft-environment-integration`: AgentLoop, PlannerProtocol, MineStudioAgentEnv, TrajectoryBuffer
- `hierarchical-planning`: HierarchicalPlanner for subgoal-driven execution
- `mpc-cem-planner` or `gradient-mpc`: Low-level action planning
