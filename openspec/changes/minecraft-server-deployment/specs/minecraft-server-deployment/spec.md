## ADDED Requirements

### Requirement: Server Connection
The system SHALL establish a connection to a Minecraft server using the Minecraft protocol. The system SHALL support Java Edition servers (vanilla, Paper, Spigot, Fabric) across Minecraft versions 1.8 through 1.20+.

#### Scenario: Successful connection to vanilla server
- **WHEN** user provides server address (host:port) and valid credentials
- **THEN** system connects to the server and agent appears as a player

#### Scenario: Connection to modded server (Paper/Spigot)
- **WHEN** user provides address of Paper or Spigot server
- **THEN** system negotiates protocol and connects successfully

#### Scenario: Connection failure with invalid address
- **WHEN** user provides unreachable server address
- **THEN** system raises `ConnectionError` with descriptive message

### Requirement: Authentication Support
The system SHALL support Microsoft account authentication for online-mode servers. The system SHALL support offline mode for local/private servers without authentication.

#### Scenario: Microsoft account authentication
- **WHEN** user provides Microsoft account credentials and server is in online mode
- **THEN** system completes OAuth flow and authenticates with Mojang servers

#### Scenario: Offline mode connection
- **WHEN** user specifies offline mode and provides username
- **THEN** system connects without authentication to offline-mode server

#### Scenario: Authentication failure
- **WHEN** user provides invalid Microsoft credentials
- **THEN** system raises `AuthenticationError` with clear error message

### Requirement: Session Persistence
The system SHALL maintain a persistent connection to the server. The system SHALL automatically reconnect on disconnect using exponential backoff (1s, 2s, 4s, 8s, max 60s). The system SHALL persist last known position and inventory state for recovery.

#### Scenario: Automatic reconnection after network disconnect
- **WHEN** network connection drops unexpectedly
- **THEN** system attempts reconnection with exponential backoff and resumes from last known state

#### Scenario: Reconnection after server restart
- **WHEN** server restarts (scheduled maintenance)
- **THEN** system detects disconnect, waits for server availability, and reconnects

#### Scenario: Maximum reconnection attempts exceeded
- **WHEN** reconnection fails after 10 attempts
- **THEN** system logs error and exits gracefully with non-zero exit code

### Requirement: Action Throttling
The system SHALL throttle action execution to match server tick rate (20 TPS = 50ms per action). The system SHALL queue actions from planner and execute them at server-compatible rate. The system SHALL provide backpressure when planner produces actions faster than server can consume.

#### Scenario: Action execution at server tick rate
- **WHEN** planner produces action sequence
- **THEN** system executes actions at 50ms intervals (20 TPS)

#### Scenario: Backpressure on fast planner
- **WHEN** planner produces actions faster than 20 TPS
- **THEN** system queues actions and logs warning when queue depth exceeds threshold

#### Scenario: Server lag compensation
- **WHEN** server TPS drops below 20 (lag)
- **THEN** system adjusts action timing to match actual server TPS

### Requirement: Action Execution
The system SHALL execute planner-generated actions on the Minecraft server. The system SHALL translate planner action vectors to Minecraft protocol packets (movement, block interaction, inventory).

#### Scenario: Movement action execution
- **WHEN** planner outputs movement action (forward, turn)
- **THEN** system sends corresponding position/rotation packets to server

#### Scenario: Block interaction execution
- **WHEN** planner outputs block break/place action
- **THEN** system sends block interaction packets and updates world state

#### Scenario: Inventory action execution
- **WHEN** planner outputs inventory action (craft, equip)
- **THEN** system sends inventory management packets

### Requirement: Graceful Shutdown
The system SHALL disconnect cleanly from server on shutdown signal (SIGINT, SIGTERM). The system SHALL save agent state (position, inventory, goal progress) before disconnecting. The system SHALL send disconnect packet to server.

#### Scenario: Clean shutdown on SIGINT
- **WHEN** user presses Ctrl+C
- **THEN** system saves state, sends disconnect packet, and exits with code 0

#### Scenario: State persistence on shutdown
- **WHEN** shutdown signal received
- **THEN** system writes state to checkpoint file for later resumption

### Requirement: Safety Bounds
The system SHALL enforce configurable safety bounds on agent actions. The system SHALL prevent dangerous actions (breaking bedrock, interacting with lava, falling into void). The system SHALL support action cooldowns to prevent spam.

#### Scenario: Prevent bedrock breaking
- **WHEN** planner attempts to break bedrock block
- **THEN** system blocks action and logs warning

#### Scenario: Prevent lava interaction
- **WHEN** planner attempts to place block in lava or interact with lava
- **THEN** system blocks action and logs warning

#### Scenario: Action cooldown enforcement
- **WHEN** planner outputs same action within cooldown period
- **THEN** system delays action until cooldown expires

### Requirement: Logging and Monitoring
The system SHALL log all executed actions with timestamps. The system SHALL log agent position at regular intervals. The system SHALL log server events (chat messages, player joins, deaths). The system SHALL write logs to file and optionally to stdout.

#### Scenario: Action logging
- **WHEN** agent executes action
- **THEN** system logs action type, parameters, and timestamp to log file

#### Scenario: Position tracking
- **WHEN** agent position changes
- **THEN** system logs position (x, y, z) every 5 seconds

#### Scenario: Event logging
- **WHEN** server sends chat message or event
- **THEN** system logs event with timestamp and content

### Requirement: CLI Deployment Interface
The system SHALL provide `wally-deploy` CLI command. The command SHALL accept server address, credentials, model checkpoint path, and goal specification. The command SHALL support configuration file for advanced options.

#### Scenario: Basic deployment
- **WHEN** user runs `wally-deploy --server host:port --checkpoint model.pt --goal "collect wood"`
- **THEN** system loads model, connects to server, and begins autonomous gameplay

#### Scenario: Configuration file deployment
- **WHEN** user runs `wally-deploy --config deploy.yaml`
- **THEN** system reads all parameters from YAML config and deploys agent

#### Scenario: Invalid checkpoint path
- **WHEN** user provides non-existent checkpoint path
- **THEN** system exits with error message before attempting connection
