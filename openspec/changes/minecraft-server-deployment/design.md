## Context

The Wally project currently operates in two modes:
1. **Data collection**: MineStudio environment captures gameplay trajectories for training
2. **Evaluation**: Local MineStudio environment runs trained LeWorldModel planner for benchmarking

Both modes use MineStudio's single-player, locally-hosted Minecraft instances with direct environment access. The planner outputs action sequences that execute in a controlled, synchronous loop.

To deploy on live servers, we need to bridge the gap between the planner's action output and Minecraft's network protocol. Live servers introduce network latency, authentication requirements, multi-player dynamics, and server-side rate limiting (20 TPS tick rate).

**Constraints:**
- AMD GPU (RX 6700 XT) for inference - must run efficiently
- ROCm/PyTorch stack - prefer Python-native solutions
- MineStudio already provides environment abstraction - leverage where possible
- Server compatibility: vanilla, Paper, Spigot, Fabric (cover majority of servers)

## Goals / Non-Goals

**Goals:**
- Deploy trained planner agent to live Minecraft servers (vanilla and modded)
- Maintain persistent connection with automatic reconnection on disconnect
- Respect server tick rate and prevent action spam
- Provide CLI for easy deployment (`wally-deploy`)
- Support Microsoft account authentication and offline mode
- Log agent actions and events for analysis

**Non-Goals:**
- Multi-agent coordination (multiple agents working together) - future work
- Server-side plugin/mod development - client-side only
- Real-time video streaming - screenshots/logs only for now
- GUI for monitoring - CLI-based for initial version
- Support for Bedrock Edition - Java Edition only

## Decisions

### 1. Protocol Bridge: pyCraft (Python-native Minecraft protocol library)

**Decision:** Use `pyCraft` library for Minecraft server communication.

**Rationale:**
- Python-native - integrates directly with PyTorch/ROCm stack without Node.js subprocess
- Supports Minecraft 1.8-1.20+ (covers most servers)
- Active maintenance, handles protocol complexity
- Avoids mineflayer (Node.js) which would require subprocess management and IPC

**Alternatives considered:**
- **mineflayer (Node.js)**: Mature but requires Node.js runtime, subprocess management, and JSON-RPC/IPC bridge - adds complexity
- **MineStudio server mode**: Limited documentation, unclear if supports all server types
- **Custom protocol implementation**: Too complex, protocol changes between versions

### 2. Architecture: Deployer Package with Modular Components

**Decision:** Create `src/deployer/` package with three core modules:
- `server_connector.py`: pyCraft wrapper, handles protocol, authentication, connection
- `session_manager.py`: High-level session lifecycle (join, reconnect, heartbeat, shutdown)
- `action_throttler.py`: Rate limiting, action queue, TPS synchronization

**Rationale:**
- Separation of concerns - each module has single responsibility
- Testable in isolation (mock pyCraft for unit tests)
- Reusable for future multi-agent scenarios
- Matches existing code patterns (collector, exporter, validator packages)

**Alternatives considered:**
- **Single monolithic deployer class**: Simpler but harder to test and maintain
- **Extend MineStudio env wrapper**: Tight coupling to MineStudio limits flexibility

### 3. Action Execution: Async Queue with Throttling

**Decision:** Use asyncio queue to decouple planner output from server execution. Planner produces actions, throttler consumes at server tick rate (20 TPS = 50ms per action).

**Rationale:**
- Planner may produce actions faster than server can process (network latency, tick rate)
- Queue provides natural backpressure
- Async allows concurrent inference and execution
- Matches Minecraft's tick-based architecture

**Alternatives considered:**
- **Synchronous execution**: Blocks planner, can't overlap inference and execution
- **Fixed delay (sleep 50ms)**: Doesn't account for network latency or server lag

### 4. Reconnection Strategy: Exponential Backoff with State Persistence

**Decision:** On disconnect, attempt reconnection with exponential backoff (1s, 2s, 4s, 8s, max 60s). Persist last known position and inventory to resume from similar state.

**Rationale:**
- Servers restart, network issues occur - agent must be resilient
- Exponential backoff prevents server spam during outages
- State persistence allows graceful recovery (not starting from spawn)

**Alternatives considered:**
- **Immediate reconnect**: Can spam server, may get banned
- **No reconnection**: Agent dies on first disconnect - unacceptable for persistent deployment

### 5. Authentication: Support Microsoft OAuth and Offline Mode

**Decision:** Implement both Microsoft account authentication (for online servers) and offline mode (for local/private servers). Use `pyCraft`'s built-in auth support.

**Rationale:**
- Microsoft auth required for most public servers (Minecraft online mode)
- Offline mode essential for local testing and private servers
- pyCraft handles OAuth flow - minimal custom code

**Alternatives considered:**
- **Microsoft auth only**: Can't test locally without online server
- **Offline mode only**: Can't join most public servers

## Risks / Trade-offs

**[Risk] Network latency causes action desync** → Mitigation: Action throttler tracks server TPS and adjusts timing. Log warnings if latency exceeds threshold.

**[Risk] Server compatibility issues (protocol variations, anti-cheat)** → Mitigation: Test against vanilla, Paper, Spigot. Document known incompatible servers. Allow protocol version configuration.

**[Risk] Agent gets stuck or behaves unexpectedly** → Mitigation: Implement safety bounds (no breaking bedrock, no lava interaction). Add configurable action cooldowns. Provide emergency stop command.

**[Risk] pyCraft library unmaintained or incompatible** → Mitigation: Abstract protocol layer allows swapping to mineflayer or alternative if needed. Monitor pyCraft releases.

**[Trade-off] Python-native (pyCraft) vs Node.js (mineflayer)** → Chose Python for simpler integration with PyTorch stack, but mineflayer has larger community. Acceptable trade-off for reduced complexity.

**[Trade-off] Single-agent focus vs multi-agent architecture** → Designing for single agent now, but modular architecture allows future multi-agent extension. Acceptable to defer complexity.

## Open Questions

- How to handle server-side events (chat messages, player interactions, death)?
- Should agent respond to chat commands from server admins?
- How to specify goals for server deployment (text description, target coordinates, item collection)?
- What metrics to log for server deployment (success rate, distance traveled, items collected)?
- How to handle server restarts (scheduled maintenance) vs unexpected crashes?
