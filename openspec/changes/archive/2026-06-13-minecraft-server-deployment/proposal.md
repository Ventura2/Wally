## Why

The agent can plan and act in MineStudio's local environment, but cannot join a real Minecraft server. Deploying to a live server is the ultimate validation of the trained world model and planner — it proves the agent can operate in uncontrolled, multi-player environments with network latency, server-side physics, and real-world consequences.

## What Changes

- Add `ServerConnector` wrapping pyCraft for Minecraft protocol, authentication (Microsoft OAuth + offline mode), and connection state management
- Add `SessionManager` for session lifecycle: join, reconnect with exponential backoff, heartbeat, graceful shutdown
- Add `ActionThrottler` — async queue-based rate limiter at 20 TPS (50ms/action) with adaptive timing for server lag
- Add `ActionExecutor` translating continuous `(25,)` action vectors to Minecraft protocol packets (movement, block interaction, inventory)
- Add `SafetyFilter` with configurable guards: bedrock breaking, lava interaction, void fall, action cooldowns
- Add `ServerEnv` adapter that reads observations from the server connection, satisfying the same interface as `MineStudioAgentEnv` so `AgentLoop` works unchanged
- Add `DeployConfig` (Pydantic) and `wally-deploy` CLI entry point
- Add structured logging, position tracking, and state persistence for reconnection

## Capabilities

### New Capabilities
- `server-deployment`: Full Minecraft server agent deployment — connection, authentication, session persistence, action throttling, action execution, safety filtering, and CLI orchestration via pyCraft

### Modified Capabilities

_None. The existing `minecraft-environment-integration` capability (AgentLoop, PlannerProtocol) is consumed as-is through the ServerEnv adapter._

## Impact

- **New dependency**: `pyCraft` (Minecraft protocol library) added to `pyproject.toml`
- **New package**: `src/deployer/` with ~7 modules
- **New CLI entry point**: `wally-deploy` registered in `pyproject.toml`
- **Dependencies on existing capabilities**: `minecraft-environment-integration` (AgentLoop, PlannerProtocol, TrajectoryBuffer), `hierarchical-planning` (HierarchicalPlanner), `mpc-cem-planner` / `gradient-mpc` (low-level planning)
- **No breaking changes** to existing code — purely additive
