## Context

The `minestudio-agent-loop` change introduces `AgentLoop`, `PlannerProtocol`, and `MineStudioAgentEnv` — enabling plan-execute-observe cycles in MineStudio's local environment. Server deployment extends this to real Minecraft servers by replacing the MineStudio backend with a pyCraft network connection while keeping `AgentLoop` unchanged.

Key constraints:
- `AgentLoop` and `PlannerProtocol` are not yet implemented (in-progress in `minestudio-agent-loop` change)
- pyCraft is the only mature Python Minecraft protocol library supporting 1.8–1.20+
- Minecraft servers run at 20 TPS (50ms tick), so action execution must be rate-limited
- Observations from pyCraft arrive as structured packets (not raw pixels), requiring frame reconstruction

## Goals / Non-Goals

**Goals:**
- Deploy the trained planner agent as an autonomous player on a live Minecraft server
- Reuse `AgentLoop` and `PlannerProtocol` unchanged via a `ServerEnv` adapter matching `MineStudioAgentEnv`'s interface
- Provide reliable session persistence with automatic reconnection
- Enforce safety bounds to prevent destructive or nonsensical agent behavior
- Support both offline-mode (local) and online-mode (Microsoft auth) servers

**Non-Goals:**
- Multi-agent coordination (one agent per deployment instance)
- Real-time human-agent collaboration or chat interaction
- Support for Bedrock Edition servers (Java Edition only via pyCraft)
- GPU-accelerated rendering of server observations (CPU-only frame reconstruction)
- Server administration (OP commands, world editing, plugin management)

## Decisions

### 1. pyCraft as protocol bridge

**Choice**: Use `pyCraft` library for Minecraft protocol handling.

**Rationale**: pyCraft is the only actively maintained Python library for Minecraft protocol (supports 1.8–1.20+, Java Edition). Alternatives considered:
- **Custom protocol implementation**: Months of work, fragile, not worth it
- **node-minecraft-protocol (Node.js)**: Would require a separate process and IPC bridge, adding complexity
- **Java-based bots (Mineflayer)**: Same cross-language problem; Wally is Python-native

**Trade-off**: pyCraft's API is callback-based and synchronous; we'll wrap it in an async layer.

### 2. ServerEnv adapter pattern

**Choice**: Create `ServerEnv` with the same interface as `MineStudioAgentEnv` (`reset()`, `step()`, `close()`), reading observations from pyCraft instead of MineStudio.

**Rationale**: This lets `AgentLoop` work unchanged. The adapter pattern is already established — `MineStudioAgentEnv` wraps `MineStudioEnv`; `ServerEnv` wraps `ServerConnector`. Both produce preprocessed `(C,H,W)` tensors and accept continuous `(25,)` action vectors.

**Alternative considered**: Subclassing `MineStudioAgentEnv` — rejected because the underlying observation source is fundamentally different (network packets vs. simulator frames), and inheritance would couple us to MineStudio internals.

### 3. Async architecture with asyncio

**Choice**: Use Python `asyncio` for the connection layer, action throttler, and event handling.

**Rationale**: Minecraft server interaction is inherently I/O-bound and event-driven. The action throttler needs precise timing (50ms intervals), reconnection needs non-blocking waits, and pyCraft events arrive asynchronously. Running the planner in a thread pool executor keeps the event loop responsive.

**Alternative considered**: Threading with queues — simpler but harder to manage reconnection state and graceful shutdown cleanly.

### 4. Frame reconstruction from server packets

**Choice**: Reconstruct RGB frames from pyCraft's chunk data + entity position/rotation using a lightweight software renderer (or Minecraft's own map rendering if available via protocol).

**Rationale**: pyCraft provides chunk block data and the bot's position/look direction. We need to synthesize a first-person view image matching the `(H,W,3)` format the world model expects. Options:
- **Chunk-based rendering**: Reconstruct a local voxel grid from chunk packets, raycast from player position to produce a first-person image. Computationally cheap for small render distances.
- **Map item protocol**: Some servers support map rendering via protocol, but this is server-dependent and unreliable.

We'll start with chunk-based rendering at a short render distance (4–6 chunks) and iterate.

### 5. Config via Pydantic BaseModel

**Choice**: `DeployConfig` as a Pydantic `BaseModel`, following the pattern established by `AgentConfig`, `CEMConfig`, etc.

**Rationale**: Consistent with existing config classes. Pydantic provides validation, YAML loading via `from_yaml()`, and `default()` factory. Fields: server address, auth mode, checkpoint path, goal frame path, safety filter toggles, reconnect policy.

### 6. Package location: `src/deployer/`

**Choice**: New top-level package `src/deployer/` using `src.` import prefix.

**Rationale**: Follows the pattern of `src/collector/`, `src/agent/`, `src/validator/` — each a standalone top-level package under `src/`. The `wally/` package is reserved for the core ML pipeline.

## Risks / Trade-offs

- **[Observation fidelity]** Rendered frames from chunk data won't match MineStudio's pixel-perfect rendering → Mitigation: Keep render distance short, focus on block-level accuracy. Fine-tuning on server-collected data can close the gap later.
- **[pyCraft version compatibility]** pyCraft may lag behind newest Minecraft versions → Mitigation: Target a stable Minecraft version (1.20.x) and pin pyCraft version. Version negotiation at connection time.
- **[Network latency]** High latency causes observation-action delay, degrading planner performance → Mitigation: Action throttler decouples planner timing from network. Log latency metrics for monitoring.
- **[Authentication complexity]** Microsoft OAuth flow requires browser interaction for initial token → Mitigation: Token caching for session reuse. Offline mode available for local testing.
- **[Safety filter completeness]** Unknown edge cases in block interaction (e.g., TNT chains, water flow) → Mitigation: Conservative default filters, configurable per-deployment, comprehensive logging for post-hoc analysis.
