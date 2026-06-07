## Why

The trained LeWorldModel planner currently operates in local MineStudio environments for evaluation and data collection. To enable persistent, autonomous gameplay in multi-player settings and demonstrate real-world capability, we need to deploy the agent to live Minecraft servers. This bridges the gap between controlled evaluation and practical deployment.

## What Changes

- Add server connector to join Minecraft servers (support for vanilla/Paper/Spigot/Fabric via protocol bridge like mineflayer or MineStudio server mode)
- Implement session manager for persistent connections (join, reconnect on disconnect, heartbeat)
- Add action throttler to respect server tick rate (20 TPS) and prevent action spam
- Implement graceful shutdown (save state, clean disconnect)
- Add server authentication support (Microsoft accounts, offline mode)
- Create deployment CLI entry point (`wally-deploy`) for launching agents on servers
- Add safety mechanisms (action cooldowns, no-griefing rules, configurable behavior bounds)

## Capabilities

### New Capabilities
- `minecraft-server-deployment`: Server connector, session management, action throttling, and deployment CLI for running trained agents on live Minecraft servers

### Modified Capabilities
<!-- No existing capabilities require spec-level changes -->

## Impact

- **Code**: New `src/deployer/` package with server connector, session manager, and action throttler
- **CLI**: New `wally-deploy` entry point in `src/wally/cli/`
- **Dependencies**: May require `mineflayer` (Node.js) or Python Minecraft protocol library (e.g., `minecraft.py`, `pyCraft`)
- **Infrastructure**: Requires access to Minecraft server (local or remote) for testing
- **Integration**: Depends on trained LeWorldModel checkpoints and planner from `minecraft-latent-planner` capability
