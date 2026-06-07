## Minecraft Server Agent Deployment

Purpose:
- Deploy the trained planner agent as a player on a live Minecraft server
- Enable persistent, autonomous gameplay in a multi-player environment

Components:
- Server connector (MineStudio env or protocol bridge like mineflayer)
- Session manager (join, reconnect, heartbeat)
- Action throttler (respect server tick rate, prevent spam)
- Graceful shutdown (save state, disconnect cleanly)

Input:
- Trained LeWorldModel checkpoint
- Server address and credentials
- Goal specification (frame, text, or latent)
- Replanning config (frequency, horizon)

Output:
- Agent joins server and acts autonomously
- Logs of actions, positions, and events
- Optional: live video stream or screenshots

## Open Questions
- Server type? MineStudio, vanilla, Paper/Spigot, Fabric?
- Protocol bridge needed (mineflayer) or MineStudio env only?
- How to handle server restarts / disconnections?
- Rate limiting actions to match server TPS?
- Multi-agent support (multiple planners on same server)?
- Authentication (Microsoft account, offline mode)?
- Safety: action cooldowns, no-griefing rules?
