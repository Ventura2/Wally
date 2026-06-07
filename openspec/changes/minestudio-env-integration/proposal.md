## Why

The project can collect trajectories and train a LeWorldModel, but has no way to deploy the trained model as an interactive agent in Minecraft. The MineStudio environment integration bridges this gap, enabling closed-loop planning and execution — the final piece needed for an end-to-end AI agent.

## What Changes

- Add a MineStudio gym-compatible environment wrapper that handles frame capture, action translation, and episode lifecycle
- Implement a plan-execute-observe loop that uses the CEM-based MPC planner to generate action sequences and executes them step-by-step in the environment
- Add a configurable replanning strategy (replan every N steps or on divergence detection)
- Add safety bounds: action clipping, episode timeout, and graceful shutdown
- Add a CLI entry point (`wally-play`) to run the agent against a live Minecraft instance

## Capabilities

### New Capabilities
- `minestudio-env`: MineStudio environment wrapper, action execution loop, replanning strategy, safety bounds, and `wally-play` CLI entry point

### Modified Capabilities
- `minecraft-environment-integration`: Resolves open questions (replanning frequency, action interpolation, goal specification, episode termination) and converts from sketch to implementable spec

## Impact

- **New package**: `src/agent/` with env wrapper, planner loop, and CLI
- **Dependencies**: MineStudio (Minecraft environment library), potentially `gymnasium` for interface compatibility
- **Existing code**: Reads from `src/wally/models/` (trained LeWorldModel for latent prediction) and integrates with the planner from `minecraft-latent-planner`
- **CLI**: New `wally-play` entry point registered in `pyproject.toml`
- **Tests**: New test suite in `tests/` covering env wrapper, replanning logic, and safety bounds with mocked MineStudio environment
