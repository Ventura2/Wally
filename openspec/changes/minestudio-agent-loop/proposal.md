## Why

The planner and world model are functional in isolation, but there is no closed-loop agent that can plan, execute actions in Minecraft, observe the result, and replan. The `MineStudioAgentEnv`, `AgentConfig`, and `TrajectoryBuffer` exist as scaffolding, but the orchestration layer (`AgentLoop`), planner abstraction (`PlannerProtocol` / `PlanResult`), and `wally-play` CLI are missing. Without these, we cannot run the trained model against a live environment to validate the full pipeline.

## What Changes

- Add `PlannerProtocol` and `PlanResult` dataclass to unify flat and hierarchical planner outputs behind a single interface
- Add `AgentLoop` implementing the plan-execute-observe loop with fixed-interval replanning, warm-start CEM, graceful shutdown on `KeyboardInterrupt`, and episode statistics
- Add `wally-play` CLI entry point that loads a checkpoint, constructs the planner, and runs one episode
- Wire `wally-play` into `pyproject.toml` `[project.scripts]`
- Add tests for `PlannerProtocol` normalization, `AgentLoop` step logic (with mock env/planner), and CLI argument parsing

## Capabilities

### New Capabilities
- `agent-loop`: Plan-execute-observe orchestration, planner protocol abstraction, and `wally-play` CLI entry point

### Modified Capabilities
<!-- No existing spec requirements are changing; this change implements the already-defined minecraft-environment-integration spec. -->

## Impact

- **New files**: `src/agent/loop.py`, `src/agent/protocol.py`, `src/agent/play.py` (CLI)
- **Modified files**: `src/agent/__init__.py` (exports), `pyproject.toml` (new script entry point)
- **Dependencies**: No new dependencies; uses existing `torch`, `numpy`, `PIL`, `pydantic`
- **Tests**: New test files `tests/test_agent_loop.py`, `tests/test_play_cli.py`
- **APIs**: `wally-play` CLI added; no breaking changes to existing APIs
