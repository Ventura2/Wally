## Why

`wally-deploy` connects to a Minecraft server, sits there, and never moves. The `server-deployment` spec already requires the CLI to load the LeWorldModel checkpoint and start autonomous gameplay, and the `ActionExecutor` to translate action vectors into protocol packets sent on the connection, but the implementation is a stub: `cli.py:95-96` hardcodes `action = torch.zeros(25)`, `executor.execute()` only returns a list of dicts that are stuffed into `info["packets"]` and never sent, and the trained world model + planner from `src/wally/planner/` are never invoked. Every checkpoint, every planner, every CEM optimization is wasted during deployment.

## What Changes

- Make `ActionExecutor` actually send packets. Replace the dict-only return with real pyCraft packet writes (`PlayerPositionAndLookPacket`, `PlayerDiggingPacket`, `use_entity`, inventory select) on the live `Connection`. Keep the dict shape for tests and for the `info` payload.
- Wire `ActionThrottler` into the deploy loop so movement is rate-limited to 20 TPS as the spec requires, and so the throttler's queue-flush-on-shutdown and TPS-adaptive-timing behavior actually runs.
- Make `wally-deploy` load the trained LeWorldModel checkpoint, build a CEM (or gradient / hierarchical) planner, and drive a `ServerEnv` through the existing `AgentLoop` plan-execute-observe cycle (with warm-start on replan) — the same machinery `wally-play` uses against MineStudio. The CLI no longer holds its own gameplay loop.
- Add a small adapter so `ServerEnv` is usable by `AgentLoop` (it already has `reset` / `step` / `close`; the only real shape work is making `step(action)` not crash on the planner's continuous `(25,)` vector and ensuring the env reports `done` cleanly).
- Add a `wally-deploy --planner {cem,gradient,hierarchical}` flag mirroring `wally-play --planner`, plus a `wally-deploy --mock` flag that drives the same loop against a mock `ServerEnv` (no real server) so the planner integration can be smoke-tested without standing up a Minecraft server.

No BREAKING changes to existing CLIs (`wally-collect`, `wally-train`, `wally-play`); only `wally-deploy` changes.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `server-deployment`: add a requirement that `ActionExecutor` writes translated packets on the live pyCraft `Connection` (not only returns a dict), and that `wally-deploy` loads the LeWorldModel checkpoint + planner and runs the agent through `AgentLoop` (or an equivalent plan-execute-observe loop) against `ServerEnv`. The existing requirements "Action execution translating vectors to packets" and "DeployConfig and wally-deploy CLI" stay, but their scenario "Step executes action and returns observation" is currently not satisfiable by the implementation, so the delta will tighten the observable contract (packets are sent on the wire; the CLI's loop is planner-driven).
- `agent-loop`: no spec change needed — the existing `AgentLoop` already implements the plan-execute-observe cycle with warm-start replanning. The change is a new scenario documenting that `AgentLoop` is the same loop used by `wally-deploy`, but we keep it as a code-level refactor on the deployer side rather than a spec change.

## Impact

- **Code**:
  - `src/deployer/executor.py` — add real packet writers, keep dict return for tests.
  - `src/deployer/env.py` — actually call the executor's send path (or hand packets to `ActionThrottler`); make `step()` end-to-end consistent with what `AgentLoop` expects.
  - `src/deployer/throttler.py` — surface a sync `submit` for use from the env's step path, or run the async loop in a background task during the deploy.
  - `src/deployer/cli.py` — load checkpoint via `LatentRollout.from_checkpoint`, build planner via the same `_build_planner` helper used in `src/agent/play.py` (or a small refactor to share it), construct `ServerEnv` + `AgentLoop`, run an episode; add `--planner` and `--mock`.
  - `src/agent/play.py` — extract `_build_planner` into a shared helper if the deployer wants to reuse it (small refactor).
- **Tests**: extend `tests/test_deployer_cli.py` and `tests/test_deployer_integration.py` to cover (a) packets actually get written, (b) the planner is called, (c) the `--mock` path runs an episode. Keep existing dict-only tests for the executor's pure translation logic.
- **Docs / config**: `pyproject.toml` already exposes `wally-deploy`; no new entry points. `configs/deploy_default.yaml` (if it exists) will need `--planner` and `--mock` documented; otherwise nothing.
- **Dependencies**: pyCraft is already an implicit dependency. No new packages.
- **Hardware**: planner inference runs on the same Windows TheRock PyTorch stack training uses; no change to the deploy runtime.
