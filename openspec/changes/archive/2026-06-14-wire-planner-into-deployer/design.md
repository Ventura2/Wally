## Context

The `wally-deploy` entry point connects to a Minecraft server, then enters a `while not done` loop that always passes `torch.zeros(25)` to `env.step()`. Two distinct things are broken:

1. **No packet is ever sent.** `ActionExecutor.execute(action)` returns `list[dict]` and `env.step()` tucks that list into `info["packets"]`. Nothing in the deployer ever calls `connection.write_packet(...)`. A grep for `write_packet` / `send_packet` / `register_packet` in `src/deployer/` returns zero hits.
2. **No model is ever consulted.** The CLI does not load the LeWorldModel checkpoint, does not build a planner, and does not call into `src/wally/planner/`. The trained model is unused at deploy time.

Meanwhile, the agent-side `wally-play` already does the right thing for MineStudio: it loads a checkpoint via `LatentRollout.from_checkpoint`, builds a `GoalConditionedPlanner` (or `GradientMPC` / `HierarchicalPlanner`), wraps it in a `PlannerProtocol` adapter, and drives it through `AgentLoop.run_episode()`. The deployer should do the same, just with `ServerEnv` instead of `MineStudioAgentEnv`.

The `server-deployment` spec at `openspec/specs/server-deployment/spec.md` already requires both behaviors (the executor sends packets; the CLI loads the checkpoint and starts the agent loop). The code does not implement them.

## Goals / Non-Goals

**Goals:**
- A deployed `wally-deploy` agent actually walks around in Minecraft, driven by a trained LeWorldModel + CEM (or gradient / hierarchical) planner.
- The deploy loop reuses `AgentLoop` so warm-start-on-replan, replan interval, and trajectory recording all work the same way they do for `wally-play`.
- The packet-sending path goes through the existing `ActionThrottler` so 20 TPS rate limiting, backpressure warnings, adaptive timing, and queue flush on shutdown all work.
- A `--mock` mode lets the planner integration be smoke-tested locally without a Minecraft server (mock env returns synthetic frames, executor packets are recorded but not sent).

**Non-Goals:**
- Re-training the world model. We only consume checkpoints that already exist (`checkpoints/checkpoint_100000.pt` etc.).
- Changing the action vector format (25-dim continuous, MineStudio vocabulary) — out of scope; the action layout is shared with training.
- Changing MineStudio or the agent loop. The `wally-play` flow stays exactly as it is.
- Standing up a real Minecraft server, integration-testing the deploy against a live server in CI. The mock mode is the CI-friendly path; live-server tests are a manual `wally-deploy` run.
- Auto-reconnect on disconnect. `SessionManager` already supports it, but wiring it into `AgentLoop` is a separate concern and not required to make the agent move.

## Decisions

### Decision 1: `ActionExecutor` writes packets as a side effect of `execute()`, and keeps returning the dict list

- **Why:** Every existing test in `tests/test_deployer_integration.py` mocks `executor.execute` and asserts the returned packet list. If we change the return type or move the write into a separate method, every test breaks for cosmetic reasons. Adding a side effect (write on the configured `_connection`) keeps the API stable.
- **Mechanism:** `ActionExecutor` gains a `connection` attribute (already partially present, set in `env.reset()`). Each `_translate_*` helper writes the corresponding pyCraft packet on `self._connection` and then appends the dict representation to the returned list. When `_connection is None` (tests, mock mode), only the dict list is returned. `validate()` short-circuits before any write.
- **Alternatives considered:**
  - Split into `translate()` (pure) and `send()` (I/O) — cleaner separation, but doubles the surface area and forces `env.step()` to know about the protocol. The current executor already validates bounds and ignores invalid actions, so a single entry point is fine.
  - Send packets from `env.step()` directly, leaving `executor.execute()` pure — this is the cleanest separation, but the dict list is the documented payload (`info["packets"]` is asserted by `test_step_returns_packets_in_info`), and the executor is the natural place for "this continuous vector becomes a Minecraft protocol event." Putting the write there keeps the abstraction coherent.

### Decision 2: Run `ActionThrottler` as a background asyncio task during deployment, with a sync `submit` shim

- **Why:** `ActionThrottler` is already an async queue. `env.step()` is sync (matches `AgentLoop`'s sync contract). The simplest reconciliation is to start the throttler in `ServerEnv.__init__` (or in `ServerEnv.reset()`), run `_process_loop` as a background task, and have the throttler's `handler` call the executor's `send_packets_only` (a thin wrapper that writes the dicts and returns nothing). `env.step()` pushes the packet list via `throttler.submit_sync(packets)`.
- **Mechanism:**
  - Add `ActionThrottler.submit_sync(self, item)` — wraps `asyncio.run_coroutine_threadsafe(self.submit(item), self._loop)` (or schedules with `loop.call_soon_threadsafe`).
  - `ActionExecutor.send_packets(packets: list[dict])` — writes each dict on the live connection; no return value.
  - `env.step()` becomes: `packets = self._executor.execute(action_np)` (translate + send on the live connection, return for `info`); if the throttler is enabled (`interval > 0`), call `throttler.submit_sync(packets)`.
  - `ActionThrottler.__init__` gains an optional `interval: float | None = 0.05`; when `None` or `0`, the throttler is bypassed and `submit_sync` writes directly. This keeps existing tests happy.
- **Alternatives considered:**
  - Make `env.step()` async and have `AgentLoop` await it — `AgentLoop` is sync; making it async would touch a lot of code and break the "play" path.
  - Drop the throttler entirely and just `time.sleep(0.05)` in `step()` — works but loses adaptive TPS, queue-flush-on-shutdown, and the backpressure warning that the spec already requires.

### Decision 3: `wally-deploy` mirrors `wally-play` — load checkpoint, build planner, build `ServerEnv`, run `AgentLoop`

- **Why:** `AgentLoop` already implements the plan-execute-observe cycle with warm-start replan. Reusing it gives us warm-start CEM "for free," plus trajectory recording, plus `KeyboardInterrupt` handling. Trying to write a new deploy-specific loop would duplicate this logic and risk drift.
- **Mechanism:** in `src/deployer/cli.py`,
  1. Parse `--planner {cem,gradient,hierarchical}` and `--mock` (default `cem`, `mock=False`).
  2. Validate `--checkpoint` and `--goal-frame` paths up front (mirroring `wally-play`).
  3. `rollout = LatentRollout.from_checkpoint(args.checkpoint)`; `encoder = rollout._model.encode`.
  4. Build planner via a shared helper (`_build_planner`) extracted from `src/agent/play.py:83-110` into `src/agent/planner_factory.py` so both CLIs use the same logic. `wally-play` imports it; `wally-deploy` imports it.
  5. Build env: `ServerEnv(config)` for live mode, `MockServerEnv(config)` for `--mock`. Both expose `reset()` / `step(action) -> (frame, reward, done, info)` / `close()`.
  6. `loop = AgentLoop(env, planner, agent_config)`; `loop.run_episode(goal_frame)`.
  7. Print `EpisodeResult` summary; if `--record`, dump `.npz` to `--output-dir`.
- **Alternatives considered:**
  - Keep `wally-deploy`'s own inline loop, just feed it a planner instead of `torch.zeros(25)` — this is the smallest change, but we lose `AgentLoop`'s warm-start replan and recording. The proposal explicitly asked to use the agent loop machinery, so reuse is the right call.
  - Build a `ServerEnv` that *wraps* `AgentLoop` (server-in-the-loop planner) — over-engineered for this change.

### Decision 4: `MockServerEnv` is a deterministic, no-server env for smoke tests

- **Why:** CI on a Windows box cannot stand up a Minecraft server. A mock env lets us exercise (a) checkpoint loading, (b) planner construction, (c) `AgentLoop` against `ServerEnv`'s interface, and (d) packet dict generation — all without pyCraft. It also makes the integration test for "planner drives a deploy-like loop" possible.
- **Mechanism:** `MockServerEnv` implements the same `reset / step / close` interface as `ServerEnv`. `reset()` returns a `(C, H, W)` frame drawn from a seeded noise generator. `step()` runs the executor (which, in mock mode, has `_connection = None` and so only returns the dict list — no write attempts) and returns `(synthetic_frame, 0.0, False, {"packets": [...]})`. Position is tracked in a mutable state so the renderer can produce a moving viewpoint.
- **Alternatives considered:**
  - Use the existing `tests/test_integration.py` mock env — that env is wired to the collector path, not the deployer. Different interface; not a drop-in.
  - Spin up a real Minecraft server in CI — out of scope, slow, and the AGENTS.md says WSL2 GPU compute is broken anyway.

### Decision 5: `ServerEnv` clamps the planner's action vector to `[-1, 1]` and reuses `MineStudioActionVocab` for action validation

- **Why:** `MineStudioAgentEnv` already does this in `src/agent/env.py:49-54`. The planner produces a `(25,)` continuous vector in `[-1, 1]`; the executor validates bounds; nothing extra is needed. Reusing the vocab keeps action semantics consistent across `wally-play` and `wally-deploy`.
- **Mechanism:** in `env.step()`, after `action_np = action.detach().cpu().numpy()`, clamp with `np.clip(action_np, -1.0, 1.0)`. Then hand to the executor.

## Risks / Trade-offs

- [Risk] **pyCraft's `Connection` API drifts or is unavailable** at runtime → Mitigation: keep the dict-list return from `executor.execute()` so unit tests don't need pyCraft. The mock env path tests the planner integration without pyCraft at all. The live path is exercised manually with `wally-deploy --mock=false` against a real server.
- [Risk] **`asyncio.run_coroutine_threadsafe` from a sync `step()` requires the throttler to own an event loop** → Mitigation: `ActionThrottler.__init__` creates the loop, `start()` schedules `_process_loop` on it, `submit_sync` uses `run_coroutine_threadsafe`. The throttler is opt-in (`interval=None` skips it), so tests that don't care about rate limiting can disable it.
- [Risk] **Moving the write into the executor changes the timing of the test mock** — the test `test_step_calls_executor` mocks `execute` and asserts it's called once. The mock will continue to be called once; the side effect (write on `connection`) is not asserted in any existing test, so nothing breaks. New tests assert the write happened.
- [Risk] **Sharing `_build_planner` between `wally-play` and `wally-deploy` introduces a coupling** — `wally-deploy` (deployer package) imports from `wally` (training package) and from `agent` (agent loop package) → Mitigation: this coupling already exists implicitly (the deployer spec already requires loading the LeWorldModel checkpoint). Centralizing the planner factory in `src/agent/planner_factory.py` is a small refactor that makes the coupling explicit and tested.
- [Risk] **`AgentLoop` is currently CPU-only by default in `play.py`** — `rollout = LatentRollout.from_checkpoint(...)` does not push to GPU → Mitigation: the deploy runs on the same TheRock PyTorch stack training uses; users can opt into GPU via an env var or a `DeployConfig.cuda: bool` flag, mirroring what `wally-train` does. This is out of scope for the immediate "make it move" change but documented in the design.
- [Risk] **Mock env's synthetic frames will not produce sensible planner output** — CEM is meant to plan toward a goal, and the goal is a real image. In mock mode the goal-frame path still loads the user-supplied `--goal-frame`; the planner's cost function is a latent-space distance, so it works on any (latent-encoded) input. The mock env produces stable latents, so the planner will produce a stable action sequence, which is enough to verify the wiring end-to-end.

## Migration Plan

1. Land change behind `--mock=true` (default) so no existing deployer behavior changes.
2. Run `pytest -m smoke` and the new integration tests; confirm no regressions in `tests/test_deployer_*` and `tests/test_agent_*`.
3. Manual smoke test: `wally-deploy --mock --checkpoint checkpoints/checkpoint_100000.pt --goal-frame data/sample_goal.png --planner cem --record --output-dir data/mock_deploy`. Verify the `EpisodeResult` summary line prints and the `.npz` trajectory file is non-empty.
4. Manual live test: `wally-deploy --mock=false --server localhost:25565 --checkpoint checkpoints/checkpoint_100000.pt --goal-frame data/sample_goal.png --planner cem`. Connect to a Minecraft client, observe the agent move.
5. After both pass, flip the default to `--mock=false`.

Rollback: revert the commit. No data migration; no spec archive needed (this is a stub-completion, not a behavior change for callers other than `wally-deploy`).

## Open Questions

- Should `DeployConfig` gain a `cuda: bool = False` flag (or `device: str = "cpu"`) so the deployer can run planner inference on the GPU? The training stack is GPU-capable; the deploy CLI currently has no device knob. Out of scope for the immediate change, but worth a follow-up.
- Should `wally-deploy` support `--planner random` or `--planner scripted` as a fallback when no checkpoint is available? Useful for "I just want to verify the connection works," but the `--mock` path covers that case. Punt.
- Should the `SessionManager` reconnect logic be wired into `AgentLoop`'s KeyboardInterrupt handler? The deploy loop exits on disconnect today; reconnect would let long-running deploys survive a server restart. Follow-up.
