## 1. ActionExecutor packet sending

- [x] 1.1 In `src/deployer/executor.py`, add a `_send_packet(self, packet: dict)` method that maps each packet dict to a real pyCraft packet and writes it on `self._connection` when one is configured. Movement → `PlayerPositionAndLookPacket` (using session's current position + the dx/dz/dyaw/dpitch deltas), rotation-only → `PlayerPositionAndLookPacket` with `position` unchanged, jump/sneak/sprint → `EntityActionPacket` (start jump / start sneak / start sprint), dig → `PlayerDiggingPacket` (status `STARTED_DIGGING` then `FINISHED_DIGGING`), place → `PlayerBlockPlacementPacket`, pick → `PlayerDiggingPacket` (status `START_DIGGING` with `face=BLOCK_FACE.SELF` for the held block; use the executor's session position as the target), craft → no packet (server-side recipes are dispatched via `CloseWindowPacket` after opening a crafting table; for now, log and skip), select_slot → `HeldItemChangePacket` with `slot`. Each helper short-circuits when `self._connection is None` (mock mode).
- [x] 1.2 Modify `_translate_movement`, `_translate_block_interaction`, `_translate_inventory` to call `_send_packet(pkt)` for each generated packet dict before appending it to the returned list. Keep the dict return shape unchanged.
- [x] 1.3 Make `ActionExecutor.__init__` accept an optional `session: SessionManager | None = None` so movement packets can read the current `(x, y, z)` and `(yaw, pitch)`. `env.reset()` will set it via `self._executor.session = self._session`.

## 2. ActionThrottler sync submit and lifecycle

- [x] 2.1 In `src/deployer/throttler.py`, add `start(self)` and `stop(self)` lifecycle methods that create / cancel a background asyncio task running `_process_loop` on a loop owned by the throttler (`self._loop = asyncio.new_event_loop()`, `self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)`). Make sure `start()` is idempotent (don't double-start).
- [x] 2.2 Add `submit_sync(self, item)` that uses `asyncio.run_coroutine_threadsafe(self.submit(item), self._loop)` when the throttler is running and `interval > 0`. When `interval` is `None` or `0`, `submit_sync` SHALL call the handler synchronously (`self._handler(item)` — wrap in `asyncio.run` to satisfy the async type, since the existing `_noop_handler` in `env.py` is a no-op coroutine).
- [x] 2.3 Update `ActionThrottler.__init__` to accept `interval: float | None = 0.05` and pass through. When `interval is None` or `interval <= 0`, skip the queue and run the handler directly.
- [x] 2.4 Make `stop()` also shut down `self._thread` cleanly (`self._loop.call_soon_threadsafe(self._loop.stop)`; `self._thread.join(timeout=1.0)`).

## 3. ServerEnv wires executor → throttler → connection

- [x] 3.1 In `src/deployer/env.py`, in `ServerEnv.__init__`, store `self._config.throttler_interval` (a new `DeployConfig` field, default `0.05`) and construct the throttler with a real handler that calls `self._executor.send_packets(packets)` (a new method that writes each dict on the connection and returns nothing). Remove the `_noop_handler` placeholder.
- [x] 3.2 In `ServerEnv.reset()`, after `self._session.join()`, set `self._executor.session = self._session` and `self._executor.connection = self._connector.connection`, then call `asyncio.run_coroutine_threadsafe(self._throttler.start(), self._throttler._loop)` (or expose a `start_sync` helper) so the throttler's background task is up.
- [x] 3.3 In `ServerEnv.step(action)`, after the safety check, clamp the action to `[-1, 1]`, call `self._executor.execute(action_np)` (which now writes packets and returns the dict list), then submit the list via `self._throttler.submit_sync(packets)`. Keep returning `info["packets"]` from the dict list. Increment `self._step_count`.
- [x] 3.4 In `ServerEnv.close()`, call `self._throttler.stop()` before `self._session.shutdown()`.

## 4. Shared planner factory

- [x] 4.1 Create `src/agent/planner_factory.py` with a `build_planner(planner_kind: str, rollout: LatentRollout, encoder: torch.nn.Module) -> PlannerProtocol` function that does what `_build_planner` does today in `src/agent/play.py:83-110` (cem → `FlatPlannerAdapter(GoalConditionedPlanner(...))`, gradient → `FlatPlannerAdapter(GradientMPC(...))`, hierarchical → `HierarchicalPlannerAdapter(HierarchicalPlanner(...))`). Raise `ValueError` for unknown kinds.
- [x] 4.2 Refactor `src/agent/play.py` to import and call `build_planner(...)` instead of its inline helper. Delete the local `_build_planner` function. No behavior change.

## 5. MockServerEnv

- [x] 5.1 In `src/deployer/env.py`, add a `MockServerEnv` class with the same `reset() / step(action) / close()` interface as `ServerEnv`. `__init__` takes a `DeployConfig`; sets `self._closed = False`, `self._step_count = 0`, `self._position = (0.0, 64.0, 0.0)`, `self._yaw = 0.0`, `self._pitch = 0.0`, and constructs an `ActionExecutor` (with `_connection = None`) and a `FrameRenderer` (no live chunks — render to a constant-color frame).
- [x] 5.2 `MockServerEnv.reset()` returns a `(3, 224, 224)` float tensor filled with a seeded noise pattern (use `np.random.default_rng(seed=0).random(...)` for determinism). Increment a deterministic step counter.
- [x] 5.3 `MockServerEnv.step(action)` runs `executor.execute(action_np)` (no write — connection is None), updates `self._position` based on the translated movement dicts (apply `dx`/`dz`/`dyaw`/`dpitch` like a real env would), increments `self._step_count`, returns `(synthetic_frame, 0.0, False, {"packets": packets, "step": self._step_count})`.
- [x] 5.4 Export `MockServerEnv` from `src/deployer/__init__.py`.

## 6. wally-deploy CLI rewrite

- [x] 6.1 In `src/deployer/cli.py`, add `--planner {cem,gradient,hierarchical}` (default `cem`) and `--mock` (action=`store_true`, default `False`) to `parse_args`. Validate `--checkpoint` and `--goal-frame` paths up front (exit code 1 with a clear message if missing). Validate that the planner kind is one of the three.
- [x] 6.2 In `cli.main`, after config parsing, load the checkpoint: `rollout = LatentRollout.from_checkpoint(args.checkpoint)`, `encoder = rollout._model.encode`. Build the planner: `planner = build_planner(args.planner, rollout, encoder)`.
- [x] 6.3 Build the env: `env = MockServerEnv(config)` if `args.mock` else `ServerEnv(config)`. Build `AgentLoop(env, planner, AgentConfig(...))`. Run `loop.run_episode(goal_frame)`.
- [x] 6.4 Print the `EpisodeResult` summary (`steps`, `final_cost`, `duration_seconds`) using the same format `wally-play` uses. If `args.record`, write the trajectory `.npz` to `args.output_dir / "episode_0.npz"`.
- [x] 6.5 Remove the old `while not done: action = torch.zeros(25); env.step(action)` loop and the `step_count % 100` heartbeat (the new loop's `EpisodeResult` summary is the heartbeat).

## 7. Tests

- [x] 7.1 Add `tests/test_deployer_executor.py` (or extend `test_deployer_cli.py`): assert that `ActionExecutor.execute(...)` with a mock `Connection` writes the expected number of packets per action type, and that no packets are written when `connection is None`. Cover: movement-only, rotation-only, jump, dig, place, slot select.
- [x] 7.2 Add `tests/test_deployer_throttler.py`: assert that `submit_sync` calls the handler at `interval` spacing when `interval=0.05`, that `submit_sync` calls the handler immediately when `interval=None`, and that `stop()` flushes the queue and shuts the background task.
- [x] 7.3 Add `tests/test_deployer_mock_env.py`: assert `MockServerEnv.reset()` returns the right shape, that `step()` updates internal position based on movement packets, and that `info["packets"]` is populated.
- [x] 7.4 Add `tests/test_deployer_cli_planner.py`: assert that with `--mock --planner cem --checkpoint X --goal-frame Y`, the CLI loads the checkpoint, builds a `GoalConditionedPlanner`, runs `AgentLoop`, and prints a non-empty summary. Reuse a tiny dummy checkpoint (`checkpoints/_smoke_dummy.pt`) or construct an in-memory `LatentRollout` for the test.
- [x] 7.5 Add `tests/test_deployer_planner_factory.py` in the agent package: assert `build_planner("cem", ...)` returns something satisfying `PlannerProtocol`, ditto for `"gradient"` and `"hierarchical"`. Assert `ValueError` for an unknown kind.

## 8. Verification

- [x] 8.1 Run `.\.venv-windows\Scripts\python.exe -m pytest -m smoke -x --tb=short` and confirm all deployer, agent, and planner tests pass.
- [x] 8.2 Run `.\.venv-windows\Scripts\python.exe -m ruff check .` and `.\.venv-windows\Scripts\python.exe -m mypy`. Fix any new lint or type errors.
- [x] 8.3 Manual smoke: `wally-deploy --mock --checkpoint checkpoints/_smoke_dummy.pt --goal-frame <some_goal.png> --planner cem --record --output-dir data/mock_deploy` and confirm the summary line prints and the `.npz` file is non-empty.
