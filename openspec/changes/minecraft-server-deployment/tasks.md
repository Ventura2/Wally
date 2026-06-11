## 1. Setup

- [x] 1.1 Create `src/deployer/` package with `__init__.py` and module stubs (`connector.py`, `session.py`, `throttler.py`, `executor.py`, `safety.py`, `env.py`, `config.py`, `logging.py`, `frame_renderer.py`)
- [x] 1.2 Add `pyCraft` dependency to `pyproject.toml`
- [x] 1.3 Implement `DeployConfig` Pydantic model in `src/deployer/config.py` with fields: `server_host`, `server_port`, `auth_mode` (Literal["online", "offline"]), `username`, `checkpoint_path`, `goal_frame_path`, `safety` (SafetyConfig sub-model with filter toggles), `reconnect` (ReconnectConfig with max_attempts, backoff params), `log_dir`, `log_to_stdout`, `record_trajectory`, `output_dir`, `render_distance` (chunks); include `from_yaml()` and `default()` classmethods and field validators

## 2. Server Connection

- [x] 2.1 Implement `ServerConnector` in `src/deployer/connector.py`: wraps pyCraft `Connection`, manages state enum (`DISCONNECTED`, `CONNECTING`, `CONNECTED`), provides `connect()`, `disconnect()`, and event callbacks (`on_connect`, `on_disconnect`, `on_error`)
- [x] 2.2 Implement connection event listener registration and dispatch (callback registry pattern)
- [x] 2.3 Write unit tests for `ServerConnector` state transitions using mocked pyCraft connection

## 3. Authentication

- [x] 3.1 Implement offline-mode authentication in `src/deployer/auth.py`: username-only login via pyCraft offline connection
- [x] 3.2 Implement online-mode authentication: Microsoft OAuth flow via pyCraft, token caching to `~/.wally/auth_token.json`, token refresh logic
- [x] 3.3 Implement auth method auto-selection based on `DeployConfig.auth_mode`
- [x] 3.4 Write unit tests for auth mode selection and token caching

## 4. Session Management

- [x] 4.1 Implement `SessionManager` in `src/deployer/session.py`: high-level lifecycle (join, heartbeat, shutdown), wraps `ServerConnector`
- [x] 4.2 Implement automatic reconnection with exponential backoff (1s, 2s, 4s, 8s, max 60s) and max 10 attempts
- [x] 4.3 Implement state persistence: save/restore last known position `(x, y, z)`, inventory snapshot, and goal progress to JSON checkpoint file
- [x] 4.4 Implement state restoration on reconnection (resume from saved state, not server spawn)
- [x] 4.5 Write unit tests for reconnection backoff timing and state persistence

## 5. Action Throttling

- [x] 5.1 Implement `ActionThrottler` in `src/deployer/throttler.py`: async queue-based rate limiter with configurable interval (default 50ms)
- [x] 5.2 Implement backpressure warning when queue depth exceeds `max_queue_depth` (default 10)
- [x] 5.3 Implement adaptive timing: monitor server TPS from time update packets, adjust interval when TPS < 20
- [x] 5.4 Implement queue flush on shutdown
- [x] 5.5 Write unit tests for throttler timing, backpressure, and flush behavior

## 6. Action Execution

- [x] 6.1 Implement `ActionExecutor` in `src/deployer/executor.py`: translate `(25,)` action vectors to pyCraft packets
- [x] 6.2 Implement movement action translation: forward/backward/strafe → position packets, turn/pitch → rotation packets
- [x] 6.3 Implement block interaction translation: break → dig packet, place → block place packet
- [x] 6.4 Implement inventory action translation: craft, equip, use → inventory management packets
- [x] 6.5 Implement action validation: check vector shape `(25,)` and bounds before translation, reject invalid actions with warning log
- [x] 6.6 Write unit tests for action vector translation and validation

## 7. Frame Reconstruction

- [x] 7.1 Implement `FrameRenderer` in `src/deployer/frame_renderer.py`: reconstruct local voxel grid from pyCraft chunk data packets
- [x] 7.2 Implement first-person view raycasting from player position/rotation through voxel grid to produce RGB image
- [x] 7.3 Implement frame preprocessing: resize to configured resolution, normalize to `[0,1]` float32, permute to `(C,H,W)` tensor — matching `MineStudioAgentEnv._preprocess_frame`
- [x] 7.4 Write unit tests for voxel grid construction and frame preprocessing output shape/dtype

## 8. ServerEnv Adapter

- [x] 8.1 Implement `ServerEnv` in `src/deployer/env.py` with `reset() -> Tensor`, `step(action: Tensor) -> Tuple[Tensor, float, bool, dict]`, `close() -> None` matching `MineStudioAgentEnv` interface
- [x] 8.2 Wire `reset()` to `ServerConnector.connect()` + `FrameRenderer` initial frame
- [x] 8.3 Wire `step()` to `ActionExecutor` + `ActionThrottler` + observation from next server tick
- [x] 8.4 Wire `close()` to `SessionManager.shutdown()` + resource cleanup
- [x] 8.5 Write unit tests for `ServerEnv` interface compliance with mocked connector

## 9. Safety Filters

- [x] 9.1 Implement `SafetyFilter` in `src/deployer/safety.py` with filter registry: `BedrockFilter`, `LavaFilter`, `VoidFilter`, `CooldownFilter`
- [x] 9.2 Implement bedrock breaking prevention: check target block ID against bedrock, block action if matched
- [x] 9.3 Implement lava interaction prevention: check adjacent block types for lava, block placement if adjacent
- [x] 9.4 Implement void fall prevention: check player Y-coordinate against configurable threshold, trigger emergency jump
- [x] 9.5 Implement action cooldown filter: track last action timestamp per action type, reject if within cooldown window
- [x] 9.6 Implement per-filter enable/disable via `DeployConfig.safety` toggles and violation logging
- [x] 9.7 Write unit tests for each filter's blocking behavior and enable/disable toggle

## 10. Logging and Monitoring

- [x] 10.1 Implement structured logging setup in `src/deployer/logging.py`: configure `logging` module with JSON formatter, file handler with rotation, optional stdout handler
- [x] 10.2 Implement action logger: log timestamp, action vector summary, and resulting position on each step
- [x] 10.3 Implement position tracker: periodic task logging `(x, y, z)` every 5 seconds
- [x] 10.4 Implement server event logger: register pyCraft event handlers for chat messages, player joins, deaths; log with timestamps
- [x] 10.5 Write unit tests for log output format and rotation behavior

## 11. Graceful Shutdown

- [x] 11.1 Implement signal handlers for SIGINT and SIGTERM in `src/deployer/shutdown.py`
- [x] 11.2 Implement shutdown sequence: save agent state via `SessionManager`, flush action queue via `ActionThrottler`, send disconnect packet via `ServerConnector`, exit with code 0
- [x] 11.3 Write unit tests for shutdown sequence ordering and state save

## 12. CLI Entry Point

- [x] 12.1 Implement `wally-deploy` CLI in `src/deployer/cli.py`: argparse with `--server`, `--checkpoint`, `--goal-frame`, `--config`, `--record`, `--output-dir` arguments following existing CLI patterns
- [x] 12.2 Implement CLI `main()`: load `DeployConfig` (from YAML + CLI overrides), load LeWorldModel checkpoint, construct `ServerEnv`, `PlannerProtocol` adapter, `AgentLoop`, and run episode
- [x] 12.3 Register `wally-deploy` entry point in `pyproject.toml` `[project.scripts]`
- [x] 12.4 Write unit tests for CLI argument parsing and config merging

## 13. Integration Tests

- [x] 13.1 Write integration test: full deployment pipeline with mocked pyCraft connection (connect → reset → step × N → close), verifying `ServerEnv` produces valid tensors and `ActionExecutor` receives valid actions
- [x] 13.2 Write integration test: reconnection flow with mocked disconnect/reconnect cycle verifying state persistence and restoration
- [x] 13.3 Run `uv run ruff check .` and `uv run mypy` to verify lint and type compliance
- [x] 13.4 Run `uv run pytest` to verify all tests pass
