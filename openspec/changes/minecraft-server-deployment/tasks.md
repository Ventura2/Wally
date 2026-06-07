## 1. Setup and Infrastructure

- [ ] 1.1 Create `src/deployer/` package structure with `__init__.py`
- [ ] 1.2 Add `pyCraft` dependency to `pyproject.toml`
- [ ] 1.3 Create `src/deployer/config.py` with `DeployConfig` dataclass (server address, credentials, model path, goal spec, safety bounds)
- [ ] 1.4 Add YAML config loader for deployment configuration
- [ ] 1.5 Create base exception classes (`ConnectionError`, `AuthenticationError`, `DeploymentError`)

## 2. Server Connector Module

- [ ] 2.1 Create `src/deployer/server_connector.py` with pyCraft wrapper class
- [ ] 2.2 Implement `connect(host, port, username)` method for basic connection
- [ ] 2.3 Implement protocol version negotiation and server ping
- [ ] 2.4 Add support for Minecraft versions 1.8-1.20+ (pyCraft version mapping)
- [ ] 2.5 Implement connection state tracking (disconnected, connecting, connected)
- [ ] 2.6 Add connection event listeners (on_connect, on_disconnect, on_error)

## 3. Authentication

- [ ] 3.1 Implement Microsoft OAuth authentication flow using pyCraft auth
- [ ] 3.2 Implement offline mode authentication (username-only)
- [ ] 3.3 Add credential validation and error handling
- [ ] 3.4 Implement authentication token caching for session reuse
- [ ] 3.5 Add authentication method selection based on server mode (online/offline)

## 4. Session Manager Module

- [ ] 4.1 Create `src/deployer/session_manager.py` with session lifecycle management
- [ ] 4.2 Implement `start_session()` - join server and initialize agent state
- [ ] 4.3 Implement `stop_session()` - graceful disconnect with state save
- [ ] 4.4 Implement heartbeat mechanism to detect connection liveness
- [ ] 4.5 Implement automatic reconnection with exponential backoff (1s, 2s, 4s, 8s, max 60s)
- [ ] 4.6 Add reconnection attempt counter with max limit (10 attempts)
- [ ] 4.7 Implement state persistence (last known position, inventory, goal progress)
- [ ] 4.8 Add state restoration on reconnection

## 5. Action Throttler Module

- [ ] 5.1 Create `src/deployer/action_throttler.py` with async queue-based throttler
- [ ] 5.2 Implement action queue with configurable depth threshold
- [ ] 5.3 Implement 20 TPS (50ms) action execution timing
- [ ] 5.4 Add server TPS monitoring and adaptive timing
- [ ] 5.5 Implement backpressure warning when queue depth exceeds threshold
- [ ] 5.6 Add action queue flush on shutdown

## 6. Action Execution

- [ ] 6.1 Create `src/deployer/action_executor.py` to translate planner actions to protocol packets
- [ ] 6.2 Implement movement action translation (forward, backward, turn, strafe) to position packets
- [ ] 6.3 Implement block interaction translation (break, place) to block action packets
- [ ] 6.4 Implement inventory action translation (craft, equip, use) to inventory packets
- [ ] 6.5 Add action validation (check action vector format and bounds)
- [ ] 6.6 Implement action execution callback registration

## 7. Safety Bounds and Filtering

- [ ] 7.1 Create `src/deployer/safety_filter.py` with configurable action filters
- [ ] 7.2 Implement bedrock breaking prevention (block ID check)
- [ ] 7.3 Implement lava interaction prevention (block type check)
- [ ] 7.4 Implement void fall prevention (Y-coordinate check)
- [ ] 7.5 Add configurable action cooldowns (prevent action spam)
- [ ] 7.6 Add safety filter configuration (enable/disable specific filters)
- [ ] 7.7 Implement safety violation logging

## 8. Logging and Monitoring

- [ ] 8.1 Create `src/deployer/logger.py` with structured logging
- [ ] 8.2 Implement action logging (type, parameters, timestamp)
- [ ] 8.3 Implement position tracking (log position every 5 seconds)
- [ ] 8.4 Implement server event logging (chat messages, player joins, deaths)
- [ ] 8.5 Add log file output with rotation
- [ ] 8.6 Add optional stdout logging for debugging
- [ ] 8.7 Implement log format configuration

## 9. CLI Deployment Interface

- [ ] 9.1 Create `src/wally/cli/deploy.py` with `wally-deploy` command
- [ ] 9.2 Add CLI arguments: `--server`, `--checkpoint`, `--goal`, `--config`
- [ ] 9.3 Implement checkpoint loading and validation
- [ ] 9.4 Implement goal specification parsing (text, frame, or latent)
- [ ] 9.5 Add YAML config file support for advanced options
- [ ] 9.6 Implement deployment orchestration (load model → connect → run agent loop)
- [ ] 9.7 Add signal handlers for graceful shutdown (SIGINT, SIGTERM)
- [ ] 9.8 Add CLI help text and usage examples

## 10. Agent Integration

- [ ] 10.1 Create `src/deployer/agent.py` with main agent loop
- [ ] 10.2 Implement observation acquisition from server (current frame)
- [ ] 10.3 Implement planner invocation (current frame + goal → action sequence)
- [ ] 10.4 Implement action sequence execution via throttler
- [ ] 10.5 Implement replanning strategy (replan every N steps or fixed interval)
- [ ] 10.6 Add agent state machine (idle, planning, executing, reconnecting)

## 11. Testing

- [ ] 11.1 Create `tests/test_deployer/` directory structure
- [ ] 11.2 Write unit tests for `ServerConnector` with mocked pyCraft
- [ ] 11.3 Write unit tests for `SessionManager` reconnection logic
- [ ] 11.4 Write unit tests for `ActionThrottler` rate limiting
- [ ] 11.5 Write unit tests for `SafetyFilter` action blocking
- [ ] 11.6 Write integration test for full deployment flow (mock server)
- [ ] 11.7 Write CLI tests for argument parsing and config loading
- [ ] 11.8 Add test fixtures for mock Minecraft server responses

## 12. Documentation and Examples

- [ ] 12.1 Add deployment configuration examples to `examples/` directory
- [ ] 12.2 Document server setup requirements (vanilla, Paper, Spigot)
- [ ] 12.3 Add troubleshooting guide for common connection issues
- [ ] 12.4 Update README with deployment section
- [ ] 12.5 Add safety bounds configuration documentation
