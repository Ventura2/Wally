# MJPEG POV Relay

## Purpose

Expose the latest `info["pov"]` frame from a running `wally-play` agent loop over an HTTP MJPEG stream on loopback, so a remote OpenCV-based viewer (or browser) on the host can watch the photoreal MineStudio render while the planner runs in a separate environment. Read-only; no control channel. Owns the buffer-lock + multipart serialization.

## Requirements

### Requirement: RelayBuffer holds the latest POV frame for streaming consumers

A `RelayBuffer` class in `src/agent/relay.py` SHALL hold the most recent POV frame submitted by the `wally-play` agent loop and make it available to any number of concurrent HTTP clients. The buffer MUST use a `threading.Lock` to serialize writers from readers, MUST JPEG-encode the frame on the writer's thread (not the reader's), and MUST expose a `snapshot()` method that returns the cached JPEG bytes plus a fresh BGR ndarray copy plus a monotonic timestamp.

#### Scenario: update then snapshot returns the new frame

- **WHEN** `buffer.update(pov)` is called with a valid `(H, W, 3)` uint8 RGB ndarray, then `buffer.snapshot()` is called from a different thread
- **THEN** the snapshot's JPEG bytes decode to the same shape, the BGR copy's values match `cv2.cvtColor(pov, cv2.COLOR_RGB2BGR)`, and the snapshot's timestamp is `>=` the timestamp recorded at update time

#### Scenario: snapshot returns the last frame when no new update has arrived

- **WHEN** `buffer.update(pov1)` is called, then `snapshot()` is called twice in a row without any intervening update
- **THEN** both snapshots return identical JPEG bytes and identical BGR arrays (the cached frame, not a re-encoded copy)

#### Scenario: downsampling reduces the relayed resolution

- **WHEN** the buffer is constructed with `max_width=320, max_height=180` and `update(pov)` is called with a `(720, 1280, 3)` frame
- **THEN** the snapshot's JPEG bytes decode to `(180, 320, 3)` and the aspect ratio is preserved (letterboxed, not cropped)

#### Scenario: update with None clears the slot

- **WHEN** `buffer.update(None)` is called after a valid update
- **THEN** subsequent `snapshot()` calls return `(None, None, 0.0)` and HTTP clients see no further frames on `/stream`

### Requirement: MJPEG HTTP server exposes the latest frame as a stream

A `RelayHTTPServer` in `src/agent/relay.py` SHALL serve the latest cached POV frame from the `RelayBuffer` over HTTP using the `multipart/x-mixed-replace` content type. The server MUST be built on `http.server.ThreadingHTTPServer` and a custom `BaseHTTPRequestHandler` with a generator-style `do_GET`. The server MUST accept a `min_frame_interval_ms` parameter (default 33 ms) and MUST re-send the cached frame at that cadence when no new frame is available.

#### Scenario: GET /stream yields multipart MJPEG

- **WHEN** an HTTP client sends `GET /stream` while at least one frame has been pushed to the buffer
- **THEN** the server responds with `200 OK`, `Content-Type: multipart/x-mixed-replace; boundary=frame`, and a body of one or more `Content-Type: image/jpeg` parts separated by the boundary

#### Scenario: GET /healthz returns ok

- **WHEN** an HTTP client sends `GET /healthz` at any time, including before any frame has been pushed
- **THEN** the server responds with `200 OK`, `Content-Type: text/plain`, and body `ok\n` within 100 ms

#### Scenario: Multiple concurrent clients each get their own thread

- **WHEN** two HTTP clients each open `GET /stream` simultaneously
- **THEN** both connections receive frames at the configured `min_frame_interval_ms` cadence without either blocking the other (verified by each receiving at least one frame within 1 second)

#### Scenario: Server is a daemon thread

- **WHEN** `RelayHTTPServer.start()` is called
- **THEN** the underlying `threading.Thread` has `daemon=True`, so the thread does not prevent the Python process from exiting when the `wally-play` agent loop returns

#### Scenario: Unknown path returns 404

- **WHEN** an HTTP client sends `GET /anything-else` to the relay
- **THEN** the server responds with `404 Not Found`

### Requirement: wally-play exposes a --relay flag to start the relay

The `wally-play` CLI entry point MUST accept `--relay`, `--relay-port` (default 8081), `--relay-host` (default `0.0.0.0`), `--relay-max-size` (default `640x360`), `--relay-jpeg-quality` (default 80), and `--relay-min-frame-interval-ms` (default 33) flags. When `--relay` is not passed, no relay server is started and the existing behaviour is unchanged. When `--relay` is passed, the CLI MUST construct a `RelayBuffer` + `RelayHTTPServer`, start the server in a daemon thread, pass the buffer to `AgentLoop` as the new optional `relay` parameter, and call `buffer.update(info["pov"])` after every `env.step()` for the duration of the run. The CLI MUST also force `viewer=NullViewer()` when `--relay` is set so the relayed stream is not duplicated by a local OpenCV window.

#### Scenario: --relay defaults are sensible

- **WHEN** `wally-play --relay --checkpoint <ckpt>.pt --goal-frame <goal>.png --planner cem --viewer none` is run
- **THEN** the server binds to `0.0.0.0:8081`, the buffer is configured with `max_width=640, max_height=360, jpeg_quality=80`, and a Windows client can connect to `http://localhost:8081/stream` and see the latest POV frame updated at the agent's step rate

#### Scenario: omitting --relay preserves existing behaviour

- **WHEN** `wally-play --checkpoint <ckpt>.pt --goal-frame <goal>.png` is run (no `--relay` flag)
- **THEN** no HTTP server thread is started, no port is bound, and the local `FrameViewer` (or `NullViewer` if `--viewer none` was also passed) is used exactly as before

#### Scenario: --relay-port overrides the default port

- **WHEN** `wally-play --relay --relay-port 9999 ...` is run
- **THEN** the server binds to port 9999 and `GET http://localhost:9999/healthz` returns `ok\n`

#### Scenario: relay shuts down cleanly on episode end

- **WHEN** `wally-play --relay` finishes its episode (either by `done=True`, by `should_quit()`, or by `KeyboardInterrupt`)
- **THEN** the CLI calls `relay.update(None)` and `relay_server.stop()` in its `finally` block, the HTTP server thread terminates within 1 second, and the bound port is released

### Requirement: AgentLoop exposes an optional relay hook

`AgentLoop.__init__` in `src/agent/loop.py` MUST accept an optional `relay: RelayBuffer | None = None` keyword parameter. When `relay` is not `None`, the per-step path MUST call `self._relay.update(pov)` with `pov = info.get("pov")` from the most recent `env.step()` result, immediately adjacent to the existing `self._viewer.show(pov, ...)` call. When `relay is None`, the call is skipped. The change MUST NOT break the existing `show` / `should_quit` / `close` contract of the `FrameViewerLike` protocol.

#### Scenario: AgentLoop without a relay runs unchanged

- **WHEN** `AgentLoop(env, planner, config)` is constructed with no `relay` kwarg
- **THEN** the per-step path does not call `RelayBuffer.update` and the existing test suite in `tests/test_agent_loop.py` continues to pass without modification

#### Scenario: AgentLoop with a relay updates it on every step

- **WHEN** `AgentLoop(env, planner, config, relay=buffer)` is constructed and the env's `step()` returns `info["pov"]` of shape `(H, W, 3)` uint8
- **THEN** `buffer.snapshot()` returns a `(H, W, 3)` BGR ndarray matching `cv2.cvtColor(info["pov"], cv2.COLOR_RGB2BGR)` (or the configured downsampled shape) on the next iteration

### Requirement: WSL2 → Windows reachability works via mirrored loopback

The relay server MUST bind to `0.0.0.0` (or the value of `--relay-host`) inside the WSL2 container / process, and Windows hosts running WSL2 MUST be able to reach the stream at `http://localhost:<relay-port>/stream` without additional port forwarding. This is a property of WSL2's mirrored loopback and the documentation in `AGENTS.md` MUST include the exact URL.

#### Scenario: Windows client reaches the stream via localhost

- **WHEN** the relay is running inside a WSL2 container on port 8081
- **THEN** a Windows process running `cv2.VideoCapture("http://localhost:8081/stream")` opens the stream successfully and `cap.read()` returns `(True, frame)` for each new frame pushed by the agent

#### Scenario: Windows browser reaches the stream via localhost

- **WHEN** the relay is running inside a WSL2 container on port 8081
- **THEN** opening `http://localhost:8081/stream` in a Windows browser displays the latest POV frame as a continuously-updating MJPEG stream
