## Context

Wally trains LeWorldModel planners on Windows-native Python (TheRock GPU
on the RX 6700 XT) but the only path to evaluate a trained agent
against the photoreal MineStudio renderer is on Linux. The upstream
`minestudio` wheel hardcodes a `launchClient.sh` launcher that calls
`xvfb-run` / `vglrun` and ships Linux-only LWJGL natives in its
`runtime/` directory. The Windows wheel does not ship `runtime/` or
`mcprec-6.13.jar` (confirmed on this host), so `wally-play` against a
local MineStudio JVM cannot actually start.

The Two-Environments setup (per `AGENTS.md`) places the collector in
the `wally-dev` Podman container inside WSL2 and all training /
planning / deployer code on Windows-native Python. The agent loop
already populates `info["pov"]` on every step
(`src/agent/loop.py:104` reads `info.get("pov")` and passes it to the
viewer), so the data exists in WSL2. What is missing is a wire from
the WSL2 process to the Windows host so a human can actually watch
the agent plan in the real MineStudio render.

The current mitigations don't help:

- **`wally-deploy` against a vanilla server** — works end-to-end
  (pyCraft + planner + voxel `FrameRenderer`), but the renderer is a
  hand-mapped color-cube ray-cast (`src/deployer/frame_renderer.py`
  has a 13-entry `BLOCK_COLORS` palette and no textures, no lighting,
  no shaders). It cannot help diagnose world-model texture / lighting
  / distance errors, which is the whole point of evaluating against
  MineStudio.
- **Headless evaluation** — works for batch metrics (success rate,
  latent distance, plan cost) via `tools/eval_goals.py`, but not for
  qualitative "is the agent looking at the right thing" debugging.

The chosen approach: a minimal read-only MJPEG HTTP relay inside the
WSL2 `wally-play` process. Read-only v1 — no actions back to the
agent, no auth, localhost-only. Any OpenCV client on Windows can
consume the stream (`cv2.VideoCapture`, browser, VLC, etc.).

## Goals / Non-Goals

**Goals:**

- Push the latest `info["pov"]` frame from inside the WSL2
  `wally-play` process to a local HTTP endpoint
  (`http://0.0.0.0:8081/stream`) with sub-second latency and no new pip
  dependencies (stdlib `http.server` + `cv2.imencode` only).
- Allow a Windows-side OpenCV client
  (`cv2.VideoCapture("http://localhost:8081/stream")`, browser, VLC)
  to consume the stream and show it.
- Add a single opt-in CLI flag (`--relay [--relay-port N] ...`) to
  `wally-play` that toggles the relay on/off, defaulting to off so
  existing runs are unchanged.
- Preserve the existing `FrameViewer` / `NullViewer` / `FrameViewerLike`
  contract and the existing `wally-collect` and `wally-deploy` paths.
  The change is purely additive on the `wally-play` side.
- Reuse the existing `q` / `Esc` quit semantics inside the loop and
  the existing `info["pov"]` plumbing. No new wire protocol.
- Document the exact WSL2 → Windows workflow (container command +
  Windows URL) in `AGENTS.md`.

**Non-Goals:**

- **No bidirectional control.** No keyboard → actions back to the
  agent. That is a v2 problem; the v1 surface is strictly POV out.
- **No `RelayFrameViewer` / no `wally-watch` CLI.** The Windows consumer
  is any OpenCV client. Adding a Python viewer class is wasted work
  when `cv2.VideoCapture` is one line.
- **No `wally-collect` changes.** The collector is a random-action
  data harvester with no live-viewing value (the user can sample
  shards post-run via `wally-validate samples`).
- **No `wally-deploy` changes.** The voxel renderer stays as-is; its
  limitation is now the explicit motivation for the relay, not a
  problem to fix in this change.
- **No auth / TLS / LAN exposure.** The relay binds to `0.0.0.0` inside
  the container but is reachable only from the WSL2 host's loopback
  via mirrored loopback.
- **No recording.** The streamed bytes are a passive POV; recording
  is what `wally-play --record` already does for trajectories.
- **No GPU compute in WSL2.** The broken librocdxg path means CPU
  torch only; gradient / hierarchical planners are slow there. The
  relay's purpose is to *show* the agent planning, not to make
  planning fast.

## Decisions

### 1. Server: stdlib `http.server.ThreadingHTTPServer` + generator handler

We use `http.server.BaseHTTPRequestHandler` with a generator-style
`do_GET()` that yields `multipart/x-mixed-replace` chunks. The
`BaseHTTPRequestHandler` already supports generator handlers in
CPython (the iterator protocol works for chunked responses). The
server is a `ThreadingHTTPServer` so multiple Windows viewers can
connect without serialising.

**Why not Flask / aiohttp / starlette?** Zero new dependencies. The
relay is ~120 LoC, ships with the WSL2 environment's system Python
3.10, and works on both `wally-dev` (where `minestudio` is installed)
and on a bare CPython. Adding a web framework would also force the
WSL2 venv to be re-installed because the container has no `pip
install` policy that would accept new deps without a rebuild.

### 2. Buffer: single-slot `RelayBuffer` with `threading.Lock`

```
class RelayBuffer:
    def __init__(self, max_width=640, max_height=360, jpeg_quality=80):
        self._lock = threading.Lock()
        self._frame_jpeg: bytes | None = None
        self._frame_bgr: np.ndarray | None = None
        self._timestamp: float = 0.0

    def update(self, pov_rgb: np.ndarray | None) -> None: ...
    def snapshot(self) -> tuple[bytes, np.ndarray, float]: ...
```

Writers (`wally-play`'s per-step path) call `update(pov)` with the
latest POV. Readers (the HTTP handler) call `snapshot()` which
returns the cached JPEG bytes + a BGR copy + a timestamp. Single-slot
(not a queue) so a slow client never blocks a fast agent — the writer
always overwrites.

**Why JPEG-encode at write time, not at serve time?** Encoding
224×224×3 takes ~1–2 ms; if we did it on every poll, a 30 fps viewer
would burn 60–90 ms/s of CPU per connected client. Encoding once and
caching the bytes makes the serve path a single memcpy. Cost: a few
hundred KB of RSS for the cached BGR ndarray, which is negligible.

**Why downsample to 640×360?** The WSL2 → Windows hop is on the same
machine's loopback, but for parity with what `wally-deploy`'s voxel
renderer produces and to keep CPU encode cost down, we cap the
relayed resolution. Configurable via `--relay-max-size`.

### 3. Frame-rate back-pressure: configurable `min_frame_interval_ms`

The server's send loop sleeps `min_frame_interval_ms` (default 33 ms
≈ 30 fps) between writes. If no new frame is available it re-sends
the last cached frame. This decouples the planner's step rate from
the viewer's display rate and bounds CPU on the WSL2 side. If the
planner steps at 5 fps the viewer still sees a smooth 30 fps by
re-displaying the latest frame.

**Alternative considered:** send-only-on-update. Rejected because
Windows-side OpenCV consumers can flicker (the `waitKey(1)` loop
expects a frame every iteration).

### 4. Health endpoint: `GET /healthz`

Returns `200 OK` with `text/plain` body `ok\n`. The Windows side polls
this once at startup to block on the WSL2 server being ready, then
opens `GET /stream`. Avoids a race where `cv2.VideoCapture.open`
returns a half-broken stream because the server isn't up yet.

### 5. CLI flag: `--relay [--relay-port N] [--relay-host H] [--relay-max-size WxH] [--relay-jpeg-quality Q] [--relay-min-frame-interval-ms MS]`

Added to `wally-play` only. The collector (`wally-collect`) is
headless by design (random actions, save to disk, sample later), and
the deployer (`wally-deploy`) renders locally on Windows (no need to
cross the env boundary). The relay buys something only in the one
case where MineStudio renders photoreal frames in WSL2 and the user
is on Windows — that's `wally-play`.

Defaults: port 8081, host `0.0.0.0`, max 640×360, jpeg quality 80,
min frame interval 33 ms. All opt-in via `--relay`; default
`wally-play` behavior is unchanged.

In `wally-play` main, when `--relay` is set:

1. Construct a `RelayBuffer` with the configured
   `max_width × max_height × jpeg_quality`.
2. Construct a `RelayHTTPServer(host, port, buffer,
   min_frame_interval_ms)` and call `server.start()`.
3. Pass the buffer into `AgentLoop(env, planner, config,
   relay=buffer, viewer=NullViewer())` (the local `FrameViewer` is
   disabled when the relay is on — the stream *is* the viewer; you
   don't want a second OpenCV window popping up in the WSL2 host).
4. On the `finally` block: `buffer.update(None)` and
   `server.stop()`, both wrapped in `try/except` so a relay error
   cannot prevent env teardown.

### 6. AgentLoop hook: optional `relay: RelayBuffer | None` parameter

The cleanest hook for `relay.update(pov)` is alongside the existing
`self._viewer.show(pov, ...)` call at `src/agent/loop.py:104-105`:

```python
pov = info.get("pov") if info else None
if self._relay is not None:
    self._relay.update(pov)
self._viewer.show(pov, info=viewer_info)
```

`relay` is a new optional kwarg on `AgentLoop.__init__`, default
`None`. No signature break. Tests that don't pass a relay are
unaffected.

### 7. CPU torch in `wally-dev` container

`wally-play` imports `torch` and `LatentRollout.from_checkpoint(...)`
(see `src/agent/play.py:112`). The `wally-dev` Podman container has
`minestudio` (system Python 3.10) but no torch. We add CPU-only
torch to the container build:

```
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

CEM planner works without torch (pure numpy). Gradient MPC and
hierarchical planners need it but will be 10–50× slower than the
TheRock GPU path on Windows. This is documented as a watch-the-agent
loop, not a fast-evaluation loop. AGENTS.md gets a one-line note
under the planner table.

**Alternative considered:** mirror the Windows TheRock setup into
the container. Rejected — duplicated env, broken for RDNA2 anyway
(librocdxg issue), and the relay's value is qualitative ("watch the
agent plan"), not quantitative ("evaluate 1000 episodes").

### 8. Checkpoint flow: project is already mounted at /workspace

The `wally-dev` container is started with the project root mounted
at `/workspace` (per AGENTS.md line 48:
`cd /workspace && PYTHONPATH=src python3 -m wally.cli.collect ...`).
So checkpoints at `/workspace/checkpoints/<name>.pt` are directly
accessible from inside the container. No `podman cp`, no checkpoint
server, no volume dance. The user just runs
`wally-play --checkpoint /workspace/checkpoints/<name>.pt` from
inside the container.

### 9. Connection topology: WSL2 → Windows via mirrored loopback

WSL2 mirrors the Linux guest's loopback to the Windows host's
`localhost` for ports the guest binds. So when the relay binds to
`0.0.0.0:8081` inside the container, Windows can reach it at
`http://localhost:8081/stream`. Documented in `AGENTS.md` and in
the `wally-play --help` text. Alternative URL form for older WSL2
versions: `http://$(wsl hostname -I | awk '{print $1}'):8081/stream`.

### 10. Lifecycle / shutdown

- The HTTP server thread is `daemon=True`. When `wally-play` returns,
  the process exits and the thread dies with it. No explicit
  `server.shutdown()` needed for normal exit, but we call it anyway
  in the `finally` block to free the port promptly.
- `RelayBuffer.update(None)` clears the cached frame so a client that
  connects after the agent stops sees an empty stream and exits
  cleanly (the HTTP handler yields a final blank JPEG with a 1-second
  delay then closes the connection).
- The existing `loop.py` KeyboardInterrupt path stays as-is; the
  relay's `update(None)` + `server.stop()` in the `finally` block
  cover graceful and exception exits.

## Risks / Trade-offs

- **Single-slot buffer drops intermediate frames under load** →
  acceptable: the viewer always shows the most recent frame, which
  is what humans expect. The trade-off is no replay; if a frame is
  missed it is gone.
- **No back-pressure on the writer side** → the planner is already
  paced by MineStudio's step loop (≤ 20 Hz), so we never write
  faster than ~20 fps.
- **JPEG compression is lossy on the streamed frame** → the
  source-of-truth frame is still the uncompressed `info["pov"]`
  ndarray used by `wally-play`. Only the wire bytes are JPEG. The
  Windows viewer decompresses back to BGR before rendering.
- **CPU torch in WSL2 is 10–50× slower than TheRock GPU** →
  documented as a known limitation. The relay's value is
  qualitative (watch the agent), not quantitative (run benchmarks).
- **WSL2 mirrored loopback has historically had edge cases on older
  Windows builds (≤ 1903)** → mitigated by the documented fallback
  URL using `wsl hostname -I`.
- **No encryption on the wire** → acceptable because the relay is
  bound to the WSL2 guest's loopback. If a future user wants LAN
  exposure they must add TLS themselves.
- **Server has no rate limit / connection cap** → a malicious local
  process could DoS the agent by spamming `GET /stream`. Acceptable
  for a research tool with no LAN exposure; v2 can add a connection
  cap.

## Migration Plan

No migration needed — the change is purely additive on `wally-play`
plus a one-line container Dockerfile change.

1. Build: implement `src/agent/relay.py`, `--relay` flags in
   `wally-play`, the optional `relay` parameter on `AgentLoop`, and
   the `relay_*` fields on `AgentConfig`.
2. Container: add CPU torch to `wally-dev`.
3. Verify: start `wally-play --relay` inside the container, open
   `http://localhost:8081/stream` in a browser from Windows,
   confirm frames appear, and press `q` / `Esc` in a
   `cv2.VideoCapture`-based viewer to stop cleanly.
4. Document: add the new row to the AGENTS.md live-viewer table with
   the exact `wally-play --relay` invocation and the Windows-side
   URL.
5. Rollback: the default of `--relay` is off, so a no-op rollback is
   just "don't pass the flag." Reverting the commits is the hard
   rollback. No data migration, no state.

## Open Questions

- Should `wally-play` also expose a `?quality=NN` and `?maxsize=WxH`
  query parameter on the URL so a Windows-side client can tune the
  stream without restarting the agent? Nice-to-have; not blocking
  for v1.
- Should the relay expose a `GET /snapshot.jpg` endpoint that
  returns the latest frame as a single JPEG (for `wally-watch` style
  tools that poll at 1 Hz instead of streaming)? Defer to v2.
- Should the relay be moved to a shared agent-side hook so any
  future CLI (e.g. a `wally-eval` that runs trained policies) gets
  it for free? Defer — premature abstraction.
