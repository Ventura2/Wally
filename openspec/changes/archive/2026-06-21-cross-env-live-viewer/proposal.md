## Why

Wally trains LeWorldModel planners on Windows-native Python (TheRock GPU
on the RX 6700 XT) but the only path to evaluate a trained agent against
the photoreal MineStudio renderer is to run it on Linux. The upstream
`minestudio` wheel hardcodes a `launchClient.sh` launcher that calls
`xvfb-run` / `vglrun` and ships Linux-only LWJGL natives in its
`runtime/` directory; the Windows wheel does **not** ship
`runtime/` + `mcprec-6.13.jar`. Confirmed on this machine:

```
package: D:\...\wally\.venv-windows\Lib\site-packages\minestudio
runtime/:       False
mcprec jar:     False
```

So `wally-play` against a local MineStudio JVM is dead on this Windows
host. The current fallbacks are:

- **`wally-deploy` against a vanilla server** with a voxel-grid
  `FrameRenderer` (a hand-mapped color-cube ray-cast in
  `src/deployer/frame_renderer.py`). Works end-to-end (pyCraft + planner)
  but the renderer is stylized — no textures, no lighting, no shaders.
  It cannot help diagnose world-model texture / lighting / distance
  errors.
- **Skip live viewing** — train on Windows, evaluate against synthetic
  data on Windows, or only inspect saved shards via
  `wally-validate samples` after the fact.

Neither is adequate for the core evaluation loop: "watch the trained
LeWM planner plan and act inside the real MineStudio render." The
photoreal frames only exist in the WSL2 container where the MineStudio
JVM can actually start. We need a thin, read-only wire from the WSL2
process to the Windows host — with no new pip dependencies, no auth, no
changes to the agent loop's contract — so any OpenCV client on Windows
can consume the latest `info["pov"]` while the planner runs in WSL2.

The agent loop already populates `info["pov"]` on every step
(`src/agent/loop.py:104` reads `info.get("pov")`), so the data is there.
What's missing is a wire.

## What Changes

- **Add `src/agent/relay.py`** (~120 LoC) with `RelayBuffer`
  (single-slot locked frame cache holding the latest
  `(jpeg_bytes, bgr_ndarray, timestamp)` triple) and `RelayHTTPServer`
  (stdlib `ThreadingHTTPServer` + a generator-style `BaseHTTPRequestHandler`
  serving `multipart/x-mixed-replace` at `GET /stream` and
  `GET /healthz`). Zero new pip dependencies.
- **Add `--relay` flag to `wally-play`** (the only place MineStudio
  actually runs in this stack today), with `--relay-port` (default
  8081), `--relay-host` (default `0.0.0.0`), `--relay-max-size`
  (default `640x360`), `--relay-jpeg-quality` (default 80), and
  `--relay-min-frame-interval-ms` (default 33). All relay flags are
  opt-in; default behavior of `wally-play` is unchanged.
- **Wire the relay into `wally-play`** via a small additive change to
  `AgentLoop.__init__`: an optional `relay: RelayBuffer | None`
  parameter, and a `self._relay.update(pov)` call next to the existing
  `self._viewer.show(pov, ...)` at `src/agent/loop.py:104-105`. No
  signature break — `relay` defaults to `None`.
- **Add CPU-only torch to the `wally-dev` container** so the gradient
  and hierarchical planners work (slowly) in WSL2. CEM works without
  torch but the other two need it. This is a single `pip install` line
  in the container build, with no AMD GPU bits (librocdxg in WSL2 is
  broken for RDNA2 per AGENTS.md).
- **No `RelayFrameViewer`, no `wally-watch` CLI, no changes to
  `wally-collect`, no changes to `wally-deploy`, no changes to the
  agent loop's `show` / `should_quit` / `close` contract.** The Windows
  consumer is any OpenCV client (`cv2.VideoCapture("http://localhost:8081/stream")`,
  a browser pointed at `/stream`, VLC, etc.).
- **Update `AGENTS.md`** live-viewer table to populate the
  WSL2→Windows MJPEG row with the exact `wally-play --relay` invocation
  inside the container and the Windows-side URL, and add a
  "wally-play in WSL2" quick-start section.

## Capabilities

### New Capabilities

- `mjpeg-pov-relay`: HTTP MJPEG server that exposes the latest
  `info["pov"]` frame from a running `wally-play` process over loopback,
  so a remote OpenCV-based viewer can stream it. Read-only; no control
  channel. Owns the buffer-lock + multipart serialization.

### Modified Capabilities

None. The `live-agent-viewer` capability added in the archived
`2026-06-14-live-agent-viewer` change is not touched.

## Impact

- **New code**: `src/agent/relay.py` (~120 LoC), `src/agent/play.py`
  flag wiring (~30 LoC), `src/agent/loop.py` optional `relay` parameter
  (~5 LoC), `src/agent/config.py` `relay_*` fields (~7 lines),
  `tests/test_relay.py` (new, ~120 LoC), `tests/test_play_cli.py`
  additions (~80 LoC), `wally-dev` container Dockerfile change (one
  `pip install` line for CPU torch + a 1-line AGENTS.md note about CPU
  planner speed).
- **No new runtime dependencies**. `http.server` and `threading` are
  stdlib; `cv2` is already a transitive dep of `minestudio` (and of
  `wally-play`'s existing `FrameViewer`).
- **Affected docs**: `AGENTS.md` live-viewer table + a new
  "wally-play in WSL2" quick-start section.
- **No GPU / training / model changes**. The training pipeline stays on
  Windows-native Python with TheRock.
- **No security surface added** beyond a localhost-only HTTP port. The
  relay binds to `0.0.0.0` inside the container but is reachable from
  the WSL2 host's loopback via mirrored loopback. No auth, no TLS.
- **Backwards compatible**: default behavior of `wally-play` is
  unchanged (relay off). `wally-collect`, `wally-deploy`, the
  `live-agent-viewer` capability, and the existing `FrameViewer` /
  `NullViewer` types are not touched.
- **Performance caveat (documented)**: in-WSL2 planners run on CPU
  torch only (the broken librocdxg path means no AMD GPU compute in
  WSL2). CEM is fine; gradient MPC and hierarchical are 10–50× slower
  than TheRock GPU on Windows. This is a watch-the-agent loop, not a
  fast-evaluation loop.
