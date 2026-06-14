## Why

`wally-play` runs the goal-conditioned agent loop against a live MineStudio
session, but produces no visual feedback — operators can only see the episode
summary printed at the end (`steps`, `final_cost`, `duration_seconds`). For
debugging planner behavior, evaluating what the agent actually perceives, and
demonstrating the system to others, we need a live POV window that updates
step-by-step as the agent plans and acts.

## What Changes

- Expose the full-resolution `pov` frame (640×360) from the MineStudio
  observation dict. Today `MineStudioEnv.step()` returns only the resized
  `image` (224×224) and silently drops `obs_dict["pov"]`. The raw frame is
  plumbed through `MineStudioAgentEnv.step()` and surfaced in the `info`
  dict so callers can render it without changing the (frame, reward, done,
  info) return contract.
- Add a new `FrameViewer` module in `src/agent/viewer.py` that wraps
  OpenCV's `cv2.imshow` to display the agent's POV in real time, with an
  optional HUD overlay (step count, plan cost, FPS, done flag) and a
  `q`/`Esc` quit detector for clean episode shutdown.
- Wire a `--viewer {none,cv2}` CLI flag into `wally-play` (default `cv2`).
  When set, `wally-play` builds a `FrameViewer`, passes it to
  `AgentLoop.run_episode()`, and the loop calls `viewer.show(...)` after
  each environment step. `--viewer none` (and the equivalent
  `--no-viewer`) disables it for headless / CI runs.
- `AgentLoop.run_episode()` accepts an optional `viewer` parameter and
  calls `viewer.show()` / `viewer.should_quit()` after each step.

## Capabilities

### New Capabilities
- `live-agent-viewer`: A passive OpenCV-based window that displays the
  agent's POV frame in real time during `wally-play` episodes, with
  optional HUD overlay and graceful user-initiated exit.

### Modified Capabilities
- `minecraft-environment-integration`: The `MineStudioEnv.step()` and
  `MineStudioAgentEnv.step()` return contracts extend to expose the
  full-resolution POV frame in the `info` dict. `wally-play` adds a
  `--viewer` flag and a viewer is now part of the agent loop's optional
  responsibilities. `AgentLoop` gains an optional `viewer` parameter.

## Impact

- **New file**: `src/agent/viewer.py` (the `FrameViewer` class).
- **Modified files**:
  - `src/collector/env.py` — return `pov` in the `info` dict.
  - `src/agent/env.py` — plumb `pov` through to `info`.
  - `src/agent/loop.py` — accept optional `viewer` and call it per step.
  - `src/agent/play.py` — add `--viewer` flag, build viewer, pass to loop.
- **Dependencies**: OpenCV (`cv2`) is already a transitive dep through
  MineStudio; no new pip dependencies required.
- **Headless / CI safety**: `--viewer none` is the no-op path; no window
  is created, `cv2` is not imported at runtime in that mode (deferred via
  lazy import in `viewer.py`).
- **Performance**: `cv2.imshow` is non-blocking; with `cv2.waitKey(1)` the
  viewer adds <1 ms per step. No measurable impact on planner throughput.
