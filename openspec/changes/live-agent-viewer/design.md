## Context

Today, `wally-play` runs the goal-conditioned agent loop and prints a
summary line at the end (`steps`, `final_cost`, `duration_seconds`). There
is no way to watch what the agent is seeing in real time.

The agent loop is composed of:

- `src/agent/play.py` — CLI entry, builds env + planner + loop
- `src/agent/loop.py` — `AgentLoop.run_episode()` (the per-step
  plan → execute → observe → replan loop)
- `src/agent/env.py` — `MineStudioAgentEnv.step()` (Gym-like wrapper)
- `src/collector/env.py` — `MineStudioEnv.step()` (raw MineStudio
  bridge, returns `obs_dict["image"]` only)

MineStudio's `MinecraftSim` returns an obs dict with `pov` (640×360,
native resolution) and `image` (resized, e.g. 224×224). The collector
env currently drops `pov`. This change exposes it.

MineStudio also ships `minestudio.simulator.minerl.human_play_interface`
for human-in-the-loop play, but it is the wrong shape for "watch a
trained agent" (it expects human keyboard input and bypasses the agent
loop entirely). We use OpenCV's `cv2.imshow` instead.

## Goals / Non-Goals

**Goals:**
- A live window during `wally-play` that displays the agent's POV.
- A `--viewer` CLI flag with safe defaults (cv2 on, none for headless).
- Zero new pip dependencies; no measurable performance regression.
- Pure observer — viewer never mutates agent state or planner output.

**Non-Goals:**
- Side-by-side goal frame, planner HUD, video recording — deferred.
- Live viewer for `wally-deploy` (server-side). The deployer doesn't
  render a POV today; adding one would be a separate change.
- MineStudio human-play-interface integration. Wrong shape for this
  use case.
- Image recording. `wally-play --record` already covers trajectory
  export; recording live view pixels is a separate, larger feature.

## Decisions

### Decision 1: Plumb `pov` through the `info` dict, not the return tuple
The `MineStudioEnv.step()` and `MineStudioAgentEnv.step()` signatures
return `(frame, reward, done, info)`. Adding a fifth tuple element
would break every caller. The `info` dict is the conventional place
for observation metadata and is unused by the agent loop beyond
debug printing. We add `info["pov"]` and leave the tuple shape alone.

**Alternative considered**: Make `pov` a property on
`MineStudioAgentEnv` (`env.last_pov`). Rejected — agents and tests
already track state through return values; mixing property access
with return values is inconsistent.

### Decision 2: Optional `viewer` parameter on `AgentLoop`, not a
mandatory one
`AgentLoop` is used by tests and by `wally-play`. Tests don't need a
viewer and shouldn't pay the `cv2` import cost. The viewer is purely
an observer and is a non-mandatory dependency. We default to
`viewer=None`.

**Alternative considered**: Subclass `AgentLoop` into
`ViewingAgentLoop`. Rejected — adds a class hierarchy for a single
flag and forces callers to pick the right one.

### Decision 3: `cv2.imshow` + `cv2.waitKey(1)` over pyglet/imgui
OpenCV is already a transitive dep through MineStudio's data pipeline.
`cv2.imshow` is non-blocking and adds <1 ms per step. Pyglet/imgui
(MineStudio's `human_play_interface` choice) would add ~10 MB of
import-time overhead and need a new entry point, with no benefit for
a passive observer that doesn't need input handling.

### Decision 4: Lazy import of `cv2` in `viewer.py`
The collector's collector and agent packages are imported by code
that may run on headless servers (CI, training pipelines). The
viewer is opt-in via `--viewer cv2` and `cv2` is only needed when a
`FrameViewer` is actually constructed. Lazy import keeps the
`import agent` path free of `cv2` and avoids headless-test failures
in environments where `cv2` cannot open a display.

### Decision 5: `NullViewer` instead of branching on `viewer is None`
inside `AgentLoop`
`AgentLoop` calls `viewer.show(...)` and `viewer.should_quit()` on
every step. A `None` check on every step is brittle (typos, refactors
that pass a wrong type). A `NullViewer` no-op object is one cheap
allocation at construction time and removes the special case from
the loop's hot path. Same interface as `FrameViewer`.

## Risks / Trade-offs

- **[Risk]** `cv2` import time on first `--viewer cv2` run adds ~0.5s.
  → **Mitigation**: lazy import + a one-time log message; negligible
  compared to model load and MineStudio sim startup.
- **[Risk]** On a headless box (no `$DISPLAY`, no Windows desktop),
  `cv2.imshow` raises. → **Mitigation**: catch the exception in
  `FrameViewer.__init__`, log a clear error, and exit with a
  non-zero status pointing the user to `--viewer none`.
- **[Risk]** `info["pov"]` doubles peak memory per env step (we hold
  both `image` tensor and `pov` ndarray briefly). → **Mitigation**:
  `pov` is uint8 HxWx3 = 691 KB; trivial compared to the model and
  the agent's other buffers.
- **[Trade-off]** HUD overlay uses `cv2.putText` (blocky default font).
  Acceptable for a debug tool; not a polished UI.

## Migration Plan

This is an additive change with no breaking signature changes. Existing
callers of `MineStudioEnv.step()` and `MineStudioAgentEnv.step()`
continue to work — they just don't read the new `info["pov"]` key.

Rollout:
1. Land the change behind the default `--viewer cv2` (on by default).
2. CI smoke tests run with `--viewer none` (no display available).
3. If a user reports `cv2` import issues, they opt out with
   `--viewer none`. No code change needed.

No data migration; no checkpoint changes; no deploy steps.

## Open Questions

- Should `--viewer` accept a future `record` value that writes an MP4
  alongside the live view? Deferred — can be added later without
  breaking the existing flag set.
- Should the viewer accept a `window_name` override? Probably yes for
  multi-process runs, but not needed yet.
