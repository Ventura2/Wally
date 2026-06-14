## ADDED Requirements

### Requirement: Live POV frame is exposed by the environment
The `MineStudioEnv.step()` return tuple MUST include a `pov` key in the
`info` dict containing the full-resolution first-person frame (shape
`(360, 640, 3)`, dtype `uint8`, BGR or RGB consistent with MineStudio's
native obs dict) returned by the underlying `MinecraftSim` observation.
The existing 224×224 `image` frame in the env's return tuple MUST
continue to be returned unchanged for the agent loop's preprocessing.

#### Scenario: POV frame is plumbed through env step
- **WHEN** `MineStudioEnv.step(action)` is called and the underlying
  `MinecraftSim` returns `obs_dict` containing both `pov` and `image`
- **THEN** the returned `info` dict contains a `pov` key with the
  full-resolution frame and the function's other return values
  (`image`, `reward`, `done`) match the prior behavior

#### Scenario: Agent env surfaces POV in info
- **WHEN** `MineStudioAgentEnv.step(action)` is called and the
  underlying `MineStudioEnv` returns `info` containing `pov`
- **THEN** the agent env's returned `info` dict contains the same
  `pov` key and value, untouched by frame preprocessing

### Requirement: AgentLoop supports an optional viewer
`AgentLoop.run_episode()` MUST accept an optional `viewer` argument.
When a viewer is provided, the loop MUST call `viewer.show(pov, info)`
after each `env.step()` and MUST call `viewer.should_quit()` to detect
user-initiated shutdown. The viewer's return values MUST NOT alter the
agent's per-step action selection or planner behavior — the viewer is
purely a passive observer.

#### Scenario: Viewer is invoked per step
- **WHEN** the loop executes an environment step and a viewer is set
- **THEN** `viewer.show(...)` is called with the current observation
  and info dict, and the loop continues to the next step

#### Scenario: Viewer can trigger clean episode shutdown
- **WHEN** `viewer.should_quit()` returns `True` after a step
- **THEN** the loop closes the environment, returns an
  `EpisodeResult` with `interrupted=True` populated, and exits without
  raising

#### Scenario: Loop works without a viewer
- **WHEN** `AgentLoop.run_episode()` is called with `viewer=None`
  (the default)
- **THEN** the loop runs exactly as before — no viewer is invoked, no
  quit check is performed, and the return type is unchanged

### Requirement: wally-play CLI exposes viewer selection
`wally-play` MUST accept a `--viewer` flag with values `cv2` (default)
and `none`. When `--viewer cv2` is set, `wally-play` MUST instantiate
a `FrameViewer` and pass it to `AgentLoop.run_episode()`. When
`--viewer none` is set, `wally-play` MUST run with `viewer=None` and
MUST NOT import `cv2`. A convenience `--no-viewer` alias MUST be
accepted as equivalent to `--viewer none`.

#### Scenario: Default viewer is cv2
- **WHEN** the user runs `wally-play --checkpoint ... --goal-frame ...`
  with no `--viewer` flag
- **THEN** a `FrameViewer` is constructed and passed to the agent loop
  and a POV window appears during the episode

#### Scenario: --viewer none disables the window
- **WHEN** the user runs `wally-play ... --viewer none` (or `--no-viewer`)
- **THEN** no `FrameViewer` is constructed, `cv2` is not imported, and
  the loop runs headlessly

#### Scenario: --viewer cv2 is explicit
- **WHEN** the user runs `wally-play ... --viewer cv2`
- **THEN** the behavior is identical to the default (`--viewer cv2`)
