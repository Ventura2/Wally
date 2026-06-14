## ADDED Requirements

### Requirement: FrameViewer displays agent POV in real time
A `FrameViewer` class in `src/agent/viewer.py` SHALL display the
agent's POV frame in an OpenCV window at the same rate as the
agent's environment step loop. The class MUST accept a POV frame
as a numpy `ndarray` of shape `(H, W, 3)` and a callable `info` dict
through a `show(pov, info=None)` method, and MUST use `cv2.waitKey(1)`
internally so it is non-blocking.

#### Scenario: Show a single frame
- **WHEN** `viewer.show(pov=frame)` is called with a valid BGR or RGB
  numpy array
- **THEN** the frame appears in the named OpenCV window within 1 step
  and `show` returns without raising

#### Scenario: HUD overlay shows step, cost, and FPS
- **WHEN** `show` is called with `info={"step": 42, "plan_cost": 0.31}`
  and `show_fps=True`
- **THEN** a text overlay is drawn on the frame containing the step
  count, plan cost (when present), and the rolling FPS measurement

#### Scenario: should_quit detects user input
- **WHEN** the user presses `q` or `Esc` while the viewer window has
  focus
- **THEN** `should_quit()` returns `True` on the next call

#### Scenario: Close destroys window
- **WHEN** `viewer.close()` is called
- **THEN** the OpenCV window is destroyed and `cv2` is no longer
  referenced by the viewer

### Requirement: FrameViewer is a passive observer
The `FrameViewer` MUST NOT mutate the frames it receives. It MUST NOT
call the environment, the planner, or any side-effecting function. Its
sole responsibilities are display, HUD rendering, and quit detection.

#### Scenario: Frames are not mutated
- **WHEN** `viewer.show(pov=frame)` is called
- **THEN** the input numpy array's dtype, shape, and contents are
  unchanged after the call returns

### Requirement: FrameViewer is optional and lazy-imported
`src/agent/viewer.py` MUST use a lazy import of `cv2` so that code
paths that never instantiate a `FrameViewer` do not pay the import
cost. A `NullViewer` (or equivalent no-op implementation) MUST be
provided for headless and test contexts and MUST have the same
`show` / `should_quit` / `close` interface as `FrameViewer`.

#### Scenario: NullViewer is a no-op
- **WHEN** a `NullViewer` is constructed and its `show` / `should_quit`
  / `close` methods are called
- **THEN** none of them raise, `should_quit` always returns `False`,
  and no OpenCV window is created

#### Scenario: cv2 import is deferred
- **WHEN** `src/agent/viewer.py` is imported
- **THEN** importing the module does not require `cv2` to be importable;
  `cv2` is only loaded the first time a `FrameViewer` is constructed
