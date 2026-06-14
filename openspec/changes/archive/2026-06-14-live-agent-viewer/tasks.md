## 1. Expose full-resolution POV from environment

- [x] 1.1 Modify `src/collector/env.py` so `MineStudioEnv.step()` returns `obs_dict["pov"]` in the `info` dict alongside the existing `image` (224×224). Keep the `(image, reward, done, info)` return shape unchanged.
- [x] 1.2 Modify `src/agent/env.py` so `MineStudioAgentEnv.step()` forwards `pov` from the inner env's `info` into the outer `info` dict, without disturbing the preprocessed frame tensor.
- [x] 1.3 Add a unit test verifying that `info["pov"]` is present after `MineStudioEnv.step()` and that its shape is `(360, 640, 3)` (mock the underlying sim).

## 2. Implement FrameViewer module

- [x] 2.1 Create `src/agent/viewer.py` with a `FrameViewer` class that lazy-imports `cv2`, calls `cv2.imshow(window_name, frame_bgr)` in `show(pov, info=None)`, draws an optional HUD (step count, plan cost, FPS) via `cv2.putText`, and runs `cv2.waitKey(1)` for non-blocking display.
- [x] 2.2 Implement `FrameViewer.should_quit()` returning `True` when the user has pressed `q` or `Esc` since the last call.
- [x] 2.3 Implement `FrameViewer.close()` that calls `cv2.destroyWindow(window_name)` and `cv2.destroyAllWindows()` and clears the lazy-imported module reference.
- [x] 2.4 Add a `NullViewer` no-op with the same interface (`show` / `should_quit` / `close`) for headless runs and tests.
- [x] 2.5 Add a unit test for `NullViewer` (no exceptions, `should_quit` always `False`).

## 3. Wire viewer into AgentLoop

- [x] 3.1 Add an optional `viewer` parameter to `AgentLoop.__init__()` defaulting to `None`; in `__init__`, normalize `None` to a `NullViewer`.
- [x] 3.2 In `AgentLoop.run_episode()`, call `self._viewer.show(pov=info.get("pov"), info=info)` after each `env.step()`; call `self._viewer.should_quit()` and break the loop with `interrupted=True` if it returns `True`.
- [x] 3.3 Make sure the `viewer.close()` is called from the `KeyboardInterrupt` branch and from the success/done branches in `run_episode()`.
- [x] 3.4 Add a unit test for `AgentLoop` with a stub viewer that records `show` calls; verify the loop calls `show` once per step and breaks on `should_quit`.

## 4. Wire `--viewer` flag into wally-play

- [x] 4.1 Add `--viewer {cv2,none}` (default `cv2`) and `--no-viewer` to the `argparse` in `src/agent/play.py`.
- [x] 4.2 In `play.py:main()`, build a `FrameViewer` when `args.viewer == "cv2"` and a `NullViewer` otherwise; pass the viewer to `AgentLoop`.
- [x] 4.3 Ensure `viewer.close()` is called in a `try` / `finally` around `loop.run_episode()`.
- [x] 4.4 Log a one-line INFO message when the viewer is enabled and another when it's disabled.

## 5. Tests and verification

- [x] 5.1 Run `pytest -m smoke -x --tb=short` from `.venv-windows` and confirm all smoke tests still pass (including the new viewer tests).
- [x] 5.2 Run `ruff check .` and `mypy` and fix any new lints/type errors.
- [x] 5.3 Manually verify with a stub env: `python -m agent.play --checkpoint <stub> --goal-frame <stub> --viewer cv2` opens a window; the same command with `--viewer none` runs headlessly.
