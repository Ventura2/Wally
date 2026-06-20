# src/agent — agent loop, viewer, MJPEG relay

This subpackage owns the goal-conditioned agent loop and the two ways its POV frame leaves the process.

## Files

- `loop.py` — `AgentLoop` (per-step env→planner→action loop; replan every `replan_interval` steps with warm-start shifting)
- `env.py` — `MineStudioAgentEnv` adapter for the MineStudio engine
- `planner_factory.py` — builds CEM / gradient / hierarchical planners from a `LatentRollout`
- `protocol.py` — `PlanResult`, `EpisodeResult`, `PlannerProtocol` and the flat/hierarchical adapters
- `buffer.py` — `TrajectoryBuffer` (frames + actions, optional recording)
- `config.py` — `AgentConfig` (pydantic; `relay_*` fields for the WSL2→Windows relay)
- `viewer.py` — `FrameViewer` (OpenCV `cv2.imshow`), `NullViewer` (no-op), `FrameViewerLike` protocol
- `relay.py` — `RelayBuffer` (single-slot locked JPEG/BGR/timestamp cache) + `RelayHTTPServer` (stdlib `http.server` MJPEG multicast at `/stream`, `/healthz`)
- `play.py` — `wally-play` CLI entry point

## Per-step contract

`AgentLoop` calls `env.step(action)`, which returns `(next_frame, reward, done, info)`. The `info` dict is the only place the POV leaves the env. The per-step path then:

1. If `self._relay is not None`, calls `self._relay.update(info.get("pov"))` — pushes the latest frame to the MJPEG relay.
2. Calls `self._viewer.show(pov, info=...)` — renders the local OpenCV window.
3. If `self._viewer.should_quit()`, breaks the loop.

`relay` is an optional kwarg on `AgentLoop.__init__` (default `None`); the existing `show`/`should_quit`/`close` contract of the viewer is unchanged. When `wally-play --relay` is set, the CLI builds the loop with `viewer=NullViewer()` (the relay replaces the local window) and `relay=buffer`.

## Live viewer / MJPEG relay

Two production paths; details in [`docs/live-viewer.md`](../../docs/live-viewer.md):

- **Windows**: `wally-deploy --viewer cv2` against a local vanilla server (voxel-grid renderer)
- **WSL2**: `wally-play --relay` inside the `wally-dev` container, stream the MJPEG at `http://localhost:8081/stream` to any OpenCV client on Windows

Tests for the relay live in `tests/test_relay.py`; for the play CLI wiring, in `tests/test_play_cli.py::TestRelayEndToEnd`.

## Known planner local minimum: inventory-stuck

All three planners (`cem`, `gradient`, `hierarchical`) currently converge on opening/closing the inventory forever when the goal latent is far from the agent's current state. The world model scores "inventory open" as a near-constant latent that's marginally close to many goal latents, so the CEM elites latch onto it. Symptom: the relay stream shows the inventory UI opening and closing in a tight loop.

Demo workaround (one-line, in `loop.py` — zeroes column 12 of the planned action tensor before stepping):

```python
action = plan_actions[action_index]
if action.dim() == 1 and action.shape[-1] > 12:
    action = action.clone()
    action[12] = 0.0
```

Production fix: add the same penalty to the CEM cost function in `src/wally/planner/plan.py` so the planner *learns* to avoid the action:

```python
def cost_fn(actions: torch.Tensor) -> torch.Tensor:
    pop = actions.shape[0]
    z_0_exp = z_0.expand(pop, -1)
    z_g_exp = z_g.expand(pop, -1)
    trajectory = self._world_model.rollout(z_0_exp, actions)
    z_H = trajectory[:, -1, :]
    latent_cost = self._cost_fn(z_H, z_g_exp)
    inv_penalty = 1e-3 * (actions[..., 12] ** 2).sum(dim=(-2, -1))
    return latent_cost + inv_penalty
```
