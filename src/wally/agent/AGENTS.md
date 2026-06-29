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

All four planners (`cem`, `gradient`, `hierarchical`, `hierarchical-embedding`) currently converge on opening/closing the inventory forever when the goal latent is far from the agent's current state. The world model scores "inventory open" as a near-constant latent that's marginally close to many goal latents, so the CEM elites latch onto it. Symptom: the relay stream shows the inventory UI opening and closing in a tight loop.

**For offline debugging**, the recommended way to visualize this is `tools/extract_anomalies.py <npz>`, which produces a single labeled contact sheet (PNG + JSON) with the inv-spam panel pinned to the exact step range. Unlike the live relay (which streams every frame and drowns the anomaly in noise), the contact sheet shows a 5-frame window around the moment the agent got stuck, plus the brightness, camera-shake, and cost-spike panels for context. Run it on a `ag-tests/*/episode_0.npz` that has `inv > 0.5` > 50 steps and you get the inventory-UI-overlay frame at the centre of the strip.**The one-line demo workaround is now active by default** in `src/wally/agent/loop.py::AgentLoop.run_episode` (after `plan_actions[action_index]` is selected, before `env.step`):

```python
action = plan_actions[action_index]
if action.dim() == 1 and action.shape[-1] > 12:
    action = action.clone()
    action[12] = 0.0
```

With this in place, runs against the 10k L0 + 5k L1 + tree-frame
`g1` show **zero** inventory-spam steps (`inv > 0.5` = 0). Confirmed
in `ag-tests/run_wood10k_l1_5k_tree_g1/episode_0.npz`.

Production fix (still TODO): add the same penalty to the CEM cost
function in `src/wally/planner/plan.py` so the planner *learns* to
avoid the action, instead of just masking it at the loop boundary:

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

## Hierarchical planner wiring

`wally-play --planner hierarchical-embedding` requires three CLI args
to actually reach `build_planner`:
- `--hierarchy-checkpoint PATH`
- `--layer-depth N` (1 = L0+L1, 2 = L0+L1+L2, 3 = L0+L1+L2+L3)
- `--target-embedding PATH` (the `g1`/goal tensor, in place of `--goal-frame`)

`src/wally/agent/play.py` parses these but **must forward them as
keyword arguments** to `build_planner`; passing them positionally drops
them on the floor. The factory also needs the `lowest_encoder` for
the planner to use the 64-dim L1 path (not the 192-dim L0 fallback);
without it the L1 JEPA errors with `mat1 and mat2 shapes cannot be
multiplied (1x192 and 64x128)`. Both fixes are in
`src/wally/agent/planner_factory.py::build_planner`.

L2 (and deeper) is not runnable yet — see
`src/wally/hierarchy/AGENTS.md#l2-path-is-not-viable-yet`. Passing
`--layer-depth 2` against the L1-only checkpoint will load the L1
weights for the L2 slot and produce random targets.

## Goal embeddings (`g1`, `g2`, `g3`)

The `g1` tensor is a 64-dim vector saved with `torch.save({"g": g},
"checkpoints/g1_get_wood.pt")`. The agent loads it via
`--target-embedding PATH`. The contents matter: a centroid of random
training chunks will pull the agent toward whatever the dataset mean
looks like (e.g. a "water + seagrass" centroid leads the agent into
the nearest lake; see `ag-tests/run_wood10k_l1_5k/episode_0.npz` where
`mine_block` totals = 1213 × `tall_seagrass`).

`logs/make_g1_tree.py` is the recipe that worked: scan the shards,
score each frame for sky + green leaves + brown trunk, take the
centroid of the top 32 frames, encode through the trained L1Encoder.
Result: `g1` with `norm ≈ 2.2` and the L1 drift drops to ~3-4 (vs
~5-7 with a random centroid). For `g2` (L2) and `g3` (L3) the recipe
will be the same shape but with the appropriate encoder and frame
target ("have wood" / "have shelter").
