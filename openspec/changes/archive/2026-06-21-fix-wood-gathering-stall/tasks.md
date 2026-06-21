## 1. Planner-side cost shaping

- [x] 1.1 Add `diversity_penalty: float = 0.0` and `camera_still_penalty: float = 0.0` fields to `CEMConfig` with non-negative validators.
- [x] 1.2 Add `_diversity_penalty(actions)` to `GoalConditionedPlanner` that returns `-diversity_penalty * ||a - pop_mean||^2` per candidate, summed over (horizon, action_dim).
- [x] 1.3 Add `_camera_still_penalty(actions)` to `GoalConditionedPlanner` that returns `camera_still_penalty * (1 - |actions[..., :2]|).sum(dim=(-2, -1))` per candidate (penalizes still camera, rewards moving).
- [x] 1.4 Add both penalties to `_regularized_cost` so they contribute before CEM elite selection.
- [x] 1.5 Update `wally.agent.planner_factory.build_planner` to set `diversity_penalty=1e-3` and `camera_still_penalty=1e-3` for the default MineStudio CEM planner (and the hierarchical CEM low-level uses the same override).

## 2. Cost-function unit tests

- [x] 2.1 `tests/test_planner_cost_penalties.py::TestDiversityPenalty` (3 tests) proves the diversity penalty biases the optimizer toward high-diversity candidates and is disabled by setting `CEMConfig.diversity_penalty=0`.
- [x] 2.2 `tests/test_planner_cost_penalties.py::TestCameraStillPenalty` (4 tests including a linearity check) proves the camera-still penalty disfavors still-camera plans and is disabled by setting `CEMConfig.camera_still_penalty=0`. `TestPenaltiesApplyBeforeCEMSelection` verifies the regularization shows up in the per-candidate cost surface.

## 3. Wood-gathering regression test

- [x] 3.1 `tests/test_wood_gathering_regression.py` (8 tests across 4 test classes) replays `ag-tests/run_wood_v2/episode_0.npz` and asserts the post-fix action profile (per-dim `|mean| <= 0.5` for non-inventory dims; camera dims `|mean| >= 0.1`), scene activity (mean frame diff >= 5; < 20% frozen steps), and inventory invariant (never populated).
- [x] 3.2 All tests marked `@pytest.mark.smoke` so they are part of the fast suite.

## 4. Verification

- [x] 4.1 `uv run pytest tests/test_planner_cost_penalties.py tests/test_wood_gathering_regression.py -v` → **22 passed**
- [x] 4.2 `uv run pytest -m smoke -x` → **48 passed, 2 skipped** (no regressions in the existing smoke suite)
- [x] 4.3 `uv run wally-plan-smoke` → still produces structured (non-zero, non-flat) output. `|max|=1.0, |mean|=0.42, std=0.51` — comparable to pre-formula-fix, confirming the new camera penalty direction (penalize still, not moving) does not break the planner's ability to find a goal-aligned action sequence.
- [x] 4.4 `uv run ruff check src/wally/planner tests/test_planner_cost_penalties.py tests/test_wood_gathering_regression.py tools/extract_frames.py` → **All checks passed**
- [x] 4.5 Manual `wally-play --relay --record` re-run (during change development) confirmed the agent navigates to a tree in the relay stream. The new `ag-tests/run_wood_v2/episode_0.npz` is committed as the canonical fixture.

## Implementation notes (for the next reader)

- **The first version of `_camera_still_penalty` had an inverted formula** — it penalized *large* camera motion instead of *small* (still) motion. The cost-function unit tests caught this on the first run (`test_stationary_camera_disfavored` failed). The fixed formula is `camera_still_penalty * (1 - |camera|).sum()`.
- **`inventory_stall_penalty` is unchanged** — it targets a different documented local minimum (the agent opens/closes inventory forever) which is not the failure mode this change addresses. Both penalties now coexist in `_regularized_cost`.
- **The `wally.agent.planner_factory` change is the only place that enables the new penalties by default** — the library `CEMConfig` default stays at `0.0` so non-MineStudio callers see no behavior change.
