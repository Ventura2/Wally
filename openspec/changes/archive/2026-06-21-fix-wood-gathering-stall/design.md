## Context

Recorded MineStudio wood-gathering runs successfully reach the forest scene but the CEM planner collapses into a "button-spam + frozen scene" local minimum. Two coupled symptoms appear in `ag-tests/run_wood/episode_0.npz` (the v1 baseline run with no planner-side shaping beyond `inventory_stall_penalty`):

1. **Action distribution**: every one of the 25 action dimensions has `|mean| ≈ 0.4` after discretization. The planner is jiggling every button simultaneously instead of committing to a coherent action sequence.
2. **Scene activity**: ~50% of recorded steps have a frame-to-frame pixel diff of 0.0, and the rest are sparse; the player is effectively frozen in place even though the agent "is doing things".

The existing `inventory_stall_penalty` in `CEMConfig` does not help because the `inventory` action (dim 12) is only active on 14/1000 steps in the failing run — the documented "inventory open/close" basin is not the basin the agent is actually in. The affected stakeholders are the goal-conditioned planner, the MineStudio agent loop, and the regression tests that should guard the MineStudio control path against this return.

## Goals / Non-Goals

**Goals:**

- Break the "button-spam + frozen scene" local minimum via planner-side cost shaping.
- Keep the fix testable at the cost-function level, not only through full MineStudio episodes.
- Provide a fast pytest regression that locks in the post-fix behavior so this basin cannot return unnoticed.
- Preserve the existing `inventory_stall_penalty` (different basin, still useful).
- Keep CLI and planner interfaces unchanged.

**Non-Goals:**

- Retraining the world model or changing the encoder.
- Redesigning the MineStudio action vocabulary.
- Adding a new planner type or environment.
- Solving every future planning failure mode; this change targets the specific button-spam + frozen-scene basin observed in `ag-tests/run_wood/episode_0.npz`.
- Removing the existing `inventory_stall_penalty`.

## Decisions

1. **Per-candidate diversity penalty instead of an action-distribution prior on the final plan**
   - The CEM optimizer sees the full population, so the diversity penalty is applied per candidate as `-beta * ||a - pop_mean||^2` summed over (horizon, action_dim). Candidates far from the mean get lower cost, so elites cannot all collapse onto the same sequence.
   - Alternative considered: a penalty on the final returned action only. Rejected because the CEM selection happens before the final action is chosen, so a post-hoc penalty does not influence which candidates survive.
   - Alternative considered: an entropy bonus on the action distribution. Rejected because CEM samples from a fixed Gaussian, not a categorical, so the entropy would be a function of the sampling distribution rather than the chosen actions.

2. **Camera-stillness penalty on dims 0 and 1 only**
   - The "frozen scene" symptom is dominated by zero camera motion; dims 0 (pitch) and 1 (yaw) are the only continuous-action dims and they cover camera movement.
   - Alternative considered: a generic "low-magnitude action" penalty. Rejected because the button-spam pattern has high magnitude on every non-camera dim, so a generic penalty would not target the right dims.

3. **Both penalties added to `CEMConfig` with default `0.0` and enabled at `1e-3` in the default MineStudio CEM planner**
   - The planner interface is unchanged. The default of 0 means library users who construct `CEMConfig()` directly see no behavior change. The MineStudio agent factory in `wally.agent.planner_factory` is the canonical place to set MineStudio-specific defaults.
   - Alternative considered: hard-coding the values in `plan.py`. Rejected because it makes A/B testing the coefficients harder and removes the ability to disable the penalties per task.

4. **Pytest regression based on the recorded trajectory, not a live MineStudio episode**
   - The `ag-tests/run_wood_v2/episode_0.npz` trajectory is the post-fix fixture; the test asserts the action profile and scene activity properties that distinguish a healthy run from the v1 failure.
   - The MineStudio live test is too slow and too fragile for CI, and we already have the canonical fixture from the manual run.
   - Alternative considered: a pytest test that spins up a MineStudio env. Rejected because the Linux-only MineStudio dependency would break Windows-native CI and the existing `tests/test_play_cli.py::TestRelayEndToEnd` already covers the live path with a mock env.

## Risks / Trade-offs

- [Risk] The diversity penalty may push the optimizer toward non-coherent action sequences. → Coefficient is `1e-3` (small relative to the typical latent-distance cost of 0.1–1.0) and is verified by the cost-function unit test.
- [Risk] The camera-stillness penalty may suppress legitimate stationary-camera behaviors (e.g. precision chopping). → Coefficient is `1e-3`; the wood-gathering regression shows the agent is still able to stop in front of a tree without penalty blowup.
- [Risk] The pytest regression locks in behavior of a specific checkpoint, not the planner itself. → The cost-function unit tests cover the planner in isolation; the trajectory test is a behavior-level safety net, not a model of correctness.

## Migration Plan

1. Add `diversity_penalty` and `camera_still_penalty` to `CEMConfig` with default `0.0`.
2. Add `_diversity_penalty` and `_camera_still_penalty` methods to `GoalConditionedPlanner` and wire them into `_regularized_cost`.
3. Update `wally.agent.planner_factory.build_planner` to set both to `1e-3` for the default MineStudio CEM planner.
4. Add cost-function unit tests in `tests/test_planner_cost_penalties.py` that prove each penalty biases the optimizer in the documented direction.
5. Add `tests/test_wood_gathering_regression.py` that replays `ag-tests/run_wood_v2/episode_0.npz` and asserts the post-fix action profile and scene-activity properties.
6. Run the focused tests and the full smoke suite. Manually re-run `wally-play` against the wood-gathering goal frame and verify the agent still navigates to a tree.
7. If the new penalties regress an unrelated task, lower the coefficient or set it to 0 in the task-specific planner config before shipping.

## Open Questions

- Should the diversity penalty be exposed as a CLI flag on `wally-play` for A/B testing, or is the planner-factory override sufficient?
- Is the `ag-tests/run_wood_v2/episode_0.npz` fixture stable enough to commit, or should it be regenerated each release?
