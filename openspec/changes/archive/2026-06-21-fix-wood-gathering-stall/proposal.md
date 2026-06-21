## Why

The current MineStudio wood-gathering runs reach the target forest scene but the planner collapses into a "button-spam + frozen scene" local minimum instead of producing a goal-directed plan. Recorded episodes show two coupled symptoms:

- **Action profile**: every one of the 25 action dimensions has `|mean| ≈ 0.4` after discretization — the planner is jiggling every button simultaneously instead of committing to a coherent action sequence.
- **Scene activity**: ~50% of recorded steps have a frame-to-frame pixel diff of 0.0, meaning the player is effectively frozen in place even though the agent "is doing things".

This is a different failure mode from the documented "inventory open/close" minimum (the `inventory` action dim 12 stays low in these runs), so the existing `inventory_stall_penalty` does not help. We need planner-side cost shaping that breaks the new basin specifically, plus a regression test that catches it returning.

## What Changes

- Add a per-candidate **diversity penalty** in the planner cost path that rewards action sequences diverging from the population mean, so CEM elites cannot all collapse onto the same "press everything" sequence.
- Add a per-candidate **camera-stillness penalty** in the planner cost path that penalizes zero camera motion across the planned horizon, so the planner is forced to commit to camera movement and the agent visibly turns the view.
- Thread both penalties through `CEMConfig` with default values of `1e-3`, small enough to leave the goal-latent distance as the primary objective but strong enough to break the local minimum.
- Enable both penalties in the default MineStudio CEM planner built by `wally.agent.planner_factory.build_planner`.
- Add cost-function-level unit tests that prove each penalty biases the optimizer in the documented direction.
- Add a pytest regression test that replays a recorded wood-gathering trajectory (`ag-tests/run_wood_v2/episode_0.npz`) and asserts the action profile and scene-activity properties that the fix is meant to produce.

## Capabilities

### Modified Capabilities
- `goal-conditioned-planning`: planner cost shaping now includes a per-candidate diversity term and a camera-stillness term so the CEM optimizer avoids the button-spam + frozen-scene local minimum during MineStudio planning.

## Impact

- Planner implementation in `src/wally/planner/plan.py` (`_regularized_cost` and two new private penalty methods), `src/wally/planner/config.py` (two new `CEMConfig` fields with default `0.0` and validators), and `src/wally/agent/planner_factory.py` (default MineStudio CEM planner enables both penalties).
- New pytest regression `tests/test_wood_gathering_regression.py` that replays the recorded fixture and asserts behavior. This test does not need MineStudio; it is fast and CI-friendly.
- New cost-function unit tests in `tests/test_planner_cost_penalties.py` for the two new penalty terms.
- No API break: `CEMConfig.from_yaml` accepts the new fields, and the planner interface is unchanged.
- `inventory_stall_penalty` is preserved (still enabled at 0.25 in the default MineStudio planner) — it targets a different documented local minimum and is not in scope for this change.
