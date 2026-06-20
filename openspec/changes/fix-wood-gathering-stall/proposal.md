## Why

The current MineStudio wood-gathering runs can reach the target forest scene but then stall in a local minimum instead of sustaining a chopping sequence long enough to obtain wood. Recent recorded episodes showed the agent spending the full timeout without a verifiable wood pickup, which means the planner is optimizing toward a visually plausible state but not the actual task outcome.

## What Changes

- Move the anti-stall behavior into the planner cost path instead of relying on a loop-level action mask.
- Add a small, configurable action regularization term that penalizes inventory-toggle behavior during goal-conditioned planning.
- Remove the temporary `AgentLoop` inventory-action clamp once planner-side regularization is in place.
- Add regression tests that reproduce the stall pattern and verify the planner no longer converges on it.
- Keep trajectory recording available for future diagnosis, but do not make it part of the control fix.

## Capabilities

### Modified Capabilities
- `goal-conditioned-planning`: planner cost shaping now includes a MineStudio-specific inventory-stall penalty so the planner avoids the repeated inventory local minimum during wood-gathering episodes.

## Impact

- Planner implementation in `src/wally/planner/plan.py` and related cost-shaping code.
- `AgentLoop` action selection path in `src/agent/loop.py` to remove the temporary workaround once tests prove the planner fix is sufficient.
- Unit and smoke tests covering the stall regression, planner behavior, and the wood-gathering episode path.
- No API break is expected; this is a behavior fix for existing goal-conditioned planning.
