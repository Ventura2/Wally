## Context

Recorded MineStudio runs can reach the wood-gathering scene and then settle into a planner local minimum instead of sustaining a chopping sequence. The current workaround lives in `AgentLoop` and suppresses the inventory action after planning, which hides the symptom but keeps the real planner objective unchanged.

The affected stakeholders are the goal-conditioned planner, the episode loop, and the regression tests that guard the MineStudio control path.

## Goals / Non-Goals

**Goals:**

- Move the anti-stall behavior into the planner cost path.
- Keep the fix testable at the cost-function level, not only through full episodes.
- Remove the episode-loop workaround once the planner-side fix is covered by tests.
- Preserve existing CLI and environment interfaces.

**Non-Goals:**

- Retraining the world model or changing the encoder.
- Redesigning the MineStudio action vocabulary.
- Adding a new planner type or a new environment.
- Solving every future planning failure mode; this change targets the known inventory local minimum.

## Decisions

1. **Use planner-side action regularization instead of loop-level action masking**
   - The inventory mask in `AgentLoop` is a symptom fix, not a model of the problem.
   - Putting the penalty in the planner keeps the search objective honest and makes the behavior unit-testable with a deterministic cost function.
   - Alternative considered: keep the loop clamp and add more logging. Rejected because it preserves the hidden objective mismatch.

2. **Make the penalty small, configurable, and action-dimension specific**
   - The planner should still prioritize goal-latent distance.
   - The penalty only needs to bias the optimizer away from the known inventory-open local minimum.
   - Alternative considered: a large hard constraint or a generic sparse-action penalty. Rejected because it could suppress legitimate inventory use or overconstrain unrelated tasks.

3. **Keep the change compatible with existing planner interfaces**
   - The planner should still expose the same `plan(current_frame, goal_frame)` shape and return the same action tensor contract.
   - Any regularization tuning should be internal or config-driven, not a new mandatory CLI flag.
   - Alternative considered: introduce a separate planner mode for wood gathering. Rejected because it fragments behavior for what should be a default fix.

4. **Use regression tests at two levels**
   - Unit tests should verify the planner cost prefers the non-inventory candidate when latent costs are otherwise equal.
   - A smoke/regression test should exercise the known wood-gathering path to prevent the local minimum from returning.
   - Alternative considered: rely only on the end-to-end MineStudio episode. Rejected because it is too slow and fragile for TDD.

## Risks / Trade-offs

- [Risk] The inventory penalty may suppress legitimate inventory actions in some future task. → Keep the coefficient small and scoped to the MineStudio planner configuration, and verify the regression does not add a broad action ban.
- [Risk] The fix may only address one local minimum while other planning loops remain. → Add a focused regression and leave the regularizer configurable so future failures can be tuned without interface changes.
- [Risk] Removing the loop workaround before the planner fix is fully covered could reintroduce the stall. → Land the tests first, then remove the clamp only after the planner-side behavior is proven.

## Migration Plan

1. Add planner-cost regression tests that fail with the current loop-level workaround.
2. Implement planner-side inventory regularization.
3. Remove the temporary `AgentLoop` inventory clamp.
4. Run the focused unit tests and the wood-gathering smoke path.
5. If the new planner penalty hurts unrelated behavior, roll back the regularizer coefficient before reverting the interface.

## Open Questions

- What default penalty coefficient is small enough to avoid overfitting but strong enough to break the known inventory loop?
- Should the penalty be exposed as a config field immediately, or kept internal until another local minimum appears?
