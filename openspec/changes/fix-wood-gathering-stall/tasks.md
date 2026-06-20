## 1. Regression tests first

- [ ] 1.1 Add a planner-level unit test that compares two otherwise equal candidate action sequences and asserts the one with repeated inventory toggles has higher total cost.
- [ ] 1.2 Add a planner-level unit test that verifies the inventory-stall penalty can be disabled and that zero penalty preserves the baseline latent-distance objective.
- [ ] 1.3 Add a wood-gathering regression test that exercises the recorded goal fixture and asserts the planner no longer depends on the inventory local minimum to complete the episode.

## 2. Planner-side fix

- [ ] 2.1 Add MineStudio inventory-stall regularization to the goal-conditioned planner cost path so the penalty is applied before CEM elite selection.
- [ ] 2.2 Thread the regularization coefficient through the planner configuration in a way that keeps the default behavior enabled without changing CLI call sites.
- [ ] 2.3 Remove the temporary `AgentLoop` inventory-action clamp once the planner-side fix is active.

## 3. Verification

- [ ] 3.1 Run the focused planner and agent-loop tests and fix any failures introduced by the new cost shaping.
- [ ] 3.2 Run the wood-gathering smoke path against the known goal fixture and confirm the run no longer stalls on the inventory local minimum.
