## ADDED Requirements

### Requirement: Inventory-stall regularization
The goal-conditioned planner SHALL support an action regularization term that penalizes repeated inventory toggling during MineStudio planning. The regularization SHALL be applied per candidate action sequence before CEM elite selection, and the default MineStudio planner configuration SHALL enable a small non-zero penalty for the inventory action dimension so the planner does not converge on the known inventory-open local minimum.

#### Scenario: Inventory-stalling plan is disfavored
- **WHEN** two candidate action sequences have the same latent goal cost but one toggles the inventory action more often than the other
- **THEN** the planner SHALL assign the inventory-heavy sequence a higher total cost and SHALL prefer the lower-inventory sequence

#### Scenario: Penalty can be disabled
- **WHEN** the inventory-stall penalty is configured to zero
- **THEN** the planner SHALL fall back to the baseline latent-distance objective without adding any inventory-specific bias

#### Scenario: Regularization is applied before CEM selection
- **WHEN** the planner evaluates a population of candidate plans
- **THEN** the inventory-stall regularization SHALL contribute to each candidate's cost before elites are selected

### Requirement: Wood-gathering regression avoids the known local minimum
The MineStudio wood-gathering regression fixture SHALL not spend the entire episode on the known inventory-open local minimum when the inventory-stall regularization is enabled. The planner output used by the episode loop SHALL remain free of the temporary loop-level inventory clamp once the planner-side regularization is in place.

#### Scenario: Wood regression no longer depends on the loop clamp
- **WHEN** the wood-gathering regression test runs with the planner-side inventory regularization enabled
- **THEN** the episode path SHALL not rely on `AgentLoop` mutating the planned action to zero the inventory dimension

#### Scenario: Wood regression remains executable
- **WHEN** the wood-gathering smoke test is executed against the recorded goal fixture
- **THEN** the planner SHALL produce a finite action sequence and the run SHALL complete without triggering the inventory-open loop
