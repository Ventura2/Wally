## MODIFIED Requirements

### Requirement: Per-candidate diversity penalty
The goal-conditioned planner SHALL support a per-candidate diversity penalty that rewards action sequences diverging from the population mean, so CEM elites cannot all collapse onto the same action sequence. The penalty SHALL be applied to each candidate's cost before CEM elite selection. The default value SHALL be configurable via `CEMConfig.diversity_penalty`, and the default MineStudio planner factory SHALL set it to a small non-zero value so the optimizer is biased away from the button-spam local minimum.

#### Scenario: Low-diversity plan is disfavored
- **WHEN** two candidate action sequences have the same latent goal cost but one has every dimension near the population mean
- **THEN** the planner SHALL assign the low-diversity sequence a higher total cost and SHALL prefer the higher-diversity sequence

#### Scenario: Penalty can be disabled
- **WHEN** `CEMConfig.diversity_penalty` is set to zero
- **THEN** the planner SHALL fall back to the baseline latent-distance objective without adding any diversity bias

#### Scenario: Regularization is applied before CEM selection
- **WHEN** the planner evaluates a population of candidate plans
- **THEN** the diversity penalty SHALL contribute to each candidate's cost before elites are selected

### Requirement: Camera-stillness penalty
The goal-conditioned planner SHALL support a camera-stillness penalty that penalizes plans with zero camera motion (action dims 0 and 1, the continuous pitch and yaw dimensions), so the optimizer is forced to commit to camera movement and the agent visibly turns the view. The penalty SHALL be applied to each candidate's cost before CEM elite selection. The default value SHALL be configurable via `CEMConfig.camera_still_penalty`, and the default MineStudio planner factory SHALL set it to a small non-zero value.

#### Scenario: Stationary-camera plan is disfavored
- **WHEN** two candidate action sequences have the same latent goal cost but one has dims 0 and 1 near zero for every timestep
- **THEN** the planner SHALL assign the still-camera sequence a higher total cost and SHALL prefer the moving-camera sequence

#### Scenario: Penalty can be disabled
- **WHEN** `CEMConfig.camera_still_penalty` is set to zero
- **THEN** the planner SHALL fall back to the baseline latent-distance objective without adding any camera-stillness bias

### Requirement: Wood-gathering regression avoids the button-spam local minimum
The MineStudio wood-gathering regression fixture SHALL not exhibit the button-spam + frozen-scene failure mode when the diversity and camera-stillness penalties are enabled. The recorded trajectory SHALL have a per-dim `|mean|` bounded away from the spam pattern and SHALL have non-trivial scene activity across the episode.

#### Scenario: Action profile is not button-spam
- **WHEN** the wood-gathering regression trajectory is loaded
- **THEN** for every non-inventory action dimension the per-dim `|mean|` over the episode SHALL be bounded by 0.5 (button-spam signature: every dim at ~0.4 fails this)

#### Scenario: Camera is active
- **WHEN** the wood-gathering regression trajectory is loaded
- **THEN** the per-dim `|mean|` for dims 0 (camera pitch) and 1 (camera yaw) SHALL be at least 0.1 (frozen-camera signature: dims 0 and 1 near zero fails this)

#### Scenario: Scene is not frozen
- **WHEN** the wood-gathering regression trajectory is loaded
- **THEN** the mean frame-to-frame pixel diff SHALL be at least 5 and fewer than 20% of consecutive steps SHALL have a frame diff below 0.5 (frozen-scene signature: >50% of steps with diff 0.0 fails this)

#### Scenario: Inventory is never populated
- **WHEN** the wood-gathering regression trajectory is loaded (it stops in front of a tree without chopping)
- **THEN** the inventory SHALL never contain a non-`none` item (this confirms the test fixture matches the documented "approach the tree" milestone, not a spurious wood pickup)
