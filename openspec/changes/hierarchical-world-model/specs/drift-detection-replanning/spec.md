## ADDED Requirements

### Requirement: Continuous drift detection
The system SHALL continuously compare each layer's predicted state embedding to the actual state embedding received from the layer below, and compute a drift scalar as the L2 distance.

#### Scenario: Drift computed every tick
- **WHEN** a layer receives an actual state embedding
- **THEN** the layer SHALL compute `drift = ||actual_s - predicted_s||` and store it

#### Scenario: Per-layer threshold
- **WHEN** the system is initialized
- **THEN** the drift threshold SHALL be `ε_n * sqrt(D_n)` with defaults `ε = [0.05, 0.10, 0.20, 0.30]` for L0, L1, L2, L3

### Requirement: Background replanning on drift
The system SHALL surface from the background loop when drift exceeds the per-layer threshold, and SHALL either adjust the target embedding, re-plan, or escalate.

#### Scenario: Gentle correction when drift is small
- **WHEN** drift exceeds the threshold by less than 2x
- **THEN** the layer SHALL adjust its current target embedding by gradient descent on the embedding-distance cost

#### Scenario: Re-plan when drift is large
- **WHEN** drift exceeds 2x the threshold
- **THEN** the layer SHALL re-plan from scratch by running its planner again

#### Scenario: Escalation to the layer above
- **WHEN** re-planning fails (no feasible target found within a budget)
- **THEN** the layer SHALL send an "escalation" signal to the layer above with the current state embedding

### Requirement: Drift distribution validation
The system SHALL log the drift distribution on a held-out trajectory and expose the per-layer p50, p90, p99 drift values as a CLI output of `wally-train-hierarchy`.

#### Scenario: Drift stats are part of the train-hierarchy output
- **WHEN** `wally-train-hierarchy` finishes
- **THEN** the CLI SHALL print the per-layer drift p50, p90, p99 and the corresponding threshold
