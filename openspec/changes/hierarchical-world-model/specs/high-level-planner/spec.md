## MODIFIED Requirements

### Requirement: Subgoal conditioning for low-level planner
The system SHALL convert high-level subgoal latents into goal embeddings that can be consumed by the low-level planner as `target_embedding` arguments. The conversion SHALL be a learned linear projection from the high-level embedding space to the L0 latent space, trained jointly with the hierarchy.

#### Scenario: Provide subgoal targets as embeddings
- **WHEN** the high-level planner produces a subgoal sequence
- **THEN** the system SHALL convert each subgoal latent into a `target_embedding: Tensor[D0]` via a learned linear projection, and pass it to the L0 planner as the `target_embedding` argument

#### Scenario: Sequential subgoal execution
- **WHEN** executing a plan with multiple subgoals
- **THEN** the system executes subgoals in order, advancing to the next subgoal only after the drift between predicted and actual state embeddings drops below the per-layer threshold (or after a per-subgoal timeout, whichever comes first)

### Requirement: Replanning on drift or failure
The system SHALL detect two conditions and trigger replanning: (1) the drift between the high-level world's predicted state embedding and the actual state embedding from below exceeds the per-layer threshold; (2) the low-level planner fails to reach a subgoal within a configurable timeout.

#### Scenario: Drift-based replanning
- **WHEN** the L1 layer's predicted state embedding diverges from the actual state embedding by more than `threshold_L1`
- **THEN** the high-level planner SHALL re-issue a new subgoal sequence starting from the current actual state embedding

#### Scenario: Subgoal timeout detection
- **WHEN** the low-level planner exceeds the maximum number of steps for a subgoal without reaching it
- **THEN** the system flags the subgoal as failed

#### Scenario: Request new subgoal on failure
- **WHEN** a subgoal is flagged as failed or drift exceeds the threshold
- **THEN** the system requests a new subgoal sequence from the high-level planner starting from the current state
