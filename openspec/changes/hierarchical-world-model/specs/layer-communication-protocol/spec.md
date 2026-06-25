## ADDED Requirements

### Requirement: Continuous-embedding messages between layers
The system SHALL communicate between layers using only continuous `Tensor[D]` embeddings. No strings, no symbolic task names, no discrete skill IDs shall appear in the runtime path.

#### Scenario: Top-down message
- **WHEN** a higher layer issues a plan to the layer below
- **THEN** the message SHALL be a single `target_embedding: Tensor[D_n]` with no accompanying labels

#### Scenario: Bottom-up message
- **WHEN** a lower layer reports to the layer above
- **THEN** the message SHALL be a single `state_embedding: Tensor[D_n]` plus a continuous drift scalar

#### Scenario: No vocabulary at runtime
- **WHEN** the agent loop is inspected at runtime
- **THEN** no `str` value representing a skill, task, or subgoal SHALL appear in any inter-layer message

### Requirement: Streaming state from lower to higher
The system SHALL stream state embeddings upward continuously, not on request. Every layer SHALL emit a state embedding every L0 tick.

#### Scenario: L0 produces a state embedding every tick
- **WHEN** L0 runs one planning step
- **THEN** L0 SHALL emit its current state embedding to L1's input queue

#### Scenario: L1 receives a state embedding asynchronously
- **WHEN** L1 is in its background loop
- **THEN** L1 SHALL read from its input queue non-blockingly and update its belief with the latest state

### Requirement: Target embedding from higher to lower
The system SHALL send a target embedding downward only when the higher layer has decided on a new plan, not continuously.

#### Scenario: L2 sends a new target to L1
- **WHEN** L2 finishes planning a new sub-plan
- **THEN** L2 SHALL send one `target_embedding: Tensor[D1]` to L1

#### Scenario: L1 holds the previous target until a new one arrives
- **WHEN** L1 is in its background loop and no new target has arrived
- **THEN** L1 SHALL continue using the last received target embedding
